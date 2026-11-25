"""Two-stage semantic search on top of the live INSPIRE API.

INSPIRE's search engine joins plain words with AND, so a natural-language question
often returns nothing. This module:

1. candidate generation: turns the question into a relaxed INSPIRE query (content words
   joined with OR, plus the strict AND query when it has hits) and takes the top `pool` records;
2. reranking: scores candidates against the question with the best reranker available: a
   cross-encoder that reads the question with each title and abstract, then dense embeddings
   (both need the optional `fastembed` dependency), then BM25;
3. fusion: only the weaker rerankers (embeddings, BM25) are combined with INSPIRE's own order by
   reciprocal rank fusion; the cross-encoder order is used as is, which is the configuration
   measured in eval/ (see docs/evaluation.md).

No index has to be hosted: the database stays at INSPIRE, only the candidates are embedded,
and their vectors are cached on disk by record id.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import re
import sqlite3
import threading
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from .client import InspireClient, InspireError, Paper

STOPWORDS = frozenset(
    """a about after all also an and any are as at be been by can could did do does for from
    had has have how i if in into is it its me my of on or our paper papers please show shows
    so some than that the their them then there these they this those to under up was we were
    what when where which who whom why will with would you your find found give tell look looking
    article articles work works study studies original first new approach result results using
    based between one two""".split()
)
INSPIRE_RESERVED = frozenset({"and", "or", "not", "to", "a", "t", "j", "cn", "date", "topcite"})
TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-]*[A-Za-z0-9]|[A-Za-z]{2,}|\d+(?:\.\d+)*")
MAX_TERMS = 12
RRF_K = 60


def content_terms(question: str, max_terms: int = MAX_TERMS) -> list[str]:
    """Distinctive words of a question, in order, without stopwords or duplicates."""
    seen: dict[str, None] = {}
    for tok in TOKEN.findall(question):
        low = tok.lower().strip("-")
        if len(low) < 2 or low in STOPWORDS or low in INSPIRE_RESERVED or low in seen:
            continue
        seen[low] = None
    return list(seen)[:max_terms]


def phrase_terms(question: str) -> list[str]:
    """Pairs of content words adjacent in the question (no stopword in between), e.g. 'extra dimension'."""
    out: dict[str, None] = {}
    prev = None
    for tok in TOKEN.findall(question):
        low = tok.lower().strip("-")
        if len(low) < 2 or low in STOPWORDS or low in INSPIRE_RESERVED:
            prev = None
            continue
        if prev:
            out[f"{prev} {low}"] = None
        prev = low
    return list(out)


def _quote(term: str, always: bool = False) -> str:
    # Hyphens and dots confuse the INSPIRE parser ("d-brane" reads as the date operator "d"), so quote them.
    return f'"{term}"' if always or not term.isalnum() else term


def relaxed_query(question: str, quote_all: bool = False) -> str:
    return " or ".join(_quote(t, quote_all) for t in content_terms(question))


def doc_text(p: Paper, max_chars: int = 1200) -> str:
    return f"{p.title}. {p.abstract or ''}"[:max_chars]


class Reranker(Protocol):
    name: str
    # Whether to fuse the reranker's order with INSPIRE's by RRF. Measured per reranker in eval/:
    # fusion helps the weaker rerankers and hurts the cross-encoder.
    fuse_with_inspire: bool

    def scores(self, query: str, docs: Sequence[Paper]) -> list[float]: ...


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) > 1]


class BM25Reranker:
    """Okapi BM25 over the candidate set. Cheap and dependency free, but IDF on a small pool is noisy."""

    name = "bm25"
    fuse_with_inspire = True

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b

    def scores(self, query: str, docs: Sequence[Paper]) -> list[float]:
        tfs = [Counter(_tokens(doc_text(d))) for d in docs]
        if not tfs:
            return []
        lens = [sum(tf.values()) for tf in tfs]
        avg = sum(lens) / len(lens) or 1.0
        n = len(docs)
        df = Counter(t for tf in tfs for t in tf)
        out = []
        for tf, length in zip(tfs, lens, strict=True):
            s = 0.0
            for t in set(_tokens(query)):
                if t in tf:
                    idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                    s += idf * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * length / avg))
            out.append(s)
        return out


class DenseReranker:
    """Cosine similarity of bge embeddings (fastembed, ONNX on CPU) with an on-disk vector cache."""

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    fuse_with_inspire = True

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", cache_path: Path | None = None) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError("dense reranking needs the 'semantic' extra: pip install inspire-mcp[semantic]") from exc
        import numpy as np

        self._np = np
        self.model_name = model
        self.name = f"dense:{model.split('/')[-1]}"
        self._model = TextEmbedding(model)
        cache_path = cache_path or Path(os.environ.get("INSPIRE_MCP_CACHE", Path.home() / ".cache" / "inspire-mcp"))
        cache_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  # one inference at a time, and sqlite writes from one thread at a time
        self._db = sqlite3.connect(cache_path / "embeddings.sqlite", check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")

    def _key(self, p: Paper) -> str:
        digest = hashlib.sha1(doc_text(p).encode()).hexdigest()[:16]
        return f"{self.model_name}|{p.recid}|{digest}"

    def _embed_docs(self, docs: Sequence[Paper]):
        np = self._np
        keys = [self._key(d) for d in docs]
        cached: dict[str, bytes] = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i : i + 500]
            marks = ",".join("?" * len(chunk))
            cached.update(self._db.execute(f"SELECT key, vec FROM emb WHERE key IN ({marks})", chunk).fetchall())
        missing = [i for i, k in enumerate(keys) if k not in cached]
        if missing:
            vecs = list(self._model.embed([doc_text(docs[i]) for i in missing], batch_size=32))
            rows = [(keys[i], np.asarray(v, dtype=np.float32).tobytes()) for i, v in zip(missing, vecs, strict=True)]
            self._db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)", rows)
            self._db.commit()
            cached.update(rows)
        return np.stack([np.frombuffer(cached[k], dtype=np.float32) for k in keys])

    def scores(self, query: str, docs: Sequence[Paper]) -> list[float]:
        if not docs:
            return []
        np = self._np
        with self._lock:
            mat = self._embed_docs(docs)
            q = np.asarray(next(iter(self._model.embed([self.QUERY_PREFIX + query]))), dtype=np.float32)
        mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
        q /= np.linalg.norm(q)
        return (mat @ q).tolist()


DEFAULT_CROSS_ENCODER = "Xenova/ms-marco-MiniLM-L-6-v2"


def load_cross_encoder(text_cross_encoder: type, model: str):
    """A fastembed cross-encoder: a model fastembed ships, a directory with an ONNX export
    (config and tokenizer files, onnx/model.onnx), or a Hugging Face repository laid out the same way."""
    supported = {m["model"] for m in text_cross_encoder.list_supported_models()}
    if model in supported:
        return text_cross_encoder(model)
    from fastembed.common.model_description import ModelSource

    path = Path(model).expanduser()
    name = f"local/{path.resolve().name}" if path.is_dir() else model
    if name not in supported:  # registering a name twice raises
        text_cross_encoder.add_custom_model(name, sources=ModelSource(hf=name), model_file="onnx/model.onnx")
    if path.is_dir():
        return text_cross_encoder(name, specific_model_path=str(path))
    return text_cross_encoder(name)


class CrossEncoderReranker:
    """Scores (question, abstract) pairs jointly with a cross-encoder: slower per pair than comparing
    cached embeddings, but much better at telling apart abstracts on the same topic."""

    fuse_with_inspire = False

    def __init__(self, model: str = DEFAULT_CROSS_ENCODER) -> None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError("cross-encoder reranking needs the 'semantic' extra") from exc
        self.model_name = model
        self.name = f"cross-encoder:{model.rstrip('/').split('/')[-1]}"
        self._model = load_cross_encoder(TextCrossEncoder, model)
        self._lock = threading.Lock()

    def scores(self, query: str, docs: Sequence[Paper]) -> list[float]:
        if not docs:
            return []
        with self._lock:
            return [float(s) for s in self._model.rerank(query, [doc_text(d) for d in docs], batch_size=16)]


def default_reranker() -> Reranker:
    """Best reranker available: cross-encoder, then dense embeddings, then BM25 (see docs/evaluation.md)."""
    for factory in (CrossEncoderReranker, DenseReranker):
        try:
            return factory()
        except RuntimeError:
            continue
    return BM25Reranker()


def rrf(*rankings: Sequence[int], k: int = RRF_K) -> list[int]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for pos, recid in enumerate(ranking):
            fused[recid] = fused.get(recid, 0.0) + 1.0 / (k + pos + 1)
    return sorted(fused, key=lambda recid: -fused[recid])


@dataclass
class SemanticResult:
    papers: list[Paper]
    scores: dict[int, float]
    inspire_query: str
    pool_size: int
    strict_hits: int
    reranker: str
    fusion: str


Strategy = Literal["or", "multi"]


def interleave(*lists: Sequence[Paper]) -> list[Paper]:
    """Round-robin merge without duplicates, so each source keeps its best results in the pool."""
    merged: dict[int, Paper] = {}
    for i in range(max((len(x) for x in lists), default=0)):
        for x in lists:
            if i < len(x):
                merged.setdefault(x[i].recid, x[i])
    return list(merged.values())


async def _search_or_empty(client: InspireClient, query: str, size: int) -> list[Paper]:
    if not query or size <= 0:
        return []
    try:
        _, papers = await client.search_literature(query, size=size)
        return papers
    except InspireError:
        return []


async def candidates(
    client: InspireClient,
    question: str,
    pool: int = 50,
    influential: int = 0,
    min_citations: int = 250,
    strategy: Strategy = "or",
) -> tuple[list[Paper], str, int]:
    """Candidate pool from the live INSPIRE index.

    Strict AND results first (high precision when they exist), then the relaxed OR query by INSPIRE
    relevance. Strategy "multi" interleaves the OR results with quoted adjacent-word phrases and a
    title-only OR query, three reformulations that miss different papers. With `influential`, that many
    candidates come from the relaxed query restricted to well cited papers, a prior for questions about
    foundational work that relevance ranking alone tends to bury.
    """

    async def strict_search() -> tuple[int, list[Paper]]:
        try:
            return await client.search_literature(question, size=min(pool, 25))
        except InspireError:  # the raw question is not valid INSPIRE syntax, e.g. unbalanced quotes
            return 0, []

    relaxed = relaxed_query(question)
    # Independent queries: run them together, the client keeps to INSPIRE's rate limit.
    (strict_total, strict), loose = await asyncio.gather(strict_search(), _search_or_empty(client, relaxed, pool))
    if relaxed and not loose:
        relaxed = relaxed_query(question, quote_all=True)
        loose = await _search_or_empty(client, relaxed, pool)
    if strategy == "multi":
        phrases = " or ".join(f'"{p}"' for p in phrase_terms(question))
        titles = " or ".join(f"t {_quote(t)}" for t in content_terms(question))
        loose = interleave(
            loose, await _search_or_empty(client, phrases, pool // 2), await _search_or_empty(client, titles, pool // 2)
        )
    cited = await _search_or_empty(client, f"({relaxed}) and topcite {min_citations}+", influential) if relaxed else []
    head = pool - len(cited)
    merged: dict[int, Paper] = {}
    for p in [*strict, *loose[:head], *cited, *loose[head:]]:
        merged.setdefault(p.recid, p)
    return list(merged.values())[:pool], relaxed, strict_total


async def semantic_search(
    client: InspireClient,
    question: str,
    k: int = 10,
    pool: int = 50,
    reranker: Reranker | None = None,
    fuse_with_inspire: bool | None = None,
    influential: int = 0,
    strategy: Strategy = "or",
) -> SemanticResult:
    reranker = reranker or BM25Reranker()
    if fuse_with_inspire is None:
        fuse_with_inspire = getattr(reranker, "fuse_with_inspire", True)
    docs, relaxed, strict_total = await candidates(client, question, pool, influential=influential, strategy=strategy)
    # Models are CPU-bound and synchronous: run them in a worker thread so the event loop keeps serving.
    scores = await asyncio.to_thread(reranker.scores, question, docs) if docs else []
    by_score = [d.recid for d, _ in sorted(zip(docs, scores, strict=True), key=lambda x: -x[1])]
    inspire_order = [d.recid for d in docs]
    order = rrf(by_score, inspire_order) if fuse_with_inspire else by_score
    lookup = {d.recid: d for d in docs}
    return SemanticResult(
        papers=[lookup[r] for r in order[:k]],
        scores={d.recid: round(s, 4) for d, s in zip(docs, scores, strict=True)},
        inspire_query=relaxed,
        pool_size=len(docs),
        strict_hits=strict_total,
        reranker=reranker.name,
        fusion="rrf(reranker, inspire)" if fuse_with_inspire else "reranker only",
    )


async def similar_papers(
    client: InspireClient, paper: Paper, k: int = 10, pool: int = 50, reranker: Reranker | None = None
) -> SemanticResult:
    """Papers on the same subject: candidates from the title and keywords, ranked against the abstract."""
    seed = " ".join([paper.title, *paper.keywords[:4]])
    query_text = f"{paper.title}. {paper.abstract or ''}"[:600]
    reranker = reranker or BM25Reranker()
    docs, relaxed, strict_total = await candidates(client, seed, pool + 1)
    docs = [d for d in docs if d.recid != paper.recid][:pool]
    scores = await asyncio.to_thread(reranker.scores, query_text, docs) if docs else []
    by_score = [d.recid for d, _ in sorted(zip(docs, scores, strict=True), key=lambda x: -x[1])]
    fuse = getattr(reranker, "fuse_with_inspire", True)
    order = rrf(by_score, [d.recid for d in docs]) if fuse else by_score
    lookup = {d.recid: d for d in docs}
    return SemanticResult(
        papers=[lookup[r] for r in order[:k]],
        scores={d.recid: round(s, 4) for d, s in zip(docs, scores, strict=True)},
        inspire_query=relaxed,
        pool_size=len(docs),
        strict_hits=strict_total,
        reranker=reranker.name,
        fusion="rrf(reranker, inspire)" if fuse else "reranker only",
    )
