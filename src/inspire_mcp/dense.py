"""Optional local semantic index of INSPIRE, for services: candidates that keyword search misses.

INSPIRE's search matches words. A question describing a paper in other words often leaves the paper out of
the candidates, and no reranker can then find it. A dense index of the records' titles and abstracts
returns the nearest records by meaning; semantic_search takes half of its candidates from it and half from
INSPIRE (`candidates(..., dense=...)`). On agent-written development questions this raised the share of
targets among the 50 candidates from 0.39 to 0.50 for citation-like descriptions and from 0.61 to 0.82 for
famous papers (docs/evaluation.md).

The index holds vectors and record ids only, no text: the records found are fetched from INSPIRE live, and
papers newer than the index still come from the live keyword search. Only records whose abstract comes from
arXiv are indexed (arXiv metadata is CC0). Layout of an index directory:

    index.json    model, query prefix, record count, newest earliest-date covered
    vectors.npy   float16, one normalized row per record (loaded as float32: ~1.3 GB for 836k x 384)
    recids.npy    int64 INSPIRE record ids, same order

    inspire-mcp index build DIR            # every record, a few hours on a CPU
    inspire-mcp index update DIR           # records since the newest date in the index
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .client import InspireClient

log = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# bge retrieval models expect this before a query (not before a passage); fastembed does not add it
QUERY_PREFIXES = {"BAAI/bge": "Represent this sentence for searching relevant passages: "}
HARVEST_QUERY = "abstracts.source:arXiv"
HARVEST_FIELDS = "control_number,titles.title,abstracts,earliest_date"
PAGE, RESULT_LIMIT = 1000, 10_000
FIRST_DATE = date(1960, 1, 1)
MAX_CHARS = 1200  # semantic.doc_text


def query_prefix(model: str) -> str:
    return next((p for k, p in QUERY_PREFIXES.items() if model.startswith(k)), "")


def _record(m: dict[str, Any]) -> tuple[int, str, str] | None:
    """(recid, text to embed, earliest date) of a record with an arXiv abstract."""
    abstract = next((a["value"] for a in m.get("abstracts", []) if a.get("source") == "arXiv"), None)
    if not abstract or not m.get("titles"):
        return None
    return m["control_number"], f"{m['titles'][0]['title']}. {abstract}"[:MAX_CHARS], str(m.get("earliest_date", ""))


async def _range(client: InspireClient, lo: date, hi: date) -> list[tuple[int, str, str]] | None:
    """Records whose earliest date is in [lo, hi]; None when more than the search API returns (split it)."""
    query = f"{HARVEST_QUERY} and de {lo.isoformat()}->{hi.isoformat()}"
    rows: list[tuple[int, str, str]] = []
    page = 1
    while True:
        data = await client._get("/literature", {"q": query, "size": PAGE, "page": page, "fields": HARVEST_FIELDS})
        total = data["hits"]["total"]
        if total > RESULT_LIMIT:
            return None
        rows += [r for h in data["hits"]["hits"] if (r := _record(h["metadata"]))]
        if page * PAGE >= total:
            return rows
        page += 1


def _months(first: date, last: date) -> list[tuple[date, date]]:
    out, start = [], first
    while start <= last:
        nxt = date(start.year + start.month // 12, start.month % 12 + 1, 1)
        out.append((start, min(nxt - timedelta(days=1), last)))
        start = nxt
    return out


async def harvest(client: InspireClient, since: date = FIRST_DATE) -> AsyncIterator[list[tuple[int, str, str]]]:
    """Batches of (recid, text, earliest date), month by month, within INSPIRE's rate limit (the client's)."""

    async def fetch(lo: date, hi: date) -> list[tuple[int, str, str]]:
        rows = await _range(client, lo, hi)
        if rows is not None:
            return rows
        if lo == hi:
            raise RuntimeError(f"more than {RESULT_LIMIT} records on {lo}")
        mid = lo + (hi - lo) // 2
        return await fetch(lo, mid) + await fetch(mid + timedelta(days=1), hi)

    for lo, hi in _months(since, date.today()):
        yield await fetch(lo, hi)


def _embedder(model: str) -> Any:
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError("the dense index needs the 'semantic' extra") from exc
    return TextEmbedding(model)


def _embed(embedder: Any, texts: list[str]) -> Any:
    import numpy as np

    emb = np.array(list(embedder.embed(texts, batch_size=64)), dtype=np.float32)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


async def build(
    path: Path, model: str = DEFAULT_MODEL, update: bool = False, client: InspireClient | None = None
) -> int:
    """Build the index in `path`, or with `update` add the records dated since its newest date. Returns
    the number of records embedded in this run."""
    import numpy as np

    path.mkdir(parents=True, exist_ok=True)
    meta = json.loads((path / "index.json").read_text()) if update else {}
    model = meta.get("model", model)
    since = date.fromisoformat(meta["newest"]) - timedelta(days=31) if update else FIRST_DATE
    embedder = _embedder(model)
    recids: list[int] = []
    vectors: list[Any] = []
    newest = meta.get("newest", FIRST_DATE.isoformat())
    async with client or InspireClient() as inspire:
        async for batch in harvest(inspire, since):
            if not batch:
                continue
            emb = await asyncio.to_thread(_embed, embedder, [text for _, text, _ in batch])
            vectors.append(emb.astype(np.float16))
            recids += [r for r, _, _ in batch]
            dates = [d[:10] for _, _, d in batch if d]
            newest = max([newest, *dates])
            log.info("index: %d records embedded", len(recids))
    new_ids = np.array(recids, dtype=np.int64)
    new_vecs = np.concatenate(vectors) if vectors else np.zeros((0, 0), dtype=np.float16)
    if update and (path / "recids.npy").exists():
        old_ids, old_vecs = np.load(path / "recids.npy"), np.load(path / "vectors.npy")
        keep = ~np.isin(old_ids, new_ids)  # a re-embedded record replaces its old row
        new_ids = np.concatenate([old_ids[keep], new_ids])
        new_vecs = np.concatenate([old_vecs[keep], new_vecs]) if len(new_vecs) else old_vecs[keep]
    # a record listed twice (e.g. harvested in two months) keeps one row: the last, most recently fetched
    _, from_end = np.unique(new_ids[::-1], return_index=True)
    last = np.sort(len(new_ids) - 1 - from_end)
    np.save(path / "recids.npy", new_ids[last])
    np.save(path / "vectors.npy", new_vecs[last])
    meta = {"model": model, "query_prefix": query_prefix(model), "records": int(len(last)), "newest": newest}
    (path / "index.json").write_text(json.dumps(meta, indent=1))
    return len(recids)


class DenseIndex:
    """Nearest INSPIRE records to a question by meaning (cosine similarity of the model's embeddings)."""

    def __init__(self, path: Path, embedder: Any = None) -> None:
        import numpy as np

        meta = json.loads((path / "index.json").read_text())
        self.path, self.model, self.prefix = path, meta["model"], meta.get("query_prefix", "")
        self.records, self.newest = meta["records"], meta.get("newest")
        # float32 in memory: a matrix-vector product in ~50 ms for 836k records (float16 is ~20x slower in numpy)
        self._vectors = np.load(path / "vectors.npy").astype(np.float32)
        self._recids = np.load(path / "recids.npy")
        if len(self._recids) != len(self._vectors):
            raise ValueError(f"{path}: {len(self._recids)} record ids for {len(self._vectors)} vectors")
        self._embedder = embedder or _embedder(self.model)
        self.name = f"dense-index:{self.model.split('/')[-1]}:{self.records}"

    def search(self, question: str, k: int) -> list[int]:
        import numpy as np

        q = np.asarray(next(iter(self._embedder.embed([self.prefix + question]))), dtype=np.float32)
        scores = self._vectors @ (q / np.linalg.norm(q))
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        return [int(r) for r in self._recids[top[np.argsort(-scores[top])]]]
