"""Known-item retrieval evaluation over INSPIRE-HEP.

Question: for natural-language questions a researcher might ask an AI assistant,
how often does each retrieval strategy put the intended paper in the top k?

Systems compared on the same fixed corpus (every INSPIRE record with 1500+ citations):
  bm25    lexical baseline (Okapi BM25 over title + abstract)
  dense   embedding retrieval (BAAI/bge-small-en-v1.5 via fastembed, CPU, ONNX)
  hybrid  reciprocal rank fusion of bm25 and dense
plus the live INSPIRE search API queried with the raw question (its own index, no fixed corpus).

Usage:  uv run --extra eval python eval/run_eval.py [--refresh]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import statistics
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inspire_mcp.client import InspireClient  # noqa: E402

HERE = Path(__file__).parent
CACHE = HERE / "cache"
RESULTS = HERE / "results"
CORPUS_QUERY = "topcite 1500+"
MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
KS = (1, 5, 10)
RRF_K = 60

STOPWORDS = set(
    "a an and are as at be by for from in into is it of on or that the this to with which what paper "
    "we our its their using use used show shown".split()
)


# ---------- data ----------


async def fetch_corpus(client: InspireClient) -> list[dict]:
    total, first = await client.search_literature(CORPUS_QUERY, size=250, sort="mostcited", page=1)
    papers = list(first)
    for page in range(2, math.ceil(total / 250) + 1):
        _, more = await client.search_literature(CORPUS_QUERY, size=250, sort="mostcited", page=page)
        papers.extend(more)
    return [p.to_dict() for p in papers]


async def resolve_gold(client: InspireClient, queries: list[dict]) -> None:
    for q in queries:
        paper = await client.get_paper(q["arxiv"])
        if paper is None:
            raise SystemExit(f"gold paper {q['arxiv']} not found on INSPIRE")
        q["recid"] = paper.recid
        q["title"] = paper.title


async def live_inspire(client: InspireClient, queries: list[dict], k: int) -> dict[str, dict]:
    out = {}
    for q in queries:
        t0 = time.perf_counter()
        total, papers = await client.search_literature(q["query"], size=k)
        out[q["id"]] = {
            "ranking": [p.recid for p in papers],
            "total_hits": total,
            "latency_s": time.perf_counter() - t0,
        }
    return out


def doc_text(p: dict) -> str:
    return f"{p['title']}. {p.get('abstract') or ''}".strip()


# ---------- retrievers ----------


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) > 1]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = np.array([sum(tf.values()) for tf in self.tfs], dtype=float)
        self.avg = self.lens.mean()
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.tfs))
        norm = self.k1 * (1 - self.b + self.b * self.lens / self.avg)
        for t in set(tokenize(query)):
            if t not in self.idf:
                continue
            tf = np.array([d.get(t, 0) for d in self.tfs], dtype=float)
            s += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
        return s


def dense_index(docs: list[str], recids: list[int]) -> tuple[object, np.ndarray, float]:
    from fastembed import TextEmbedding

    model = TextEmbedding(MODEL)
    path = CACHE / f"emb-{MODEL.replace('/', '_')}-{len(recids)}-{hash(tuple(recids)) & 0xFFFFFFFF:x}.npy"
    t0 = time.perf_counter()
    if path.exists():
        emb = np.load(path)
        build_s = float("nan")
    else:
        emb = np.array(list(model.embed(docs, batch_size=64)))
        np.save(path, emb)
        build_s = time.perf_counter() - t0
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return model, emb, build_s


def rank(scores: np.ndarray, recids: list[int], k: int) -> list[int]:
    top = np.argsort(-scores)[:k]
    return [recids[i] for i in top]


def rrf(*rankings: list[int], k: int = RRF_K) -> list[int]:
    fused: Counter = Counter()
    for r in rankings:
        for pos, recid in enumerate(r):
            fused[recid] += 1.0 / (k + pos + 1)
    return [recid for recid, _ in fused.most_common()]


# ---------- metrics ----------


def metrics(rankings: dict[str, list[int]], gold: dict[str, int], ids: list[str]) -> dict[str, float]:
    out = {}
    for k in KS:
        out[f"R@{k}"] = statistics.mean(gold[i] in rankings[i][:k] for i in ids)
    out["MRR@10"] = statistics.mean(
        1 / (rankings[i][:10].index(gold[i]) + 1) if gold[i] in rankings[i][:10] else 0.0 for i in ids
    )
    return out


def position(ranking: list[int], recid: int) -> str:
    return str(ranking.index(recid) + 1) if recid in ranking[:10] else "-"


# ---------- main ----------


async def gather_data(refresh: bool) -> tuple[list[dict], list[dict], dict]:
    CACHE.mkdir(exist_ok=True)
    queries = [json.loads(line) for line in (HERE / "queries.jsonl").read_text().splitlines() if line.strip()]
    corpus_path = CACHE / "corpus.json"
    async with InspireClient() as client:
        if refresh or not corpus_path.exists():
            corpus_path.write_text(json.dumps(await fetch_corpus(client)))
        await resolve_gold(client, queries)
        live = await live_inspire(client, queries, k=10)
    return json.loads(corpus_path.read_text()), queries, live


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-download the corpus snapshot")
    args = parser.parse_args()

    corpus, queries, live = asyncio.run(gather_data(args.refresh))
    recids = [p["recid"] for p in corpus]
    docs = [doc_text(p) for p in corpus]
    gold = {q["id"]: q["recid"] for q in queries}
    missing = [q["arxiv"] for q in queries if q["recid"] not in set(recids)]
    if missing:
        raise SystemExit(f"gold papers missing from corpus: {missing}")

    bm25 = BM25(docs)
    model, emb, build_s = dense_index(docs, recids)

    runs: dict[str, dict[str, list[int]]] = {"bm25": {}, "dense": {}, "hybrid": {}, "inspire_live": {}}
    query_latency = []
    for q in queries:
        b = rank(bm25.scores(q["query"]), recids, 100)
        t0 = time.perf_counter()
        qv = np.array(list(model.embed([BGE_QUERY_PREFIX + q["query"]])))[0]
        qv /= np.linalg.norm(qv)
        d = rank(emb @ qv, recids, 100)
        query_latency.append(time.perf_counter() - t0)
        runs["bm25"][q["id"]] = b
        runs["dense"][q["id"]] = d
        runs["hybrid"][q["id"]] = rrf(b, d)
        runs["inspire_live"][q["id"]] = live[q["id"]]["ranking"]

    ids = [q["id"] for q in queries]
    named = [q["id"] for q in queries if re.search(r"\b[A-Z][A-Za-z]*[A-Z0-9]+\b", q["query"])]
    descriptive = [i for i in ids if i not in named]
    summary = {name: metrics(r, gold, ids) for name, r in runs.items()}
    by_style = {
        style: {name: metrics(r, gold, subset)["R@10"] for name, r in runs.items()}
        for style, subset in (("with_acronym_or_name", named), ("descriptive_only", descriptive))
    }
    zero_hits = sum(live[i]["total_hits"] == 0 for i in ids)

    report = {
        "date": date.today().isoformat(),
        "corpus": {"query": CORPUS_QUERY, "size": len(corpus)},
        "model": MODEL,
        "n_queries": len(queries),
        "summary": summary,
        "recall_at_10_by_query_style": {
            k: {"n": len(s), **by_style[k]}
            for k, s in (("with_acronym_or_name", named), ("descriptive_only", descriptive))
        },
        "inspire_live": {
            "zero_result_queries": zero_hits,
            "median_latency_s": statistics.median(v["latency_s"] for v in live.values()),
        },
        "dense_cost": {
            "index_build_s": build_s,
            "median_query_embed_s": statistics.median(query_latency),
            "embedding_dim": int(emb.shape[1]),
        },
        "per_query": [
            {
                "id": q["id"],
                "query": q["query"],
                "gold": q["title"],
                **{name: position(runs[name][q["id"]], q["recid"]) for name in runs},
                "inspire_total_hits": live[q["id"]]["total_hits"],
            }
            for q in queries
        ],
    }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"results-{report['date']}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print_markdown(report)
    print(f"\nsaved {out.relative_to(HERE.parent)}")


def print_markdown(r: dict) -> None:
    print(f"Corpus: {r['corpus']['size']} INSPIRE records ({r['corpus']['query']}), {r['n_queries']} queries\n")
    print("| system | R@1 | R@5 | R@10 | MRR@10 |\n|---|---|---|---|---|")
    for name, m in r["summary"].items():
        print(f"| {name} | {m['R@1']:.2f} | {m['R@5']:.2f} | {m['R@10']:.2f} | {m['MRR@10']:.2f} |")
    print("\nR@10 by query style:")
    for style, v in r["recall_at_10_by_query_style"].items():
        print(f"  {style} (n={v['n']}): " + ", ".join(f"{k}={v[k]:.2f}" for k in r["summary"]))
    print(
        f"\nINSPIRE live: {r['inspire_live']['zero_result_queries']} queries with zero hits, "
        f"median latency {r['inspire_live']['median_latency_s']:.2f}s"
    )
    c = r["dense_cost"]
    print(
        f"Dense: dim {c['embedding_dim']}, index build {c['index_build_s']:.1f}s, "
        f"median query embed {c['median_query_embed_s'] * 1000:.0f} ms (CPU)"
    )
    print("\n| id | bm25 | dense | hybrid | live | gold |\n|---|---|---|---|---|---|")
    for p in r["per_query"]:
        print(f"| {p['id']} | {p['bm25']} | {p['dense']} | {p['hybrid']} | {p['inspire_live']} | {p['gold'][:60]} |")


if __name__ == "__main__":
    main()
