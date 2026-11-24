"""Known-item retrieval evaluation over INSPIRE-HEP.

Question: for natural-language questions a researcher might ask an AI assistant, how often does
each retrieval strategy put the intended paper in the top k?

Part A, offline (fixed corpus = every INSPIRE record with 1500+ citations, queries whose paper is in it):
  bm25, dense (bge-small-en-v1.5), hybrid (reciprocal rank fusion)
Part B, live INSPIRE index (all queries, whole database):
  raw            the question sent as is to INSPIRE search
  or             content words joined with OR, INSPIRE relevance ranking
  or+bm25        the 'or' candidate pool reranked with BM25
  or+dense       the pool reranked with dense embeddings
  or+dense+rrf   dense order fused with INSPIRE order (what the semantic_search tool does)
  infl+dense+rrf same, with part of the pool reserved to well cited papers (influential=25)

Usage:  uv run --extra eval python eval/run_eval.py [--refresh] [--pool 50]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import statistics
import sys
import time
from collections.abc import Sequence
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inspire_mcp.client import InspireClient, Paper  # noqa: E402
from inspire_mcp.semantic import (  # noqa: E402
    BM25Reranker,
    CrossEncoderReranker,
    DenseReranker,
    candidates,
    relaxed_query,
    rrf,
)

HERE = Path(__file__).parent
CACHE = HERE / "cache"
RESULTS = HERE / "results"
CORPUS_QUERY = "topcite 1500+"
KS = (1, 5, 10)
BOOTSTRAP = 2000
SEED = 13


# ---------- metrics ----------


def rank_of(ranking: Sequence[int], gold: int) -> int | None:
    return ranking.index(gold) + 1 if gold in ranking else None


def per_query_scores(ranks: list[int | None]) -> dict[str, list[float]]:
    out = {f"R@{k}": [1.0 if r is not None and r <= k else 0.0 for r in ranks] for k in KS}
    out["MRR@10"] = [1.0 / r if r is not None and r <= 10 else 0.0 for r in ranks]
    return out


def bootstrap_ci(values: list[float], rng: random.Random) -> tuple[float, float]:
    n = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=n)) for _ in range(BOOTSTRAP))
    return means[int(0.025 * BOOTSTRAP)], means[int(0.975 * BOOTSTRAP) - 1]


def paired_permutation_p(a: list[float], b: list[float], rng: random.Random, n: int = 10000) -> float:
    """Two-sided sign-flip test on per-query differences."""
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    observed = abs(sum(diffs))
    hits = sum(abs(sum(d if rng.random() < 0.5 else -d for d in diffs)) >= observed - 1e-12 for _ in range(n))
    return (hits + 1) / (n + 1)


def summarize(ranks: dict[str, list[int | None]], rng: random.Random) -> dict[str, dict]:
    out = {}
    for name, rs in ranks.items():
        scores = per_query_scores(rs)
        out[name] = {m: round(statistics.fmean(v), 3) for m, v in scores.items()}
        for m in ("R@10", "MRR@10"):
            lo, hi = bootstrap_ci(scores[m], rng)
            out[name][f"{m}_ci95"] = [round(lo, 3), round(hi, 3)]
    return out


# ---------- data ----------


def load_queries() -> list[dict]:
    return [json.loads(line) for line in (HERE / "queries.jsonl").read_text().splitlines() if line.strip()]


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
        q["recid"], q["title"], q["citations"] = paper.recid, paper.title, paper.citation_count


async def live_pools(client: InspireClient, queries: list[dict], pool: int) -> dict[str, dict]:
    out = {}
    for q in queries:
        t0 = time.perf_counter()
        _, raw = await client.search_literature(q["query"], size=10)
        raw_s = time.perf_counter() - t0
        _, loose = await client.search_literature(relaxed_query(q["query"]), size=pool)
        docs, _, strict_total = await candidates(client, q["query"], pool)
        infl, _, _ = await candidates(client, q["query"], pool, influential=25)
        out[q["id"]] = {
            "raw": raw,
            "or": loose,
            "pool": docs,
            "pool_infl": infl,
            "strict_total": strict_total,
            "raw_latency_s": raw_s,
        }
    return out


# ---------- main ----------


def top_ids(scores: list[float], ids: list[int], k: int = 100) -> list[int]:
    return [ids[i] for i in sorted(range(len(ids)), key=scores.__getitem__, reverse=True)[:k]]


def offline(
    corpus: list[Paper], queries: list[dict], dense: DenseReranker, xenc: CrossEncoderReranker, rng: random.Random
) -> dict:
    recids = {p.recid for p in corpus}
    subset = [q for q in queries if q["recid"] in recids]
    ids = [p.recid for p in corpus]
    t0 = time.perf_counter()
    dense.scores("warm up the cache", corpus)
    build_s = time.perf_counter() - t0
    bm25 = BM25Reranker()
    ranks: dict[str, list[int | None]] = {"bm25": [], "dense": [], "hybrid": [], "dense>xenc": []}
    by_id = {p.recid: p for p in corpus}
    for q in subset:
        b = top_ids(bm25.scores(q["query"], corpus), ids)
        d = top_ids(dense.scores(q["query"], corpus), ids)
        top = [by_id[i] for i in d[:50]]
        x = [p.recid for p, _ in sorted(zip(top, xenc.scores(q["query"], top), strict=True), key=lambda t: -t[1])]
        for name, r in (("bm25", b), ("dense", d), ("hybrid", rrf(b, d)), ("dense>xenc", x)):
            ranks[name].append(rank_of(r[:10], q["recid"]))
    return {
        "n_queries": len(subset),
        "corpus_size": len(corpus),
        "index_build_s": round(build_s, 1),
        "summary": summarize(ranks, rng),
        "ranks": {k: dict(zip([q["id"] for q in subset], v, strict=True)) for k, v in ranks.items()},
    }


def live(
    queries: list[dict], pools: dict[str, dict], dense: DenseReranker, xenc: CrossEncoderReranker, rng: random.Random
) -> dict:
    bm25 = BM25Reranker()
    names = (
        "raw",
        "or",
        "or+bm25",
        "or+dense",
        "or+dense+rrf",
        "or+xenc",
        "or+xenc+rrf",
        "infl+dense+rrf",
        "infl+xenc",
        "infl+xenc+rrf",
    )
    ranks: dict[str, list[int | None]] = {k: [] for k in names}
    pool_recall = {"pool": [], "pool_infl": []}
    latency = {"dense_cached_vectors": [], "cross_encoder": []}

    def order(docs: list[Paper], scores: list[float]) -> list[int]:
        return [d.recid for d, _ in sorted(zip(docs, scores, strict=True), key=lambda x: -x[1])]

    for q in queries:
        p = pools[q["id"]]
        g, text = q["recid"], q["query"]
        docs, infl = p["pool"], p["pool_infl"]
        t0 = time.perf_counter()
        dense_order = order(docs, dense.scores(text, docs))
        latency["dense_cached_vectors"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        xenc_order = order(docs, xenc.scores(text, docs))
        latency["cross_encoder"].append(time.perf_counter() - t0)
        infl_dense = order(infl, dense.scores(text, infl))
        infl_xenc = order(infl, xenc.scores(text, infl))
        inspire_order, infl_order = [d.recid for d in docs], [d.recid for d in infl]
        systems = {
            "raw": [d.recid for d in p["raw"]],
            "or": [d.recid for d in p["or"]],
            "or+bm25": order(docs, bm25.scores(text, docs)),
            "or+dense": dense_order,
            "or+dense+rrf": rrf(dense_order, inspire_order),
            "or+xenc": xenc_order,
            "or+xenc+rrf": rrf(xenc_order, inspire_order),
            "infl+dense+rrf": rrf(infl_dense, infl_order),
            "infl+xenc": infl_xenc,
            "infl+xenc+rrf": rrf(infl_xenc, infl_order),
        }
        for name in names:
            ranks[name].append(rank_of(systems[name][:10], g))
        pool_recall["pool"].append(g in set(inspire_order))
        pool_recall["pool_infl"].append(g in set(infl_order))

    mrr = {k: per_query_scores(v)["MRR@10"] for k, v in ranks.items()}
    pairs = [
        ("or+dense+rrf", "or"),
        ("or+xenc", "or"),
        ("or+xenc", "or+dense+rrf"),
        ("or+xenc+rrf", "or+xenc"),
        ("infl+xenc", "infl+dense+rrf"),
        ("infl+xenc", "or+xenc"),
        ("or", "raw"),
    ]
    return {
        "n_queries": len(queries),
        "summary": summarize(ranks, rng),
        "pool_recall": {k: round(statistics.fmean(v), 3) for k, v in pool_recall.items()},
        "raw_zero_hits": sum(pools[q["id"]]["strict_total"] == 0 for q in queries),
        "paired_permutation_p_mrr": {
            f"{x} vs {y}": round(paired_permutation_p(mrr[x], mrr[y], rng), 4) for x, y in pairs
        },
        "median_rerank_latency_s": {k: round(statistics.median(v), 3) for k, v in latency.items()},
        "ranks": {k: dict(zip([q["id"] for q in queries], v, strict=True)) for k, v in ranks.items()},
    }


async def gather(refresh: bool, pool: int) -> tuple[list[Paper], list[dict], dict]:
    CACHE.mkdir(exist_ok=True)
    queries = load_queries()
    corpus_path = CACHE / "corpus.json"
    async with InspireClient() as client:
        if refresh or not corpus_path.exists():
            corpus_path.write_text(json.dumps(await fetch_corpus(client)))
        await resolve_gold(client, queries)
        pools = await live_pools(client, queries, pool)
    corpus = [Paper(**p) for p in json.loads(corpus_path.read_text())]
    return corpus, queries, pools


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-download the offline corpus snapshot")
    parser.add_argument("--pool", type=int, default=50, help="live candidates to rerank")
    args = parser.parse_args()
    rng = random.Random(SEED)

    corpus, queries, pools = asyncio.run(gather(args.refresh, args.pool))
    dense = DenseReranker(cache_path=CACHE)
    xenc = CrossEncoderReranker()
    report = {
        "date": date.today().isoformat(),
        "model": dense.model_name,
        "pool": args.pool,
        "offline": offline(corpus, queries, dense, xenc, rng),
        "cross_encoder": xenc.model_name,
        "live": live(queries, pools, dense, xenc, rng),
        "queries": [{k: q[k] for k in ("id", "query", "arxiv", "recid", "title", "citations")} for q in queries],
    }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"results-{report['date']}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print_markdown(report)
    print(f"\nsaved {out.relative_to(HERE.parent)}")


def print_markdown(r: dict) -> None:
    def table(summary: dict) -> None:
        print("| system | R@1 | R@5 | R@10 (95% CI) | MRR@10 (95% CI) |\n|---|---|---|---|---|")
        for name, m in summary.items():
            print(
                f"| {name} | {m['R@1']:.2f} | {m['R@5']:.2f} | {m['R@10']:.2f} "
                f"[{m['R@10_ci95'][0]:.2f}, {m['R@10_ci95'][1]:.2f}] | {m['MRR@10']:.2f} "
                f"[{m['MRR@10_ci95'][0]:.2f}, {m['MRR@10_ci95'][1]:.2f}] |"
            )

    off, liv = r["offline"], r["live"]
    print(f"## Offline: {off['corpus_size']} papers, {off['n_queries']} queries\n")
    table(off["summary"])
    print(f"\n## Live INSPIRE: {liv['n_queries']} queries, pool {r['pool']}\n")
    table(liv["summary"])
    print(f"\nPool recall (upper bound for reranking): {liv['pool_recall']}")
    print(f"Raw question with zero hits: {liv['raw_zero_hits']}/{liv['n_queries']}")
    print(f"Paired permutation p-values on MRR@10: {liv['paired_permutation_p_mrr']}")
    print(f"Median rerank latency per query (pool {r['pool']}): {liv['median_rerank_latency_s']}")
    print("\n| id | " + " | ".join(liv["ranks"]) + " | gold |\n|---|" + "---|" * (len(liv["ranks"]) + 1))
    titles = {q["id"]: q["title"] for q in r["queries"]}
    for qid in titles:
        cells = [str(liv["ranks"][k][qid] or "-") for k in liv["ranks"]]
        print(f"| {qid} | " + " | ".join(cells) + f" | {re.sub(r'[$|]', '', titles[qid])[:55]} |")


if __name__ == "__main__":
    main()
