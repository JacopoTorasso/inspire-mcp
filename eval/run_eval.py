"""Known-item retrieval evaluation over INSPIRE-HEP.

Question: for natural-language questions a researcher might ask an AI assistant, how often does
each retrieval strategy put the intended paper in the top k?

Query sets
  famous     queries.jsonl: 60 well-cited papers (709 to 24k citations), questions written by the author
  longtail   queries_longtail.jsonl: 40 papers with 20-200 citations sampled at random by
             sample_longtail.py, questions written with fixed rules (paraphrase, at most 20 words,
             no three consecutive words from the title; the overlap is measured and reported)

Part A, offline (famous set, fixed corpus = every INSPIRE record with 1500+ citations):
  bm25, dense (bge-small-en-v1.5), hybrid (reciprocal rank fusion), dense top 50 then cross-encoder
Part B, live INSPIRE index (both sets, whole database). Candidate pools:
  or       content words joined with OR, INSPIRE relevance ranking
  multi    OR results interleaved with quoted adjacent-word phrases and a title-only query
  infl     OR results with half of the pool from papers with 250+ citations
Each pool is reranked by dense embeddings or a cross-encoder, with or without reciprocal rank
fusion with INSPIRE's order; `raw` sends the question unchanged.

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
QUERY_SETS = {"famous": "queries.jsonl", "longtail": "queries_longtail.jsonl"}
# Held out: used once, to confirm a choice made on the sets above (see eval/README.md).
# citations: sentences of citing papers, frozen by freeze_citations.py; the text stays in eval/cache.
HELDOUT_SETS = {"heldout": "queries_heldout.jsonl", "citations": "queries_citations.jsonl"}
CITATION_TEXTS = CACHE / "citation_contexts_judged.jsonl"
CORPUS_QUERY = "topcite 1500+"
KS = (1, 5, 10)
BOOTSTRAP = 2000
SEED = 13
POOLS = ("or", "multi", "infl")
LIVE_SYSTEMS = (
    "raw", "or", "multi",
    "or+dense+rrf", "multi+dense+rrf", "infl+dense+rrf", "infl+dense",
    "or+xenc", "multi+xenc", "infl+xenc",
)  # fmt: skip
LIVE_TESTS = (
    ("or", "raw"),
    ("or+dense+rrf", "or"),
    ("multi+dense+rrf", "or+dense+rrf"),
    ("multi+xenc", "or+xenc"),
    ("infl+dense+rrf", "or+dense+rrf"),
    ("infl+dense", "infl+dense+rrf"),
    ("or+xenc", "or+dense+rrf"),
)


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
    if observed == 0:
        return 1.0
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


def title_overlap(query: str, title: str, n: int = 3) -> bool:
    """True when the question repeats n consecutive words of the target title."""
    words = lambda t: re.findall(r"[a-z0-9]+", t.lower())  # noqa: E731
    qw, tw = words(query), words(title)
    grams = {tuple(tw[i : i + n]) for i in range(len(tw) - n + 1)}
    return any(tuple(qw[i : i + n]) in grams for i in range(len(qw) - n + 1))


# ---------- data ----------


def load_queries(name: str) -> list[dict]:
    path = HERE / {**QUERY_SETS, **HELDOUT_SETS}[name]
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if rows and "query" not in rows[0]:  # ids only: the text comes from eval/cache, checked by hash
        from freeze_citations import sentence_hash

        texts = {r["id"]: r["query"] for r in map(json.loads, CITATION_TEXTS.read_text().splitlines())}
        for r in rows:
            r["query"] = texts[r["id"]]
            if sentence_hash(r["query"]) != r["sha"]:
                raise SystemExit(f"{r['id']}: the sentence in eval/cache differs from the frozen one")
    return rows


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
        _, raw = await client.search_literature(q["query"], size=10)
        _, loose = await client.search_literature(relaxed_query(q["query"]), size=pool)
        or_pool, _, strict_total = await candidates(client, q["query"], pool)
        multi, _, _ = await candidates(client, q["query"], pool, strategy="multi")
        infl, _, _ = await candidates(client, q["query"], pool, influential=pool // 2)
        out[q["id"]] = {
            "raw": raw,
            "or_ranking": loose,
            "or": or_pool,
            "multi": multi,
            "infl": infl,
            "strict_total": strict_total,
        }
    return out


# ---------- evaluation ----------


def order(docs: list[Paper], scores: list[float]) -> list[int]:
    return [d.recid for d, _ in sorted(zip(docs, scores, strict=True), key=lambda x: -x[1])]


def top_ids(scores: list[float], ids: list[int], k: int = 100) -> list[int]:
    return [ids[i] for i in sorted(range(len(ids)), key=scores.__getitem__, reverse=True)[:k]]


def offline(
    corpus: list[Paper], queries: list[dict], dense: DenseReranker, xenc: CrossEncoderReranker, rng: random.Random
) -> dict:
    recids = {p.recid for p in corpus}
    subset = [q for q in queries if q["recid"] in recids]
    ids = [p.recid for p in corpus]
    by_id = {p.recid: p for p in corpus}
    t0 = time.perf_counter()
    dense.scores("warm up the cache", corpus)
    build_s = time.perf_counter() - t0
    bm25 = BM25Reranker()
    ranks: dict[str, list[int | None]] = {"bm25": [], "dense": [], "hybrid": [], "dense>xenc": []}
    for q in subset:
        b = top_ids(bm25.scores(q["query"], corpus), ids)
        d = top_ids(dense.scores(q["query"], corpus), ids)
        top = [by_id[i] for i in d[:50]]
        x = order(top, xenc.scores(q["query"], top))
        for name, r in (("bm25", b), ("dense", d), ("hybrid", rrf(b, d)), ("dense>xenc", x)):
            ranks[name].append(rank_of(r[:10], q["recid"]))
    mrr = {k: per_query_scores(v)["MRR@10"] for k, v in ranks.items()}
    pairs = [("dense", "bm25"), ("hybrid", "dense"), ("dense>xenc", "dense")]
    return {
        "n_queries": len(subset),
        "corpus_size": len(corpus),
        "index_build_s": round(build_s, 1),
        "summary": summarize(ranks, rng),
        "paired_permutation_p_mrr": {
            f"{x} vs {y}": round(paired_permutation_p(mrr[x], mrr[y], rng), 4) for x, y in pairs
        },
        "ranks": {k: dict(zip([q["id"] for q in subset], v, strict=True)) for k, v in ranks.items()},
    }


def live(
    queries: list[dict], pools: dict[str, dict], dense: DenseReranker, xenc: CrossEncoderReranker, rng: random.Random
) -> dict:
    ranks: dict[str, list[int | None]] = {k: [] for k in LIVE_SYSTEMS}
    in_pool: dict[str, list[float]] = {k: [] for k in POOLS}
    latency = {"dense": [], "cross_encoder": []}
    for q in queries:
        p, g, text = pools[q["id"]], q["recid"], q["query"]
        systems = {"raw": [d.recid for d in p["raw"]], "or": [d.recid for d in p["or_ranking"]]}
        for name in POOLS:
            docs = p[name]
            inspire_order = [d.recid for d in docs]
            t0 = time.perf_counter()
            dense_order = order(docs, dense.scores(text, docs))
            latency["dense"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            xenc_order = order(docs, xenc.scores(text, docs))
            latency["cross_encoder"].append(time.perf_counter() - t0)
            systems[f"{name}+dense"] = dense_order
            systems[f"{name}+dense+rrf"] = rrf(dense_order, inspire_order)
            systems[f"{name}+xenc"] = xenc_order
            in_pool[name].append(1.0 if g in set(inspire_order) else 0.0)
        systems["multi"] = [d.recid for d in p["multi"]]
        for name in LIVE_SYSTEMS:
            ranks[name].append(rank_of(systems[name][:10], g))

    mrr = {k: per_query_scores(v)["MRR@10"] for k, v in ranks.items()}
    return {
        "n_queries": len(queries),
        "title_overlap_share": round(statistics.fmean(title_overlap(q["query"], q["title"]) for q in queries), 3),
        "summary": summarize(ranks, rng),
        "pool_recall": {k: round(statistics.fmean(v), 3) for k, v in in_pool.items()},
        "pool_recall_p": {
            "multi vs or": round(paired_permutation_p(in_pool["multi"], in_pool["or"], rng), 4),
            "infl vs or": round(paired_permutation_p(in_pool["infl"], in_pool["or"], rng), 4),
        },
        "raw_zero_hits": sum(pools[q["id"]]["strict_total"] == 0 for q in queries),
        "paired_permutation_p_mrr": {
            f"{x} vs {y}": round(paired_permutation_p(mrr[x], mrr[y], rng), 4) for x, y in LIVE_TESTS
        },
        "median_rerank_latency_s": {k: round(statistics.median(v), 3) for k, v in latency.items()},
        "ranks": {k: dict(zip([q["id"] for q in queries], v, strict=True)) for k, v in ranks.items()},
    }


async def gather(refresh: bool, pool: int) -> tuple[list[Paper], dict[str, list[dict]], dict[str, dict]]:
    CACHE.mkdir(exist_ok=True)
    sets = {name: load_queries(name) for name in QUERY_SETS}
    corpus_path = CACHE / "corpus.json"
    pools = {}
    async with InspireClient() as client:
        if refresh or not corpus_path.exists():
            corpus_path.write_text(json.dumps(await fetch_corpus(client)))
        for name, queries in sets.items():
            await resolve_gold(client, queries)
            pools[name] = await live_pools(client, queries, pool)
    corpus = [Paper(**p) for p in json.loads(corpus_path.read_text())]
    return corpus, sets, pools


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-download the offline corpus snapshot")
    parser.add_argument("--pool", type=int, default=50, help="live candidates to rerank")
    args = parser.parse_args()
    rng = random.Random(SEED)

    corpus, sets, pools = asyncio.run(gather(args.refresh, args.pool))
    dense = DenseReranker(cache_path=CACHE)
    xenc = CrossEncoderReranker()
    report = {
        "date": date.today().isoformat(),
        "model": dense.model_name,
        "cross_encoder": xenc.model_name,
        "pool": args.pool,
        "offline": offline(corpus, sets["famous"], dense, xenc, rng),
        "live": {name: live(queries, pools[name], dense, xenc, rng) for name, queries in sets.items()},
        "queries": {
            name: [{k: q[k] for k in ("id", "query", "arxiv", "recid", "title", "citations")} for q in queries]
            for name, queries in sets.items()
        },
    }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"results-{report['date']}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print_markdown(report)
    print(f"\nsaved {out.relative_to(HERE.parent)}")


def print_markdown(r: dict) -> None:
    def table(summary: dict) -> None:
        print("| system | R@1 | R@5 | R@10 [95% CI] | MRR@10 [95% CI] |\n|---|---|---|---|---|")
        for name, m in summary.items():
            print(
                f"| {name} | {m['R@1']:.2f} | {m['R@5']:.2f} | {m['R@10']:.2f} "
                f"[{m['R@10_ci95'][0]:.2f}, {m['R@10_ci95'][1]:.2f}] | {m['MRR@10']:.2f} "
                f"[{m['MRR@10_ci95'][0]:.2f}, {m['MRR@10_ci95'][1]:.2f}] |"
            )

    off = r["offline"]
    print(f"## Offline: {off['corpus_size']} papers, {off['n_queries']} queries\n")
    table(off["summary"])
    print(f"\nPaired permutation p on MRR@10: {off['paired_permutation_p_mrr']}")
    for name, liv in r["live"].items():
        print(f"\n## Live INSPIRE, {name}: {liv['n_queries']} queries, pool {r['pool']}\n")
        table(liv["summary"])
        print(f"\nPool recall: {liv['pool_recall']}  p: {liv['pool_recall_p']}")
        print(f"Raw question with zero hits: {liv['raw_zero_hits']}/{liv['n_queries']}")
        print(f"Questions repeating 3+ title words: {liv['title_overlap_share']}")
        print(f"Paired permutation p on MRR@10: {liv['paired_permutation_p_mrr']}")
        print(f"Median rerank latency (s): {liv['median_rerank_latency_s']}")


if __name__ == "__main__":
    main()
