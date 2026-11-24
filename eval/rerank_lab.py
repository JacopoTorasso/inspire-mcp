"""Reranker experiments on frozen candidate pools.

Candidate pools (100 per question, the first 50 are exactly the server's default pool) are fetched once
and saved, and cross-encoder scores are cached per (model, question, record), so every model and every
pool size is compared on the same candidates and a run can resume where it stopped.

Use the development sets (queries.jsonl, queries_longtail.jsonl) to choose; confirm the choice once on a
held-out set with --sets test.

    uv run --extra eval --extra semantic python eval/rerank_lab.py --models all
    uv run --extra eval --extra semantic python eval/rerank_lab.py --sets test --models <model>,<model>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from run_eval import bootstrap_ci, load_queries, paired_permutation_p, per_query_scores, rank_of, resolve_gold  # noqa: E402

from inspire_mcp.client import InspireClient, Paper  # noqa: E402
from inspire_mcp.semantic import DenseReranker, candidates, doc_text, rrf  # noqa: E402

HERE = Path(__file__).parent
CACHE = HERE / "cache"
MODELS = [
    "Xenova/ms-marco-MiniLM-L-6-v2",
    "Xenova/ms-marco-MiniLM-L-12-v2",
    "jinaai/jina-reranker-v1-tiny-en",
    "jinaai/jina-reranker-v1-turbo-en",
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-v2-m3-int8",
    "jinaai/jina-reranker-v2-base-multilingual",
]
BASELINE = "Xenova/ms-marco-MiniLM-L-6-v2@50"
SETS = {"dev": ("famous", "longtail"), "test": ("heldout",)}


async def frozen_pools(names: tuple[str, ...]) -> dict[str, list[dict]]:
    path = CACHE / f"pools-{'-'.join(names)}.json"
    if path.exists():
        return json.loads(path.read_text())
    out: dict[str, list[dict]] = {}
    async with InspireClient() as client:
        for name in names:
            queries = load_queries(name)
            await resolve_gold(client, queries)
            for q in queries:
                docs, _, _ = await candidates(client, q["query"], pool=100)
                q["pool"] = [d.to_dict() for d in docs]
            out[name] = queries
    CACHE.mkdir(exist_ok=True)
    path.write_text(json.dumps(out))
    return out


class ScoreCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, float] = json.loads(path.read_text()) if path.exists() else {}

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data))


def score_model(model: str, sets: dict[str, list[dict]], cache: ScoreCache) -> float:
    """Score every (question, candidate) pair not yet cached; return median seconds per 50 candidates."""
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    encoder = None
    timings = []
    for queries in sets.values():
        for q in queries:
            todo = [p for p in q["pool"] if f"{model}|{q['id']}|{p['recid']}" not in cache.data]
            if not todo:
                continue
            encoder = encoder or TextCrossEncoder(model)
            start = time.perf_counter()
            texts = [doc_text(Paper(**p)) for p in todo]
            scores = list(encoder.rerank(q["query"], texts, batch_size=16))
            timings.append((time.perf_counter() - start) * 50 / len(todo))
            for p, s in zip(todo, scores, strict=True):
                cache.data[f"{model}|{q['id']}|{p['recid']}"] = float(s)
        cache.save()
    return statistics.median(timings) if timings else float("nan")


def ranking(model: str, q: dict, pool: int, cache: ScoreCache) -> list[int]:
    docs = q["pool"][:pool]
    return [p["recid"] for p in sorted(docs, key=lambda p: -cache.data[f"{model}|{q['id']}|{p['recid']}"])]


def evaluate(systems: dict[str, dict[str, list[int]]], sets: dict[str, list[dict]]) -> None:
    rng = random.Random(13)
    for name, queries in sets.items():
        gold = {q["id"]: q["recid"] for q in queries}
        print(
            f"\n## {name} ({len(queries)} questions)  in pool: "
            f"50={statistics.fmean(gold[q['id']] in [p['recid'] for p in q['pool'][:50]] for q in queries):.2f} "
            f"100={statistics.fmean(gold[q['id']] in [p['recid'] for p in q['pool']] for q in queries):.2f}"
        )
        base = per_query_scores([rank_of(systems[BASELINE][q["id"]][:10], gold[q["id"]]) for q in queries])["MRR@10"]
        rows = []
        for system, ranks in systems.items():
            r = [rank_of(ranks[q["id"]][:10], gold[q["id"]]) for q in queries]
            s = per_query_scores(r)
            p = paired_permutation_p(s["MRR@10"], base, rng) if system != BASELINE else 1.0
            rows.append(
                (
                    statistics.fmean(s["MRR@10"]),
                    system,
                    statistics.fmean(s["R@1"]),
                    statistics.fmean(s["R@10"]),
                    bootstrap_ci(s["MRR@10"], rng),
                    p,
                )
            )
        print(f"  {'system':<58} R@1   R@10  MRR   [95% CI]      p vs baseline")
        for mrr, system, r1, r10, ci, p in sorted(rows, reverse=True):
            print(f"  {system:<58} {r1:.2f}  {r10:.2f}  {mrr:.2f}  [{ci[0]:.2f}, {ci[1]:.2f}]  {p:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", choices=sorted(SETS), default="dev")
    parser.add_argument("--models", default="all", help="'all' or a comma-separated list")
    args = parser.parse_args()
    models = MODELS if args.models == "all" else args.models.split(",")
    sets = asyncio.run(frozen_pools(SETS[args.sets]))
    cache = ScoreCache(CACHE / "xenc-scores.json")
    latency = {}
    for model in models:
        latency[model] = score_model(model, sets, cache)
        print(f"scored {model}: {latency[model]:.2f}s per 50 candidates", flush=True)

    dense = DenseReranker(cache_path=CACHE)
    systems: dict[str, dict[str, list[int]]] = {}
    for queries in sets.values():
        for q in queries:
            docs = [Paper(**p) for p in q["pool"]]
            inspire_order = [d.recid for d in docs]
            dense_order = [
                d.recid
                for _, d in sorted(
                    zip(dense.scores(q["query"], docs[:50]), docs[:50], strict=True), key=lambda t: -t[0]
                )
            ]
            systems.setdefault("inspire OR order@50", {})[q["id"]] = inspire_order[:50]
            for model in models:
                for pool in (50, 100):
                    systems.setdefault(f"{model}@{pool}", {})[q["id"]] = ranking(model, q, pool, cache)
                systems.setdefault(f"{model}@50 + dense (rrf)", {})[q["id"]] = rrf(
                    ranking(model, q, 50, cache), dense_order
                )
            if len(models) >= 2:
                top = [ranking(m, q, 50, cache) for m in models]
                systems.setdefault("ensemble of all cross-encoders@50 (rrf)", {})[q["id"]] = rrf(*top)
    evaluate(systems, sets)
    print(
        "\nlatency, seconds per 50 candidates on this CPU:", {m.split("/")[-1]: round(v, 2) for m, v in latency.items()}
    )


if __name__ == "__main__":
    main()
