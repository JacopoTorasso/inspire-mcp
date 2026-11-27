"""Compare agent_eval runs of a baseline and a variant on the pooled development questions.

Each side is a list of result files (several runs and sets: dev and citedev); the per-question numbers
are averaged over the runs of a side, then pooled over the questions. Applies the efficiency criterion
fixed before the variants were tried (commit ab6df6d): a variant is kept if tool calls per question drop
by 20% or cost per question by 25%, with pooled MRR@10 down by at most 0.03.

    python eval/compare_runs.py --base eval/results/a.json eval/results/b.json --variant eval/results/c.json ...
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

CALLS_DROP, COST_DROP, MRR_LOSS = 0.20, 0.25, 0.03


def per_question(files: list[Path], arm: str) -> dict[str, dict[str, float]]:
    """(set, id) -> mean reciprocal rank, tool calls, cost and turn-limit rate over the runs in `files`."""
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for f in files:
        report = json.loads(f.read_text())
        for set_name, data in report["sets"].items():
            for q in data["per_query"]:
                r = q[arm]
                a = acc[f"{set_name}/{q['id']}"]
                a["rr"].append(1 / r["rank"] if r["rank"] else 0.0)
                a["calls"].append(len(r["queries"]))
                a["cost"].append(r["cost_usd"])
                a["turn_limit"].append(float(r["outcome"] == "turn_limit"))
    return {k: {m: statistics.fmean(v) for m, v in a.items()} for k, a in acc.items()}


def paired_p(xs: list[float], ys: list[float], n: int = 10000, seed: int = 0) -> float:
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(xs, ys, strict=True)]
    observed = abs(sum(diffs))
    hits = sum(abs(sum(d if rng.random() < 0.5 else -d for d in diffs)) >= observed for _ in range(n))
    return (hits + 1) / (n + 1)


def bootstrap(
    base: dict[str, dict[str, float]], var: dict[str, dict[str, float]], keys: list[str], n: int = 5000, seed: int = 0
) -> dict[str, tuple[float, float]]:
    """95% intervals, resampling questions, for the relative change of calls and cost and the MRR change."""
    rng = random.Random(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(n):
        ks = [keys[rng.randrange(len(keys))] for _ in keys]
        b, v = summary(base, ks), summary(var, ks)
        draws["calls"].append(v["calls"] / b["calls"] - 1)
        draws["cost"].append(v["cost"] / b["cost"] - 1)
        draws["rr"].append(v["rr"] - b["rr"])
    return {m: (sorted(d)[int(0.025 * n)], sorted(d)[int(0.975 * n)]) for m, d in draws.items()}


def summary(q: dict[str, dict[str, float]], keys: list[str]) -> dict[str, float]:
    return {m: statistics.fmean(q[k][m] for k in keys) for m in ("rr", "calls", "cost", "turn_limit")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", nargs="+", type=Path, required=True)
    parser.add_argument("--variant", nargs="+", type=Path, required=True)
    parser.add_argument("--arm", default="B")
    args = parser.parse_args()
    base, var = per_question(args.base, args.arm), per_question(args.variant, args.arm)
    keys = sorted(base.keys() & var.keys())
    if len(keys) != len(base) or len(keys) != len(var):
        print(f"warning: {len(base)} base and {len(var)} variant questions, comparing the {len(keys)} shared")
    groups: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        groups[k.split("/")[0]].append(k)
    rows: dict[str, Any] = {"pooled": keys, **groups}
    print(
        f"{'set':10s} {'n':>4s}  {'MRR base':>8s} {'var':>6s}  {'calls base':>10s} {'var':>5s}  "
        f"{'$/q base':>9s} {'var':>8s}  {'turn-lim':>8s}"
    )
    for name, ks in rows.items():
        b, v = summary(base, ks), summary(var, ks)
        print(
            f"{name:10s} {len(ks):4d}  {b['rr']:8.3f} {v['rr']:6.3f}  {b['calls']:10.2f} {v['calls']:5.2f}  "
            f"{b['cost']:9.5f} {v['cost']:8.5f}  {b['turn_limit']:.2f}>{v['turn_limit']:.2f}"
        )
    b, v = summary(base, keys), summary(var, keys)
    calls, cost, mrr = v["calls"] / b["calls"] - 1, v["cost"] / b["cost"] - 1, v["rr"] - b["rr"]
    p = {m: paired_p([var[k][m] for k in keys], [base[k][m] for k in keys]) for m in ("rr", "calls", "cost")}
    ci = bootstrap(base, var, keys)
    print(
        f"\ncalls {calls:+.1%} [{ci['calls'][0]:+.1%}, {ci['calls'][1]:+.1%}] (p={p['calls']:.3f})  "
        f"cost {cost:+.1%} [{ci['cost'][0]:+.1%}, {ci['cost'][1]:+.1%}] (p={p['cost']:.3f})  "
        f"MRR {mrr:+.3f} [{ci['rr'][0]:+.3f}, {ci['rr'][1]:+.3f}] (p={p['rr']:.3f})"
    )
    efficient = calls <= -CALLS_DROP or cost <= -COST_DROP
    kept = efficient and mrr >= -MRR_LOSS
    print(
        f"criterion (calls -{CALLS_DROP:.0%} or cost -{COST_DROP:.0%}, MRR >= -{MRR_LOSS}): "
        f"{'KEPT' if kept else 'NOT KEPT'}"
    )


if __name__ == "__main__":
    main()
