"""Claude Haiku as teacher: it ranks the candidates of every training question, for distillation.

On the development pools Haiku ranks far better than any cross-encoder (eval/llm_rerank.py: MRR 0.42 on
citedev with the target inserted, against 0.13 for MiniLM and 0.23 for bge-reranker-base). Its rankings
give the student a signal for every candidate, not only "target against the rest", and they tell which
"negatives" in fact answer the question too, so they are not pushed down as wrong.

For each training question: the target and the first 19 other candidates of its INSPIRE pool, in a random
order (the target's position must not be learnable), titles and the first 500 characters of abstracts.
Haiku returns the ten best, best first. Cached per question; the cost is printed.

    uv run --extra eval python train/teacher.py [--limit 100]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

from anthropic import AsyncAnthropic

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "train")]
from agent_eval import PRICES, load_key  # noqa: E402
from llm_rerank import MODEL, rank  # noqa: E402

CACHE = ROOT / "train" / "cache"
OUT = CACHE / "teacher.json"
SOURCES = (CACHE / "s2_judged.jsonl", CACHE / "questions.jsonl")  # real citation sentences first
CANDIDATES = 20


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--budget", type=float, default=2.5, help="stop asking once this many dollars are spent")
    args = parser.parse_args()
    papers = {p["recid"]: p for p in read_jsonl(CACHE / "papers.jsonl")}
    pools = {r["id"]: r for r in read_jsonl(CACHE / "pools.jsonl")}
    done: dict[str, dict] = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = []
    for source in SOURCES:
        for q in read_jsonl(source):
            if q["id"] in done or q["id"] not in pools:
                continue
            others = [p for p in pools[q["id"]]["pool"] if p["recid"] != q["recid"]][: CANDIDATES - 1]
            cands = [papers[q["recid"]], *others]
            random.Random(q["id"]).shuffle(cands)
            todo.append((q, cands))
    todo = todo[: args.limit]
    client = AsyncAnthropic(api_key=load_key(), max_retries=3)
    sem = asyncio.Semaphore(args.concurrency)
    price = PRICES[MODEL]
    spent = 0.0
    for i in range(0, len(todo), 200):  # in chunks, saving as it goes and checking the budget
        chunk = todo[i : i + 200]
        results = await asyncio.gather(*(rank(client, q["query"], cands, sem) for q, cands in chunk))
        for (q, cands), r in zip(chunk, results, strict=True):
            if "error" not in r:
                done[q["id"]] = {"candidates": [c["recid"] for c in cands], "ranking": r["recids"]}
                spent += r["usage"]["input"] * price["input"] / 1e6 + r["usage"]["output"] * price["output"] / 1e6
        OUT.write_text(json.dumps(done))
        print(f"{min(i + 200, len(todo))}/{len(todo)} questions, ${spent:.3f} spent", flush=True)
        if spent > args.budget:
            print("budget reached, stopping")
            break
    agree = [d["ranking"][:1] == [int(k.split("-")[0][1:])] for k, d in done.items() if d["ranking"]]
    print(f"teacher rankings: {len(done)}; target ranked first by the teacher: {sum(agree) / max(1, len(agree)):.2f}")


if __name__ == "__main__":
    asyncio.run(main())
