"""An LLM as reranker on the frozen development pools: a reference point for the cross-encoders.

Claude Haiku reads the question and the 50 candidates (title and the start of the abstract) and returns
the ten that best match, best first. On the development sets only, with the same two variants as
train/finetune.py: the pool as the server sees it, and with the target inserted when it is missing.
If the LLM ranks far better than the cross-encoder, its rankings are worth distilling into it.

    uv run --extra eval python eval/llm_rerank.py [--sets citedev]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
from pathlib import Path

from anthropic import APIStatusError, AsyncAnthropic

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from agent_eval import PRICES, load_key  # noqa: E402

CACHE = HERE / "cache"
POOLS = {"citedev": CACHE / "pools-citedev.json", "famous": CACHE / "pools-famous-longtail.json"}
GOLD = HERE.parent / "train" / "cache" / "gold_docs.json"
RANKINGS = CACHE / "llm-rankings.json"
MODEL = "claude-haiku-5-5"
POOL = 50
ABSTRACT_CHARS = 500

SYSTEM = """You help physicists find one specific paper in INSPIRE-HEP. You are given what someone wrote \
about the paper they are looking for and a numbered list of candidate papers. Rank the candidates by how \
likely each one is to be that paper."""

PROMPT = """Text about the paper:
{query}

Candidates:
{candidates}

Return the numbers of the ten candidates most likely to be the paper, best first."""

SCHEMA = {
    "type": "object",
    "properties": {"ranking": {"type": "array", "items": {"type": "integer"}}},
    "required": ["ranking"],
    "additionalProperties": False,
}


def listing(pool: list[dict]) -> str:
    return "\n".join(f"[{i}] {p['title']}. {(p.get('abstract') or '')[:ABSTRACT_CHARS]}" for i, p in enumerate(pool, 1))


async def rank(client: AsyncAnthropic, query: str, pool: list[dict], sem: asyncio.Semaphore) -> dict:
    async with sem:
        try:
            resp = await client.messages.create(
                model=MODEL,
                max_tokens=4000,
                system=SYSTEM,
                messages=[{"role": "user", "content": PROMPT.format(query=query, candidates=listing(pool))}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            )
        except APIStatusError as exc:
            return {"error": f"HTTP {exc.status_code}"}
    text = next(b.text for b in resp.content if b.type == "text")
    numbers = json.loads(text)["ranking"]
    recids = [pool[n - 1]["recid"] for n in numbers if 1 <= n <= len(pool)]
    return {"recids": recids, "usage": {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", default="citedev,famous,longtail")
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()
    gold = json.loads(GOLD.read_text())
    cache: dict[str, dict] = json.loads(RANKINGS.read_text()) if RANKINGS.exists() else {}
    client = AsyncAnthropic(api_key=load_key(), max_retries=3)
    sem = asyncio.Semaphore(args.concurrency)
    loaded = {**json.loads(POOLS["citedev"].read_text()), **json.loads(POOLS["famous"].read_text())}
    jobs = []
    for name in args.sets.split(","):
        for q in loaded[name]:
            for variant in ("pool", "+target"):
                pool = q["pool"][:POOL]
                if variant == "+target" and q["recid"] not in {p["recid"] for p in pool}:
                    pool = [*pool[: POOL - 1], gold[str(q["recid"])]]
                key = hashlib.sha256(f"{MODEL}|{PROMPT}|{q['query']}|{[p['recid'] for p in pool]}".encode()).hexdigest()
                jobs.append((name, variant, q, pool, key[:16]))
    todo = [j for j in jobs if j[4] not in cache]
    results = await asyncio.gather(*(rank(client, j[2]["query"], j[3], sem) for j in todo))
    for j, r in zip(todo, results, strict=True):
        if "error" not in r:
            cache[j[4]] = r
    RANKINGS.write_text(json.dumps(cache))
    price = PRICES[MODEL]
    spent = sum(
        r["usage"]["input"] * price["input"] + r["usage"]["output"] * price["output"] for r in results if "usage" in r
    )
    scores: dict[str, list[float]] = {}
    for name, variant, q, _, key in jobs:
        ranked = cache.get(key, {}).get("recids", [])
        rank_ = ranked.index(q["recid"]) + 1 if q["recid"] in ranked else None
        scores.setdefault(f"{name}{'' if variant == 'pool' else variant}", []).append(1 / rank_ if rank_ else 0.0)
    for name, s in scores.items():
        print(f"{name:<18} MRR@10 {statistics.fmean(s):.3f}  ({len(s)} questions)")
    print(f"cost of this run: ${spent / 1e6:.4f}; errors: {sum('error' in r for r in results)}")


if __name__ == "__main__":
    asyncio.run(main())
