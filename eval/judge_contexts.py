"""Keep only citation contexts that describe the cited paper well enough to serve as a known-item question.

Many citing sentences only use a paper ("we follow the method of [12]", "see [12] for a review") without
saying what it did. An LLM judge reads each sentence, with [CITED] where the target was cited, and says
whether it describes that paper's own content specifically enough for an expert to pick it out.

The judge sees the sentence only: never the title or the abstract. Shown the abstract, it would favour
sentences that echo it, bringing back the vocabulary bias these questions exist to remove.

Verdicts are cached per (model, sentence) in eval/cache/judge.json, so a rerun only pays for new sentences.
The API key is read like agent_eval.py does.

    uv run --extra eval python eval/judge_contexts.py [batch.jsonl ...]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

from anthropic import APIStatusError, AsyncAnthropic

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from agent_eval import PRICES, load_key  # noqa: E402

CACHE = HERE / "cache"
IN = CACHE / "citation_contexts.jsonl"
OUT = CACHE / "citation_contexts_judged.jsonl"
VERDICTS = CACHE / "judge.json"
MODEL = "claude-haiku-5-5"

SYSTEM = """You review sentences taken from physics papers. In each sentence, [CITED] marks where one \
earlier paper is cited. You decide whether the sentence could serve as a description of that cited paper \
for someone trying to find it."""

PROMPT = """Sentence:
{context}

Does this sentence say what the cited paper itself did, proposed, computed, measured or found, \
specifically enough that a physicist who knows the field could pick out that one paper among others \
on the same topic?

Answer no when the citation only supports a general statement or background fact, marks a method, code, \
dataset or result being used or compared without saying what it is, points to a review or a discussion, \
or when the sentence is mostly about the citing paper's own work."""

SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string", "description": "one short sentence"},
        "descriptive": {"type": "boolean"},
    },
    "required": ["reason", "descriptive"],
    "additionalProperties": False,
}


def key(context: str) -> str:
    return hashlib.sha256(f"{MODEL}|{PROMPT}|{context}".encode()).hexdigest()[:16]


async def judge(client: AsyncAnthropic, context: str, sem: asyncio.Semaphore) -> dict:
    async with sem:
        try:
            resp = await client.messages.create(
                model=MODEL,
                max_tokens=4000,
                system=SYSTEM,
                messages=[{"role": "user", "content": PROMPT.format(context=context)}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            )
        except APIStatusError as exc:
            return {"error": f"HTTP {exc.status_code}"}
    usage = {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}
    if resp.stop_reason == "refusal":
        return {"refusal": True, "usage": usage}
    text = next(b.text for b in resp.content if b.type == "text")
    return {**json.loads(text), "usage": usage}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("inputs", type=Path, nargs="*", default=[IN], help="batches of citation_contexts.py")
    args = parser.parse_args()
    rows = [json.loads(line) for f in args.inputs for line in f.read_text().splitlines() if line.strip()]
    cache: dict[str, dict] = json.loads(VERDICTS.read_text()) if VERDICTS.exists() else {}
    todo = [r for r in rows if key(r["context"]) not in cache]
    client = AsyncAnthropic(api_key=load_key(), max_retries=3)
    sem = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*(judge(client, r["context"], sem) for r in todo))
    for r, verdict in zip(todo, results, strict=True):
        if "error" not in verdict:
            cache[key(r["context"])] = verdict
    VERDICTS.write_text(json.dumps(cache, indent=1))

    price = PRICES[MODEL]
    spent = sum(
        v["usage"]["input"] * price["input"] + v["usage"]["output"] * price["output"] for v in results if "usage" in v
    )
    kept = []
    for r in rows:
        v = cache.get(key(r["context"]), {})
        if v.get("descriptive"):
            kept.append({**r, "judge_reason": v["reason"]})
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    errors = sum("error" in v for v in results)
    refusals = sum(bool(v.get("refusal")) for v in results)
    print(f"sentences: {len(rows)}, judged now: {len(todo)}, errors: {errors}, refusals: {refusals}")
    print(f"descriptive: {len(kept)} of {len(rows)}, for {len({r['recid'] for r in kept})} target papers")
    print(f"cost of this run: ${spent / 1e6:.4f} (list prices cached 2026-11-03)")
    if kept:
        shares = sorted(r["abstract_share"] for r in kept)
        print(f"share of words found in the target abstract, kept sentences: median {shares[len(shares) // 2]:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
