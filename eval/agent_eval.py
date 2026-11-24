"""Agentic baseline: can an LLM that drives INSPIRE search by itself match semantic_search?

Arms, on the same questions as run_eval.py:
  A  the agent has only `search_literature` (INSPIRE syntax) and may reformulate as often as it likes
  B  the agent also has `semantic_search`, i.e. the server as it is meant to be used
The agent ends by calling `submit_ranking` with up to 10 INSPIRE record ids, scored like every other
system (R@1, R@10, MRR@10), plus tool calls, tokens, cost and latency per question.

The tools call the same functions as the MCP server. The API key is read from a file and passed to the
client explicitly, so no other credential on the machine is picked up. No server-side fallback: a
silent switch to another model would contaminate the measurement, so refusals are recorded instead.

Usage:
  uv run --extra eval --extra semantic python eval/agent_eval.py --model claude-sonnet-5-5 --limit 5
  uv run --extra eval --extra semantic python eval/agent_eval.py --model claude-haiku-5-5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic, BadRequestError, RateLimitError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from run_eval import (  # noqa: E402
    QUERY_SETS,
    bootstrap_ci,
    load_queries,
    paired_permutation_p,
    per_query_scores,
    rank_of,
    resolve_gold,
)

from inspire_mcp.client import InspireClient, InspireError  # noqa: E402
from inspire_mcp.semantic import CrossEncoderReranker, semantic_search  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE / "results"
KEY_FILE = Path(os.environ.get("INSPIRE_EVAL_KEY_FILE", "~/.config/inspire-mcp/anthropic_key")).expanduser()
# USD per million tokens, Anthropic first-party list prices cached on 2026-11-03: verify before quoting.
PRICES = {
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.10, "cache_write": 2.50},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125},
}
MAX_TURNS = 8
SEED = 13

SYSTEM = """You help physicists find papers in INSPIRE-HEP, the high energy physics literature database.
The user describes one specific paper. Find it with the tools, then call submit_ranking with the INSPIRE
record ids (recid) of your best candidates, most likely first, at most 10. Always finish by calling
submit_ranking, even if you are unsure; never answer in plain text instead.

search_literature uses INSPIRE syntax: t <title words>, a <author>, cn <collaboration>, j <journal>,
date > 2015, topcite 500+ (citations), k <keyword>; combine with and / or / not. Plain words are
combined with AND, so long sentences often return nothing: use a few distinctive words."""

SEARCH_TOOL = {
    "name": "search_literature",
    "description": "Search INSPIRE-HEP with INSPIRE query syntax. Returns up to `size` papers with recid, title, "
    "year, citations, first author or collaboration and the start of the abstract.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "INSPIRE query, e.g. 't warped and t hierarchy'"},
            "size": {"type": "integer", "description": "results to return, 1-25"},
            "sort": {"type": "string", "enum": ["mostrecent", "mostcited", "bestmatch"]},
        },
        "required": ["query", "size", "sort"],
        "additionalProperties": False,
    },
    "strict": True,
}
SEMANTIC_TOOL = {
    "name": "semantic_search",
    "description": "Find papers for a description in plain English: retrieves candidates from INSPIRE and ranks "
    "them by how well each title and abstract matches. Use it first for natural-language descriptions.",
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "the description in plain English"},
            "prefer_influential": {
                "type": "boolean",
                "description": "true when the user asks for a foundational, well-known paper",
            },
        },
        "required": ["question", "prefer_influential"],
        "additionalProperties": False,
    },
    "strict": True,
}
SUBMIT_TOOL = {
    "name": "submit_ranking",
    "description": "Submit the final answer: INSPIRE recids of the best candidates, most likely first, at most 10.",
    "input_schema": {
        "type": "object",
        "properties": {"recids": {"type": "array", "items": {"type": "integer"}}},
        "required": ["recids"],
        "additionalProperties": False,
    },
    "strict": True,
}


def load_key() -> str:
    key = KEY_FILE.read_text().strip()
    if not key.startswith("sk-ant-"):
        raise SystemExit(f"{KEY_FILE} does not contain an Anthropic API key")
    return key


def brief(papers: list[Any]) -> str:
    rows = []
    for p in papers:
        who = p.collaborations[0] if p.collaborations else (p.authors[0] if p.authors else "")
        abstract = (p.abstract or "")[:300]
        rows.append(
            {
                "recid": p.recid,
                "title": p.title,
                "year": p.year,
                "citations": p.citation_count,
                "by": who,
                "abstract": abstract,
            }
        )
    return json.dumps(rows, ensure_ascii=False)


class Tools:
    def __init__(self, inspire: InspireClient, reranker: CrossEncoderReranker) -> None:
        self.inspire, self.reranker = inspire, reranker

    async def run(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        try:
            if name == "search_literature":
                size = max(1, min(int(args.get("size", 10)), 25))
                sort = args.get("sort")
                total, papers = await self.inspire.search_literature(
                    args["query"], size=size, sort=None if sort == "bestmatch" else sort
                )
                return json.dumps({"total": total}) + "\n" + brief(papers), False
            if name == "semantic_search":
                pool = 50
                result = await semantic_search(
                    self.inspire,
                    args["question"],
                    k=10,
                    pool=pool,
                    reranker=self.reranker,
                    influential=pool // 2 if args.get("prefer_influential") else 0,
                )
                return brief(result.papers), False
        except InspireError as exc:
            return f"error: {exc}", True
        return f"error: unknown tool {name}", True


async def run_question(client: AsyncAnthropic, model: str, tools: Tools, arm: str, question: str) -> dict[str, Any]:
    tool_defs = [SEARCH_TOOL, SUBMIT_TOOL] if arm == "A" else [SEMANTIC_TOOL, SEARCH_TOOL, SUBMIT_TOOL]
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    calls: list[dict[str, Any]] = []
    start = time.perf_counter()
    outcome, ranking = "no_submission", []
    for _turn in range(MAX_TURNS):
        for attempt in range(5):
            try:
                response = await client.messages.create(
                    model=model,
                    max_tokens=16000,
                    system=SYSTEM,
                    tools=tool_defs,
                    messages=messages,
                    output_config={"effort": "medium"},
                    cache_control={"type": "ephemeral"},
                )
                break
            except RateLimitError:
                await asyncio.sleep(10 * (attempt + 1))
        else:
            outcome = "rate_limited"
            break
        u = response.usage
        usage["input"] += u.input_tokens
        usage["output"] += u.output_tokens
        usage["cache_read"] += u.cache_read_input_tokens or 0
        usage["cache_write"] += u.cache_creation_input_tokens or 0
        if response.stop_reason == "refusal":
            outcome = "refusal"
            break
        messages.append({"role": "assistant", "content": response.content})  # thinking blocks passed back unchanged
        uses = [b for b in response.content if b.type == "tool_use"]
        if not uses:
            text = " ".join(b.text for b in response.content if b.type == "text")
            ranking = [int(x) for x in re.findall(r"\b\d{4,8}\b", text)][:10]  # salvage ids written as text
            outcome = "text_answer" if ranking else "no_submission"
            break
        submit = next((b for b in uses if b.name == "submit_ranking"), None)
        if submit:
            ranking = [int(r) for r in submit.input.get("recids", [])][:10]
            outcome = "submitted"
            break
        results = []
        for b in uses:
            content, is_error = await tools.run(b.name, dict(b.input))
            calls.append({"tool": b.name, "input": dict(b.input), "error": is_error})
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": content, "is_error": is_error})
        messages.append({"role": "user", "content": results})
    else:
        outcome = "turn_limit"
    price = PRICES[model]
    cost = (
        usage["input"] * price["input"]
        + usage["output"] * price["output"]
        + usage["cache_read"] * price["cache_read"]
        + usage["cache_write"] * price["cache_write"]
    ) / 1e6
    return {
        "ranking": ranking,
        "outcome": outcome,
        "tool_calls": calls,
        "usage": usage,
        "cost_usd": cost,
        "latency_s": time.perf_counter() - start,
    }


def single_call_baseline() -> dict[str, dict[str, list[int | None]]]:
    """Ranks of the plain semantic_search tool (one call, cross-encoder) from the latest run_eval.py result."""
    latest = sorted(RESULTS.glob("results-*.json"))[-1]
    live = json.loads(latest.read_text())["live"]
    return {name: live[name]["ranks"]["or+xenc"] for name in live}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(PRICES))
    parser.add_argument("--limit", type=int, default=0, help="questions per set (0 = all); a pilot samples evenly")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    client = AsyncAnthropic(api_key=load_key(), max_retries=3)
    reranker = await asyncio.to_thread(CrossEncoderReranker)
    sets = {name: load_queries(name) for name in QUERY_SETS}
    if args.limit:
        rng = random.Random(SEED)
        sets = {name: rng.sample(qs, min(args.limit, len(qs))) for name, qs in sets.items()}
    sem = asyncio.Semaphore(args.concurrency)
    async with InspireClient() as inspire:
        for queries in sets.values():
            await resolve_gold(inspire, queries)
        tools = Tools(inspire, reranker)

        async def one(q: dict[str, Any], arm: str) -> tuple[str, str, dict[str, Any]]:
            async with sem:
                try:
                    out = await run_question(client, args.model, tools, arm, q["query"])
                except BadRequestError as exc:
                    out = {
                        "ranking": [],
                        "outcome": f"api_error: {exc.message[:120]}",
                        "tool_calls": [],
                        "usage": {},
                        "cost_usd": 0.0,
                        "latency_s": 0.0,
                    }
                print(
                    f"  {arm} {q['id']:>4} {out['outcome']:<13} calls={len(out['tool_calls'])} "
                    f"rank={rank_of(out['ranking'], q['recid'])} ${out['cost_usd']:.4f}",
                    flush=True,
                )
                return q["id"], arm, out

        jobs = [one(q, arm) for qs in sets.values() for q in qs for arm in ("A", "B")]
        done = await asyncio.gather(*jobs)

    runs = {(qid, arm): out for qid, arm, out in done}
    baseline = single_call_baseline()
    rng = random.Random(SEED)
    report: dict[str, Any] = {
        "date": date.today().isoformat(),
        "model": args.model,
        "effort": "medium",
        "max_turns": MAX_TURNS,
        "prices_usd_per_mtok": PRICES[args.model],
        "sets": {},
    }
    for name, queries in sets.items():
        ranks = {
            "single_call": [baseline[name].get(q["id"]) for q in queries],
            "agent_A": [rank_of(runs[(q["id"], "A")]["ranking"][:10], q["recid"]) for q in queries],
            "agent_B": [rank_of(runs[(q["id"], "B")]["ranking"][:10], q["recid"]) for q in queries],
        }
        summary = {}
        for system, rs in ranks.items():
            s = per_query_scores(rs)
            summary[system] = {m: round(statistics.fmean(v), 3) for m, v in s.items()}
            summary[system]["MRR@10_ci95"] = [round(x, 3) for x in bootstrap_ci(s["MRR@10"], rng)]
        mrr = {k: per_query_scores(v)["MRR@10"] for k, v in ranks.items()}
        for arm in ("A", "B"):
            outs = [runs[(q["id"], arm)] for q in queries]
            summary[f"agent_{arm}"].update(
                {
                    "mean_tool_calls": round(statistics.fmean(len(o["tool_calls"]) for o in outs), 2),
                    "mean_cost_usd": round(statistics.fmean(o["cost_usd"] for o in outs), 5),
                    "median_latency_s": round(statistics.median(o["latency_s"] for o in outs), 1),
                    "outcomes": {k: sum(o["outcome"] == k for o in outs) for k in {o["outcome"] for o in outs}},
                }
            )
        report["sets"][name] = {
            "n": len(queries),
            "summary": summary,
            "paired_permutation_p_mrr": {
                f"{x} vs {y}": round(paired_permutation_p(mrr[x], mrr[y], rng), 4)
                for x, y in (("agent_B", "agent_A"), ("agent_A", "single_call"), ("agent_B", "single_call"))
            },
            "per_query": [
                {
                    "id": q["id"],
                    "gold": q["recid"],
                    **{
                        arm: {
                            "rank": rank_of(runs[(q["id"], arm)]["ranking"][:10], q["recid"]),
                            "outcome": runs[(q["id"], arm)]["outcome"],
                            "queries": [c["input"] for c in runs[(q["id"], arm)]["tool_calls"]],
                            "cost_usd": round(runs[(q["id"], arm)]["cost_usd"], 5),
                        }
                        for arm in ("A", "B")
                    },
                }
                for q in queries
            ],
        }
    total = sum(o["cost_usd"] for o in runs.values())
    report["total_cost_usd"] = round(total, 4)
    RESULTS.mkdir(exist_ok=True)
    tag = f"-pilot{args.limit}" if args.limit else ""
    out = RESULTS / f"agent-{args.model}-{report['date']}{tag}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for name, block in report["sets"].items():
        print(f"\n## {name} ({block['n']} questions)")
        for system, m in block["summary"].items():
            extra = (
                f" calls={m['mean_tool_calls']} ${m['mean_cost_usd']}/q {m['outcomes']}"
                if "mean_tool_calls" in m
                else ""
            )
            print(
                f"  {system:<12} R@1={m['R@1']:.2f} R@10={m['R@10']:.2f} MRR={m['MRR@10']:.2f} "
                f"{m['MRR@10_ci95']}{extra}"
            )
        print(f"  p: {block['paired_permutation_p_mrr']}")
    print(f"\ntotal cost ${total:.4f}  saved {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    asyncio.run(main())
