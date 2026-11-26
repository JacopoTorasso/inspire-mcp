"""Development set of agent-written queries: what semantic_search actually receives.

An assistant never passes the user's text to semantic_search as it is: on the citation dev set Claude Haiku
rewrote every sentence (0 of 64 verbatim) into about 12 words, against 25 for the sentence. So the tool's
real input is neither a citing sentence nor a paraphrase of the abstract, and rerankers should be judged
on it. This collects, from the tool calls logged by agent_eval.py on the development sets only (famous,
longtail, citedev; the held-out and citation test sets are left alone), the first question each agent
passed to semantic_search, and freezes a 100-candidate pool for each with the current candidates(), in
the format eval/rerank_lab.py reads (`--sets agentdev`). Everything stays in eval/cache/: the citedev
queries paraphrase sentences of arXiv papers.

    uv run --extra eval --extra semantic python eval/agent_queries.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from inspire_mcp.client import InspireClient  # noqa: E402
from inspire_mcp.semantic import candidates  # noqa: E402

RESULTS = HERE / "results"
CACHE = HERE / "cache"
OUT = CACHE / "pools-agentdev.json"
DEV_SETS = ("famous", "longtail", "citedev")


def agent_queries() -> dict[str, list[dict]]:
    """{"agent-<set>": [{id, query, recid, model}]}, one per (question, agent model)."""
    out: dict[str, dict[str, dict]] = {}
    for path in sorted(RESULTS.glob("agent-*.json")):
        if "heldout" in path.name or "pilot" in path.name:
            continue
        report = json.loads(path.read_text())
        model = report["model"].replace("claude-", "")
        for name, block in report["sets"].items():
            if name not in DEV_SETS:
                continue
            for q in block.get("per_query", []):
                asked = [c["question"] for c in q.get("B", {}).get("queries", []) if "question" in c]
                if asked:
                    key = f"{q['id']}-{model}"
                    out.setdefault(f"agent-{name}", {})[key] = {
                        "id": key,
                        "query": asked[0],
                        "recid": q["gold"],
                        "model": model,
                    }
    return {name: list(rows.values()) for name, rows in out.items()}


async def main() -> None:
    sets = agent_queries()
    frozen = json.loads(OUT.read_text()) if OUT.exists() else {}
    async with InspireClient() as client:
        for name, rows in sets.items():
            done = {q["id"]: q for q in frozen.get(name, [])}
            for q in rows:
                if q["id"] not in done:
                    docs, _, _ = await candidates(client, q["query"], pool=100)
                    done[q["id"]] = {**q, "pool": [d.to_dict() for d in docs]}
            frozen[name] = list(done.values())
            OUT.write_text(json.dumps(frozen))
    for name, rows in frozen.items():
        at50 = sum(q["recid"] in [p["recid"] for p in q["pool"][:50]] for q in rows)
        words = sorted(len(q["query"].split()) for q in rows)
        print(f"{name}: {len(rows)} queries, median {words[len(words) // 2]} words, target in pool@50 {at50}")


if __name__ == "__main__":
    asyncio.run(main())
