"""Title and abstract of the target paper of every development question, for a reranker-only metric.

On realistic questions the target is often missing from the candidate pool, so scoring a reranker on the
pool alone judges it on a few questions. finetune.py also inserts the target into each development pool:
that isolates the reranker and uses every question. Development sets only; the test sets are never read.

    uv run --extra eval --extra semantic python train/gold_docs.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from inspire_mcp.client import InspireClient  # noqa: E402

POOLS = (ROOT / "eval" / "cache" / "pools-citedev.json", ROOT / "eval" / "cache" / "pools-famous-longtail.json")
OUT = ROOT / "train" / "cache" / "gold_docs.json"


async def main() -> None:
    gold = json.loads(OUT.read_text()) if OUT.exists() else {}
    async with InspireClient() as client:
        for path in POOLS:
            for queries in json.loads(path.read_text()).values():
                for q in queries:
                    if str(q["recid"]) in gold:
                        continue
                    paper = await client.get_paper(str(q["recid"]))
                    if paper:
                        gold[str(q["recid"])] = {"recid": paper.recid, "title": paper.title, "abstract": paper.abstract}
    OUT.write_text(json.dumps(gold))
    print(f"{len(gold)} target papers in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
