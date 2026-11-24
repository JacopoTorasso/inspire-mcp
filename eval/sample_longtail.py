"""Reproducible sample of mid-cited INSPIRE papers for the long-tail query set.

Picks one random record from random result pages of SAMPLE_QUERY (seeded), keeping records
with an arXiv id and an abstract of at least 400 characters. Prints them as JSON lines; the
questions in queries_longtail.jsonl were written from this output.
"""

import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inspire_mcp.client import InspireClient  # noqa: E402

SAMPLE_QUERY = "topcite 20->200 and date 2015->2022 and tc p"
SEED, N, PAGE_SIZE, MAX_PAGE = 2026, 40, 25, 400


async def main() -> None:
    rng = random.Random(SEED)
    picked: dict[int, dict] = {}
    async with InspireClient() as client:
        while len(picked) < N:
            _, papers = await client.search_literature(SAMPLE_QUERY, size=PAGE_SIZE, page=rng.randint(1, MAX_PAGE))
            ok = [p for p in papers if p.arxiv_id and p.abstract and len(p.abstract) >= 400 and p.recid not in picked]
            if ok:
                p = rng.choice(ok)
                picked[p.recid] = {
                    "arxiv": p.arxiv_id,
                    "recid": p.recid,
                    "title": p.title,
                    "abstract": p.abstract,
                    "citations": p.citation_count,
                    "categories": p.categories,
                }
    for rec in picked.values():
        print(json.dumps(rec, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
