"""Titles and arXiv abstracts of every INSPIRE literature record that has one, for a local semantic index.

Only abstracts whose source is arXiv are kept: arXiv metadata is CC0, while INSPIRE notes that some fields
(publisher abstracts among them, possibly) carry restrictions. The search API returns at most 10,000 results
per query, so the harvest walks ranges of earliest date (`de`), one month or less, small enough to stay under
it, 1000 records per page,
within INSPIRE's documented limit of 15 requests per 5 s (kept by InspireClient). Resumable: each finished
range is appended to the output and skipped on a rerun.

    uv run --extra eval --extra semantic python train/harvest_corpus.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from inspire_mcp.client import InspireClient  # noqa: E402

OUT = ROOT / "train" / "cache" / "corpus.jsonl"
DONE = ROOT / "train" / "cache" / "corpus.ranges"
QUERY = "abstracts.source:arXiv"
FIELDS = "control_number,titles.title,abstracts,arxiv_eprints.value,citation_count,earliest_date"
PAGE = 1000
LIMIT = 10_000
FIRST_YEAR = 1960
CONCURRENCY = 3


def record(m: dict) -> dict | None:
    abstract = next((a["value"] for a in m.get("abstracts", []) if a.get("source") == "arXiv"), None)
    if not abstract or not m.get("titles"):
        return None
    return {
        "recid": m["control_number"],
        "title": m["titles"][0]["title"],
        "abstract": abstract,
        "arxiv": (m.get("arxiv_eprints") or [{}])[0].get("value"),
        "citations": m.get("citation_count", 0),
        "year": int(str(m.get("earliest_date", "0"))[:4] or 0),
    }


async def harvest_range(client: InspireClient, lo: date, hi: date) -> list[dict] | None:
    """Records whose earliest date is in [lo, hi], or None when they are more than LIMIT (split the range)."""
    q = f"{QUERY} and de {lo.isoformat()}->{hi.isoformat()}"
    rows: list[dict] = []
    page = 1
    while True:
        data = await client._get("/literature", {"q": q, "size": PAGE, "page": page, "fields": FIELDS})
        total = data["hits"]["total"]
        if total > LIMIT:
            return None
        rows += [r for h in data["hits"]["hits"] if (r := record(h["metadata"]))]
        if page * PAGE >= total:
            return rows
        page += 1


def months(first: date, last: date) -> list[tuple[date, date]]:
    out, start = [], first
    while start <= last:
        nxt = date(start.year + start.month // 12, start.month % 12 + 1, 1)
        out.append((start, min(nxt - timedelta(days=1), last)))
        start = nxt
    return out


async def main() -> None:
    done = set(DONE.read_text().splitlines()) if DONE.exists() else set()
    async with InspireClient() as client:
        expected = (await client._get("/literature", {"q": QUERY, "size": 1, "fields": "control_number"}))["hits"][
            "total"
        ]
        todo = [r for r in months(date(FIRST_YEAR, 1, 1), date.today()) if f"{r[0]} {r[1]}" not in done]
        lock, sem = asyncio.Lock(), asyncio.Semaphore(CONCURRENCY)
        count = [0]

        async def work(lo: date, hi: date) -> None:
            async with sem:
                rows = await harvest_range(client, lo, hi)
            if rows is None:  # more than LIMIT records: halve the range
                if lo == hi:
                    raise SystemExit(f"more than {LIMIT} records on {lo}: cannot split further")
                mid = lo + (hi - lo) // 2
                await asyncio.gather(work(lo, mid), work(mid + timedelta(days=1), hi))
                return
            async with lock:
                with OUT.open("a") as f:
                    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
                with DONE.open("a") as f:
                    f.write(f"{lo} {hi}\n")
                count[0] += len(rows)
                if rows:
                    print(f"{lo}..{hi}: {len(rows)} records, {count[0]} this run", flush=True)

        OUT.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.gather(*(work(lo, hi) for lo, hi in todo))
    recids = {json.loads(line)["recid"] for line in OUT.open()}
    print(f"{len(recids)} distinct records in {OUT.relative_to(ROOT)}; INSPIRE reports {expected} for {QUERY!r}")


if __name__ == "__main__":
    asyncio.run(main())
