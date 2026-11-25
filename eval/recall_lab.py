"""Candidate-pool recall: how often the target paper is among the candidates the reranker will see.

On citation-context questions the target is in the default 50-candidate pool for about a third of the
questions, so no reranker can do better. This compares ways of building the INSPIRE queries, on the
development sets only (citedev, famous, longtail):

  first12   the server today: strict query, then OR of the first 12 content words in question order
  idfN      OR of the N content words that are rarest in the physics literature, in question order
  first+idf the two OR queries interleaved
  and2      plus an AND of the two rarest words (few, precise results); pairs4: AND of each pair of the 4 rarest

Word rarity comes from document frequencies over the ~110k abstracts of the training pools
(train/cache/pools.jsonl), which hold no evaluation paper. Searches are cached per query string.

    uv run --extra eval --extra semantic python eval/recall_lab.py
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE)]
from run_eval import load_queries, resolve_gold  # noqa: E402

from inspire_mcp.client import InspireClient, InspireError  # noqa: E402
from inspire_mcp.semantic import INSPIRE_RESERVED, STOPWORDS, TOKEN, _quote, interleave  # noqa: E402

CACHE = HERE / "cache"
SEARCHES = CACHE / "recall-searches.json"
TRAIN_POOLS = HERE.parent / "train" / "cache" / "pools.jsonl"
SETS = ("citedev", "famous", "longtail")
POOL = 50


def document_frequencies() -> tuple[Counter, int]:
    docs: dict[int, str] = {}
    for line in TRAIN_POOLS.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        for p in row["pool"]:
            docs.setdefault(p["recid"], f"{p['title']} {p.get('abstract') or ''}")
    df: Counter = Counter()
    for text in docs.values():
        df.update({t.lower().strip("-") for t in TOKEN.findall(text)})
    return df, len(docs)


def words(question: str) -> list[str]:
    seen: dict[str, None] = {}
    for tok in TOKEN.findall(question):
        low = tok.lower().strip("-")
        if len(low) >= 2 and low not in STOPWORDS and low not in INSPIRE_RESERVED:
            seen.setdefault(low, None)
    return list(seen)


def rarest(question: str, n: int, df: Counter, total: int) -> list[str]:
    terms = words(question)
    idf = {t: math.log((total + 1) / (df[t] + 1)) for t in terms}
    keep = set(sorted(terms, key=lambda t: -idf[t])[:n])
    return [t for t in terms if t in keep]


def or_query(terms: list[str]) -> str:
    return " or ".join(_quote(t) for t in terms)


class Searcher:
    def __init__(self, client: InspireClient) -> None:
        self.client = client
        self.cache: dict[str, list[int]] = json.loads(SEARCHES.read_text()) if SEARCHES.exists() else {}

    async def __call__(self, query: str, size: int) -> list[int]:
        key = f"{size}|{query}"
        if key not in self.cache:
            try:
                _, papers = await self.client.search_literature(query, size=size) if query else (0, [])
            except InspireError:
                papers = []
            self.cache[key] = [p.recid for p in papers]
        return self.cache[key]


async def pools(search: Searcher, question: str, df: Counter, total: int) -> dict[str, list[int]]:
    strict = await search(question, 25)
    first = await search(or_query(words(question)[:12]), POOL)
    idf12 = await search(or_query(rarest(question, 12, df, total)), POOL)
    top = sorted(rarest(question, 4, df, total), key=lambda t: df[t])  # rarest first
    and2 = await search(" and ".join(_quote(t) for t in top[:2]), 25) if len(top) >= 2 else []
    pairs = [await search(f"{_quote(a)} and {_quote(b)}", 10) for i, a in enumerate(top) for b in top[i + 1 :]]
    out = {
        "first12": first,
        "idf12": idf12,
        "first12+idf12": interleave_ids(first, idf12),
        "and2+first12+idf12": interleave_ids(and2, first, idf12),
        "pairs4+first12+idf12": interleave_ids(*pairs, first, idf12),
    }
    return {name: merge(strict, loose) for name, loose in out.items()}


def interleave_ids(*lists: list[int]) -> list[int]:
    class Id:  # interleave() works on papers: wrap the ids
        def __init__(self, recid: int) -> None:
            self.recid = recid

    return [p.recid for p in interleave(*[[Id(r) for r in x] for x in lists])]


def merge(strict: list[int], loose: list[int]) -> list[int]:
    return list(dict.fromkeys([*strict, *loose]))[:POOL]


async def main() -> None:
    df, total = document_frequencies()
    print(f"document frequencies over {total} training abstracts")
    async with InspireClient() as client:
        search = Searcher(client)
        for name in SETS:
            queries = load_queries(name)
            await resolve_gold(client, queries)
            hits: dict[str, list[bool]] = {}
            for q in queries:
                for strategy, pool in (await pools(search, q["query"], df, total)).items():
                    hits.setdefault(strategy, []).append(q["recid"] in pool)
            SEARCHES.write_text(json.dumps(search.cache))
            print(f"\n## {name} ({len(queries)} questions), target in the {POOL}-candidate pool")
            for strategy, h in hits.items():
                print(f"  {strategy:<16} {sum(h) / len(h):.2f}  ({sum(h)}/{len(h)})")
    example = load_queries("citedev")[0]["query"]
    print(f"\nexample: {example}\n  first12: {words(example)[:12]}\n  idf8:    {rarest(example, 8, df, total)}")


if __name__ == "__main__":
    asyncio.run(main())
