"""Real citation sentences for the training papers, from the Semantic Scholar API.

The Haiku questions of make_data.py are short summaries of a paper; the citation-context development set
showed that a cross-encoder trained on them does not learn how a citing author writes. Semantic Scholar
returns, in one request per paper, the sentences in which other papers cite it (extracted from PDFs, so
formulas are mangled: acceptable for training, which is why the test set comes from LaTeX instead).

Kept, as in eval/citation_contexts.py: sentences with exactly one numeric citation marker that cites one
paper ("[104]", not "[11-18]" or "[21, 22]"), no author-year citation, 8 to 60 words, no three consecutive
title words; at most 3 per training paper, from different citing papers. The output has the citation as
[CITED], for eval/judge_contexts.py, and without it, as the question.

    uv run --extra eval --extra semantic python train/s2_contexts.py
    uv run --extra eval python eval/judge_contexts.py --out train/cache/s2_judged.jsonl train/cache/s2_contexts.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]
from citation_contexts import abstract_share  # noqa: E402
from run_eval import title_overlap  # noqa: E402

from inspire_mcp.client import USER_AGENT  # noqa: E402
from inspire_mcp.semantic import _tokens  # noqa: E402

CACHE = ROOT / "train" / "cache"
RAW = CACHE / "s2"
OUT = CACHE / "s2_contexts.jsonl"
API = "https://api.semanticscholar.org/graph/v1/paper/arXiv:{}/citations?fields=contexts&limit=100"
DELAY = 2.0  # the public API without a key shares one rate limit among all its users: 429s are frequent
ATTEMPTS = 8
PER_PAPER = 3
MARKER = re.compile(r"\[[^\]]*\]")
SINGLE = re.compile(r"\[\d{1,3}\]")
AUTHOR_YEAR = re.compile(r"et al|\(\D*\d{4}[a-z]?\)|\b(19|20)\d\d[a-z]?\)")


async def fetch(http: httpx.AsyncClient, arxiv: str, last: list[float]) -> dict | None:
    path = RAW / (arxiv.replace("/", "_") + ".json")
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(ATTEMPTS):
        wait = min(60.0, DELAY * 2**attempt) - (time.monotonic() - last[0])  # 2 s, 4 s, ... up to a minute
        if wait > 0:
            await asyncio.sleep(wait)
        last[0] = time.monotonic()
        try:
            resp = await http.get(API.format(arxiv))
        except httpx.TransportError:
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            continue
        data = resp.json() if resp.status_code == 200 else {"data": []}
        RAW.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return data
    return None  # transient failure: nothing cached, a rerun tries again


def usable(sentence: str, title: str) -> tuple[str, str] | None:
    """(question, context with [CITED]) or None."""
    s = sentence.strip(" …").replace("ﬁ", "fi").replace("ﬂ", "fl")
    markers = MARKER.findall(s)
    if len(markers) != 1 or not SINGLE.fullmatch(markers[0]) or AUTHOR_YEAR.search(s):
        return None
    context = MARKER.sub("[CITED]", s)
    question = re.sub(r"\s+", " ", MARKER.sub(" ", s)).strip(" ,;:")
    if len(_tokens(question)) < 8 or len(question.split()) > 60:
        return None
    if title_overlap(question, title):
        return None
    return question, context


async def main(offline: bool, out: Path) -> None:
    papers = [json.loads(line) for line in (CACHE / "papers.jsonl").read_text().splitlines() if line.strip()]
    rows, last, failed = [], [0.0], 0
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=60) as http:
        for n, p in enumerate(papers, 1):
            cached = RAW / (p["arxiv"].replace("/", "_") + ".json")
            if offline and not cached.exists():
                continue
            data = await fetch(http, p["arxiv"], last)
            if data is None:
                failed += 1
                continue
            kept, seen = 0, set()
            for citing in data.get("data") or []:
                for sentence in citing.get("contexts") or []:
                    pair = usable(sentence, p["title"])
                    if pair and pair[0] not in seen:
                        seen.add(pair[0])
                        rows.append(
                            {
                                "id": f"s{p['recid']}-{kept}",
                                "recid": p["recid"],
                                "arxiv": p["arxiv"],
                                "title": p["title"],
                                "query": pair[0],
                                "context": pair[1],
                                "abstract_share": round(abstract_share(pair[0], p["abstract"]), 3),
                            }
                        )
                        kept += 1
                        break  # one sentence per citing paper
                if kept >= PER_PAPER:
                    break
            if n % 100 == 0:
                print(f"papers {n}/{len(papers)}, sentences {len(rows)}", flush=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} sentences for {len({r['recid'] for r in rows})} papers, {failed} requests failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="only papers already downloaded")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    asyncio.run(main(args.offline, args.out))
