"""Training data for the cross-encoder: questions written by Claude Haiku, hard negatives from INSPIRE pools.

Three resumable steps, each cached in train/cache/ (git-ignored):

  sample     training papers from INSPIRE: 3/4 from the longtail population (20-200 citations, 2015-2022),
             1/4 well cited (250+), never a paper of any evaluation set (development or test), so the
             model is judged on papers it has not seen.
  questions  Haiku reads title and abstract and writes, as a physicist recalling the paper a year later
             would, two citation-style sentences and two search requests; the first of each style that
             passes the filters is kept. Questions that share more than half
             of their words with the abstract are dropped: real citation sentences of the development set
             have a median share of 0.33 and 0.46 at the 75th percentile, and the abstract-written longtail
             questions 0.75, the lexical bias the model must not learn. Questions that repeat three title
             words are dropped too, like in the evaluation sets.
  pools      for each question, the 50 candidates semantic_search would rerank (candidates()), with
             every evaluation paper removed: the hard negatives.

    uv run --extra eval --extra semantic python train/make_data.py sample --n 3000
    uv run --extra eval --extra semantic python train/make_data.py questions [--limit 20]
    uv run --extra eval --extra semantic python train/make_data.py pools
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]
from agent_eval import PRICES, load_key  # noqa: E402
from citation_contexts import abstract_share  # noqa: E402
from run_eval import title_overlap  # noqa: E402

from inspire_mcp.client import InspireClient  # noqa: E402
from inspire_mcp.semantic import candidates  # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache"
PAPERS = CACHE / "papers.jsonl"
QUESTIONS = CACHE / "questions.json"  # raw Haiku output per recid, so a rerun pays only for new papers
DATA = CACHE / "questions.jsonl"  # filtered questions
POOLS = CACHE / "pools.jsonl"
MODEL = "claude-haiku-5-5"
MAX_SHARE = 0.5
POPULATIONS = (("topcite 20->200 and date 2015->2022 and tc p", 400, 0.75), ("topcite 250+ and tc p", 200, 0.25))
PAGE_SIZE, PER_PAGE = 25, 8
EVAL_SETS = ("queries.jsonl", "queries_longtail.jsonl", "queries_heldout.jsonl")
EVAL_FROZEN = ("queries_citations.jsonl", "queries_citations_dev.jsonl")

SYSTEM = """You write search questions for a high energy physics literature search engine. You are shown one \
paper. Write what physicists who know this paper would write about it from memory, in their own words."""

PROMPT = """Title: {title}

Abstract: {abstract}

Imagine you read this paper a year ago and remember only its main point. Write four short texts that \
point to it:
- two sentences as they would appear in other papers citing it, the citation itself left out, for example \
"The two-loop corrections to this decay were first computed in";
- two requests a physicist would type to an assistant to find it, for example "paper that measured ...".
The second sentence and the second request should be vaguer and worded differently from the first ones.

Rules: at most 20 words each; only the main result or idea, not the details; write it the way it is \
remembered, in your own words and the field's everyday terms (more general words, synonyms, a different \
sentence structure), not with the vocabulary of the abstract; no author or collaboration names, no \
experiment names unless the paper is about that experiment, no arXiv numbers or years."""

STYLES = ("citation", "request")
SCHEMA = {
    "type": "object",
    "properties": {style: {"type": "array", "items": {"type": "string"}} for style in STYLES},
    "required": list(STYLES),
    "additionalProperties": False,
}


def eval_papers() -> set[str]:
    """arXiv ids of every evaluation question, development and test, including the citation batches."""
    here = ROOT / "eval"
    arxiv = {
        json.loads(line)["arxiv"]
        for f in EVAL_SETS + EVAL_FROZEN
        for line in (here / f).read_text().splitlines()
        if line.strip()
    }
    batches = here / "cache"
    for f in batches.glob("citation_contexts*.jsonl"):
        arxiv |= {json.loads(line)["arxiv"] for line in f.read_text().splitlines() if line.strip()}
    return arxiv


# ---------- sample ----------


async def sample(n: int, seed: int) -> None:
    excluded = eval_papers()
    rng = random.Random(seed)
    papers: dict[int, dict] = {}
    async with InspireClient() as client:
        for query, max_page, share in POPULATIONS:
            goal = len(papers) + round(n * share)
            while len(papers) < goal:
                data = await client._get(
                    "/literature",
                    {
                        "q": query,
                        "size": PAGE_SIZE,
                        "page": rng.randint(1, max_page),
                        "fields": "control_number,titles.title,abstracts.value,arxiv_eprints.value",
                    },
                )
                hits = [h["metadata"] for h in data["hits"]["hits"]]
                rng.shuffle(hits)
                for m in hits[:PER_PAGE]:
                    arxiv = (m.get("arxiv_eprints") or [{}])[0].get("value")
                    abstract = (m.get("abstracts") or [{}])[0].get("value", "")
                    if arxiv and arxiv not in excluded and len(abstract) >= 400 and len(papers) < goal:
                        papers.setdefault(
                            m["control_number"],
                            {
                                "recid": m["control_number"],
                                "arxiv": arxiv,
                                "title": m["titles"][0]["title"],
                                "abstract": abstract,
                            },
                        )
                print(f"{query}: {len(papers)}/{goal}", end="\r", flush=True)
            print()
    CACHE.mkdir(exist_ok=True)
    PAPERS.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in papers.values()))
    print(f"{len(papers)} training papers in {PAPERS.relative_to(ROOT)}")


# ---------- questions ----------


def prompt_key(paper: dict) -> str:
    return hashlib.sha256(f"{MODEL}|{PROMPT}|{paper['recid']}".encode()).hexdigest()[:16]


async def write_questions(client, paper: dict, sem: asyncio.Semaphore) -> dict:
    from anthropic import APIStatusError

    async with sem:
        try:
            resp = await client.messages.create(
                model=MODEL,
                max_tokens=4000,
                system=SYSTEM,
                messages=[{"role": "user", "content": PROMPT.format(title=paper["title"], abstract=paper["abstract"])}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            )
        except APIStatusError as exc:
            return {"error": f"HTTP {exc.status_code}"}
    usage = {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}
    if resp.stop_reason == "refusal":
        return {"refusal": True, "usage": usage}
    text = next(b.text for b in resp.content if b.type == "text")
    return {**json.loads(text), "usage": usage}


async def questions(limit: int | None, concurrency: int) -> None:
    from anthropic import AsyncAnthropic

    papers = [json.loads(line) for line in PAPERS.read_text().splitlines() if line.strip()][:limit]
    cache: dict[str, dict] = json.loads(QUESTIONS.read_text()) if QUESTIONS.exists() else {}
    todo = [p for p in papers if prompt_key(p) not in cache]
    client = AsyncAnthropic(api_key=load_key(), max_retries=3)
    sem = asyncio.Semaphore(concurrency)
    results = await asyncio.gather(*(write_questions(client, p, sem) for p in todo))
    for p, r in zip(todo, results, strict=True):
        if "error" not in r:
            cache[prompt_key(p)] = r
    QUESTIONS.write_text(json.dumps(cache, indent=1))

    price = PRICES[MODEL]
    spent = sum(
        r["usage"]["input"] * price["input"] + r["usage"]["output"] * price["output"] for r in results if "usage" in r
    )
    rows, dropped = [], {"abstract share": 0, "title words": 0}
    for p in papers:
        r = cache.get(prompt_key(p))
        if not r or r.get("refusal"):
            continue
        for style in STYLES:
            for q in r[style][:2]:  # the first variant that passes, so the filter does not pick the vaguest
                q = q.strip()
                share = abstract_share(q, p["abstract"])
                if share > MAX_SHARE:
                    dropped["abstract share"] += 1
                elif title_overlap(q, p["title"]):
                    dropped["title words"] += 1
                else:
                    rows.append(
                        {
                            "id": f"t{p['recid']}-{style}",
                            "recid": p["recid"],
                            "query": q,
                            "style": style,
                            "abstract_share": round(share, 3),
                        }
                    )
                    break
    DATA.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    errors = sum("error" in r for r in results)
    print(f"papers: {len(papers)}, asked now: {len(todo)}, errors: {errors}, cost of this run: ${spent / 1e6:.4f}")
    print(f"questions kept: {len(rows)}, dropped: {dropped}")
    if rows:
        shares = sorted(r["abstract_share"] for r in rows)
        print(f"abstract share of kept questions: median {shares[len(shares) // 2]:.2f}")


# ---------- pools ----------


def read_pools() -> list[dict]:
    """The pools written so far; a line cut by an interrupted run is skipped, and fetched again."""
    rows = []
    for line in POOLS.read_text().splitlines() if POOLS.exists() else []:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


async def pools(questions: Path) -> None:
    rows = [json.loads(line) for line in questions.read_text().splitlines() if line.strip()]
    done = {r["id"] for r in read_pools()}
    excluded = eval_papers()
    papers = {json.loads(line)["recid"] for line in PAPERS.read_text().splitlines() if line.strip()}
    todo = [r for r in rows if r["id"] not in done]
    async with InspireClient() as client:
        with POOLS.open("a") as out:
            for i, r in enumerate(todo, 1):
                docs, _, _ = await candidates(client, r["query"], pool=50)
                pool = [d.to_dict() for d in docs if d.arxiv_id not in excluded]
                out.write(json.dumps({"id": r["id"], "recid": r["recid"], "pool": pool}, ensure_ascii=False) + "\n")
                out.flush()
                if i % 100 == 0:
                    print(f"pools: {i}/{len(todo)}", flush=True)
    print(f"pools for {len(rows)} questions in {POOLS.relative_to(ROOT)} ({len(papers)} papers)")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="step", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=3000)
    s.add_argument("--seed", type=int, default=31)
    q = sub.add_parser("questions")
    q.add_argument("--limit", type=int)
    q.add_argument("--concurrency", type=int, default=8)
    pl = sub.add_parser("pools")
    pl.add_argument("--questions", type=Path, default=DATA, help="also train/cache/s2_judged.jsonl")
    args = parser.parse_args()
    if args.step == "sample":
        asyncio.run(sample(args.n, args.seed))
    elif args.step == "questions":
        asyncio.run(questions(args.limit, args.concurrency))
    else:
        asyncio.run(pools(args.questions))


if __name__ == "__main__":
    main()
