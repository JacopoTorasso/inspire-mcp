"""Freeze the citation-context test set: one judged sentence per target paper, chosen by a fixed rule.

Rule, decided before any model was run on these questions: among the sentences the judge kept for a
target, the one with the lowest id (c<recid>-<i>, i = order of the citing paper in citation_contexts.py).

Only ids are versioned: eval/queries_citations.jsonl holds the target, the citing paper and a hash of the
sentence, never the sentence itself (it comes from an arXiv paper under its own licence and stays in
eval/cache/). run_eval.load_queries() puts the text back from the cache and checks the hash, so anyone
can rebuild the set with citation_contexts.py and judge_contexts.py and verify it is the same.

    uv run python eval/freeze_citations.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
JUDGED = HERE / "cache" / "citation_contexts_judged.jsonl"
OUT = HERE / "queries_citations.jsonl"
TEXTS = HERE / "cache" / "queries_citations.texts.jsonl"  # the sentences, read by run_eval.load_queries()


def sentence_hash(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()[:16]


def order(row: dict) -> tuple[int, int]:
    recid, i = row["id"][1:].split("-")
    return int(recid), int(i)


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT.name} exists: the set is frozen, delete it by hand only to rebuild on purpose")
    rows = sorted((json.loads(line) for line in JUDGED.read_text().splitlines() if line.strip()), key=order)
    chosen: dict[int, dict] = {}
    for r in rows:
        chosen.setdefault(r["recid"], r)
    frozen = [
        {"id": r["id"], "arxiv": r["arxiv"], "citing_arxiv": r["citing_arxiv"], "sha": sentence_hash(r["query"])}
        for r in chosen.values()
    ]
    OUT.write_text("".join(json.dumps(r) + "\n" for r in frozen))
    TEXTS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chosen.values()))
    print(f"{len(frozen)} questions from {len(rows)} judged sentences, written to {OUT.name}")


if __name__ == "__main__":
    main()
