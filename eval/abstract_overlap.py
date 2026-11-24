"""How much vocabulary each question shares with its target's abstract, against the other candidates.

Longtail and held-out questions were written from the abstract, which is also the text the reranker
reads, so shared words may favour the reranker. For each question this script takes the share of its
content words (same tokenizer as the BM25 reranker) found in the target abstract and in every other
candidate's abstract, and counts how often word overlap alone would put the target first. It then splits
the questions into overlap bands and reports, per band, how often the default cross-encoder put the target
first (from its cached scores), to see whether the reranker only works when a question copies the abstract.

Reads the candidate pools cached by run_eval.py; no network.

    uv run python eval/abstract_overlap.py
"""

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from inspire_mcp.semantic import _tokens  # noqa: E402

POOL_FILES = ("pools-famous-longtail.json", "pools-heldout.json")
XENC = "Xenova/ms-marco-MiniLM-L-6-v2"
BANDS = ((0.0, 0.6), (0.6, 0.8), (0.8, 1.01))


def share(words: set[str], abstract: str | None) -> float:
    return len(words & set(_tokens(abstract or ""))) / len(words)


def main() -> None:
    sets: dict[str, list[dict]] = {}
    for name in POOL_FILES:
        sets |= json.loads((HERE / "cache" / name).read_text())
    xenc_path = HERE / "cache" / "xenc-scores.json"
    xenc = json.loads(xenc_path.read_text()) if xenc_path.exists() else {}
    bands: dict[str, list[tuple[float, bool | None]]] = {}
    print("| set | in pool | target abstract | other candidates | first by overlap, ties lost / won |")
    print("|---|---|---|---|---|")
    for name, questions in sets.items():
        target, others, strict, loose = [], [], 0, 0
        for q in questions:
            words = set(_tokens(q["query"]))
            hit = next((p for p in q["pool"] if p["recid"] == q["recid"] and p.get("abstract")), None)
            if hit is None or not words:
                continue
            t = share(words, hit["abstract"])
            rest = [share(words, p.get("abstract")) for p in q["pool"] if p["recid"] != q["recid"]]
            target.append(t)
            others.append(statistics.fmean(rest))
            strict += all(o < t for o in rest)
            loose += all(o <= t for o in rest)
            ce = [xenc.get(f"{XENC}|{q['id']}|{p['recid']}") for p in q["pool"][:50]]
            first = None
            if q["recid"] in [p["recid"] for p in q["pool"][:50]] and None not in ce:
                best = max(zip(ce, (p["recid"] for p in q["pool"][:50])))[1]
                first = best == q["recid"]
            bands.setdefault(name, []).append((t, first))
        n = len(questions)
        print(
            f"| {name} | {len(target)}/{n} | {statistics.fmean(target):.2f} | {statistics.fmean(others):.2f} "
            f"| {strict}/{n} / {loose}/{n} |"
        )
    print("\nCross-encoder, right paper first, by share of question words found in the target abstract")
    print("(questions whose target is in the 50-candidate pool):\n")
    print("| set | " + " | ".join(f"{lo:.0%} to {min(hi, 1):.0%}" for lo, hi in BANDS) + " |")
    print("|---|" + "---|" * len(BANDS))
    for name, rows in bands.items():
        cells = []
        for lo, hi in BANDS:
            hits = [f for t, f in rows if lo <= t < hi and f is not None]
            cells.append(f"{sum(hits)}/{len(hits)}" if hits else "-")
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
