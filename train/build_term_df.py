"""Document frequencies of words in physics abstracts, shipped with the package for candidate generation.

semantic.candidates() adds an INSPIRE query made of the question's rarest words: a long question, such
as a sentence copied from a paper, starts with generic words, and the first twelve alone often miss the
paper (eval/recall_lab.py). Rarity is measured on the ~110k abstracts of the training pools, which hold no
paper of any evaluation set. Words seen in fewer than MIN_DF abstracts are left out: unknown words count
as rare anyway.

    uv run python train/build_term_df.py
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from inspire_mcp.semantic import TERM_DF_PATH, term_key  # noqa: E402

POOLS = ROOT / "train" / "cache" / "pools.jsonl"
MIN_DF = 3


def main() -> None:
    docs: dict[int, str] = {}
    for line in POOLS.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        for p in row["pool"]:
            docs.setdefault(p["recid"], f"{p['title']} {p.get('abstract') or ''}")
    df: Counter = Counter()
    for text in docs.values():
        df.update(set(term_key(text)))
    kept = {t: n for t, n in sorted(df.items()) if n >= MIN_DF}
    TERM_DF_PATH.write_bytes(gzip.compress(json.dumps({"docs": len(docs), "df": kept}).encode(), mtime=0))
    print(f"{len(kept)} words from {len(docs)} abstracts, {TERM_DF_PATH.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
