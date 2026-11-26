"""Embed the harvested corpus (harvest_corpus.py) on the GPU, for the dense-retrieval experiment.

Each record is encoded as `title. abstract` (cut at 1200 characters, like the reranker's input) with a
sentence-transformers model that fastembed also ships, so a service could encode queries on CPU with the
same model. Output: train/cache/index-<model>/{vectors.npy (float16, normalized), recids.npy}.

    .venv-train/bin/python train/dense_index.py BAAI/bge-small-en-v1.5
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "train" / "cache" / "corpus.jsonl"


def index_dir(model: str) -> Path:
    return ROOT / "train" / "cache" / f"index-{model.split('/')[-1]}"


def main() -> None:
    model_name = sys.argv[1]
    recids, texts, seen = [], [], set()
    for line in CORPUS.open():
        r = json.loads(line)
        if r["recid"] in seen:
            continue
        seen.add(r["recid"])
        recids.append(r["recid"])
        texts.append(f"{r['title']}. {r['abstract']}"[:1200])
    model = SentenceTransformer(model_name, device="cuda")
    model.half()
    start = time.perf_counter()
    vectors = model.encode(
        texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True
    )
    out = index_dir(model_name)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vectors.npy", vectors.astype(np.float16))
    np.save(out / "recids.npy", np.array(recids, dtype=np.int64))
    print(f"{len(recids)} records, dim {vectors.shape[1]}, {time.perf_counter() - start:.0f}s, in {out}")


if __name__ == "__main__":
    main()
