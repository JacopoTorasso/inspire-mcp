"""Does a dense index bring targets into the candidate pool that INSPIRE's keyword search misses?

For every development question (raw questions of citedev, famous and longtail, and the agent-written
queries of agentdev), the frozen INSPIRE pool of the current server is compared with the nearest neighbours
of the question in the dense index (dense_index.py), and with the two merged: the pool the reranker would
then see. Development sets only.

    .venv-train/bin/python train/dense_recall.py BAAI/bge-small-en-v1.5
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from dense_index import index_dir  # noqa: E402

POOLS = ("pools-citedev-v2.json", "pools-famous-longtail-v2.json", "pools-agentdev.json")
QUERY_PREFIX = {"bge": "Represent this sentence for searching relevant passages: "}


def interleave(*lists: list[int]) -> list[int]:
    out: dict[int, None] = {}
    for i in range(max(map(len, lists))):
        for x in lists:
            if i < len(x):
                out.setdefault(x[i], None)
    return list(out)


def main() -> None:
    model_name = sys.argv[1]
    d = index_dir(model_name)
    arrays = np.load(d / "index.npz")
    vectors, recids = torch.from_numpy(arrays["vectors"]).cuda(), arrays["recids"]
    in_index = set(recids.tolist())
    model = SentenceTransformer(model_name, device="cuda")
    prefix = next((p for k, p in QUERY_PREFIX.items() if k in model_name.lower()), "")
    print(f"{model_name}: {len(recids)} records")
    for file in POOLS:
        for name, queries in json.loads((ROOT / "eval" / "cache" / file).read_text()).items():
            q_vec = model.encode(
                [prefix + q["query"] for q in queries], normalize_embeddings=True, convert_to_tensor=True
            )
            top = torch.topk(q_vec.half() @ vectors.T, 100, dim=1).indices.cpu().numpy()
            rows = []
            for q, idx in zip(queries, top, strict=True):
                dense = recids[idx].tolist()
                pool = [p["recid"] for p in q["pool"]]
                gold = q["recid"]
                rows.append(
                    {
                        "indexed": gold in in_index,
                        "inspire@50": gold in pool[:50],
                        "dense@50": gold in dense[:50],
                        "dense@100": gold in dense,
                        "25+25": gold in interleave(pool[:25], dense[:25])[:50],
                        "50+50": gold in set(pool[:50]) | set(dense[:50]),
                    }
                )
            n = len(rows)
            cols = " ".join(f"{k}={sum(r[k] for r in rows) / n:.2f}" for k in rows[0])
            print(f"  {name:<16} n={n:<3} {cols}")


if __name__ == "__main__":
    main()
