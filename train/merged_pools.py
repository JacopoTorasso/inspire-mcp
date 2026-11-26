"""Candidate pools with a dense index: half from the server's INSPIRE pool, half from the index, interleaved.

Writes eval/cache/pools-<set>-dense-<model>.json in the frozen-pool format, so that finetune.py --eval-only
--pools-file (GPU) or rerank_lab.py can measure what the reranker makes of the larger recall. Records from
the index carry the title and arXiv abstract stored in the corpus. Development sets only.

    .venv-train/bin/python train/merged_pools.py BAAI/bge-small-en-v1.5
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
from dense_index import CORPUS, index_dir  # noqa: E402
from dense_recall import POOLS, QUERY_PREFIX  # noqa: E402

HALF = 25


def main() -> None:
    model_name = sys.argv[1]
    d = index_dir(model_name)
    arrays = np.load(d / "index.npz")
    vectors, recids = torch.from_numpy(arrays["vectors"]).cuda(), arrays["recids"]
    corpus = {}
    for line in CORPUS.open():
        r = json.loads(line)
        corpus[r["recid"]] = {"recid": r["recid"], "title": r["title"], "abstract": r["abstract"]}
    model = SentenceTransformer(model_name, device="cuda")
    prefix = next((p for k, p in QUERY_PREFIX.items() if k in model_name.lower()), "")
    for file in POOLS:
        out = {}
        for name, queries in json.loads((ROOT / "eval" / "cache" / file).read_text()).items():
            q_vec = model.encode(
                [prefix + q["query"] for q in queries], normalize_embeddings=True, convert_to_tensor=True
            )
            top = torch.topk(q_vec.half() @ vectors.T, HALF * 2, dim=1).indices.cpu().numpy()
            merged_queries = []
            for q, idx in zip(queries, top, strict=True):
                inspire = q["pool"][:HALF]
                dense = [corpus[int(r)] for r in recids[idx]]
                seen, pool = set(), []
                for i in range(HALF * 2):  # interleave, then fill from the index when both lists overlap
                    for src in (inspire, dense):
                        if i < len(src) and src[i]["recid"] not in seen and len(pool) < HALF * 2:
                            seen.add(src[i]["recid"])
                            pool.append(src[i])
                merged_queries.append({**q, "pool": pool})
            out[name] = merged_queries
        tag = file.removesuffix(".json").removesuffix("-v2")
        path = ROOT / "eval" / "cache" / f"{tag}-dense-{model_name.split('/')[-1]}.json"
        path.write_text(json.dumps(out))
        print(f"{path.name}: " + ", ".join(f"{n} {len(qs)}" for n, qs in out.items()))


if __name__ == "__main__":
    main()
