"""Prepare a dense index (dense_index.py on the GPU, or `inspire-mcp index build`) for a Hugging Face dataset.

The output directory holds index.npz, index.json with the file's sha256 (checked by `inspire-mcp index
download`) and a dataset card. Upload it with your own token, then point inspire_mcp.dense.DEFAULT_REPO at
the repository if it differs:

    uv run --extra semantic python train/export_index_hf.py train/cache/index-bge-small-en-v1.5 train/cache/hf-index
    hf upload <user>/inspire-hep-dense-index train/cache/hf-index --repo-type dataset
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from inspire_mcp.dense import sha256  # noqa: E402

CARD = """---
license: cc0-1.0
pretty_name: INSPIRE-HEP dense index ({model_short})
tags: [inspire-hep, high-energy-physics, embeddings, retrieval]
size_categories: [100K<n<1M]
---

# INSPIRE-HEP dense index ({model_short})

Normalized `{model}` embeddings of the {records:,} INSPIRE-HEP literature records whose abstract comes
from arXiv, with their INSPIRE record ids, for the optional semantic index of
[inspire-mcp](https://github.com/JacopoTorasso/inspire-mcp): its semantic_search then takes half of its
candidates from the records nearest to a question by meaning.

* `index.npz`: `vectors` (float16, {records:,} x {dim}) and `recids` (int64), same order;
* `index.json`: model, query prefix, record count, newest earliest-date covered ({newest}), sha256 of
  `index.npz`.

No text is included: each vector embeds `title. abstract` (cut at 1200 characters) of an arXiv abstract,
and arXiv metadata is CC0; the records themselves are fetched live from INSPIRE. Queries must be
embedded with the same model and the prefix in `index.json`.

```bash
inspire-mcp index download DIR      # checks the sha256, then
inspire-mcp index update DIR        # adds the records published since
INSPIRE_MCP_DENSE_INDEX=DIR inspire-mcp --transport streamable-http
```
"""


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    import numpy as np

    meta = json.loads((src / "index.json").read_text())
    with np.load(src / "index.npz") as arrays:
        records, dim = arrays["vectors"].shape
        assert len(arrays["recids"]) == records, "vectors and record ids differ in length"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / "index.npz", out / "index.npz")
    meta |= {"records": int(records), "sha256": sha256(out / "index.npz")}
    (out / "index.json").write_text(json.dumps(meta, indent=1))
    model = meta["model"]
    (out / "README.md").write_text(
        CARD.format(model=model, model_short=model.split("/")[-1], records=records, dim=dim, newest=meta["newest"])
    )
    size = (out / "index.npz").stat().st_size / 1e6
    print(f"{out}: {records} records x {dim}, {size:.0f} MB, sha256 {meta['sha256'][:12]}...")


if __name__ == "__main__":
    main()
