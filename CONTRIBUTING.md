# Contributing

```bash
uv sync --extra semantic
uv run pytest            # offline, no network needed
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

Changes to retrieval (candidate generation, reranking, fusion, defaults) should come with a run of
`uv run --extra eval python eval/run_eval.py` and the before and after numbers in the pull request:
the defaults in this project are chosen from those measurements, see `eval/README.md`.
