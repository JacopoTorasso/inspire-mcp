# Changelog

All notable changes. Versions follow [semantic versioning](https://semver.org).

## 0.4.0

Production readiness.

* Reranking runs in a worker thread: a semantic search no longer blocks other requests or health probes.
* Models load in the background at startup over HTTP; `/readyz` answers 503 until the reranker is ready.
* Prometheus metrics on `/metrics`: calls by tool and outcome, latency histogram, readiness.
* Every option configurable through `INSPIRE_MCP_*` environment variables; JSON logs.
* `search_literature` pagination (`page`, `has_more`, `next_page`) and `compact` results.
* Server instructions tell the model to treat retrieved text as data, not instructions.
* Fixed dependency floor: the server requires `mcp>=2.3` (it declared `>=1.2`), and `starlette`.
* Type checked with mypy in CI; release workflow publishes to PyPI and GHCR on tags.

## 0.3.0

* Long-tail benchmark: 40 randomly sampled papers with 20-200 citations, next to the 60 famous ones.
* Cross-encoder reranking becomes the default (MRR 0.49 to 0.76 on typical papers).
* Multi-query candidate pool evaluated and rejected.

## 0.2.0

* `semantic_search`: two-stage retrieval over the live INSPIRE index, `get_bibtex`, paper resource,
  `literature_review` prompt, response cache, Kubernetes manifests, weekly evaluation workflow.

## 0.1.0

* First release: literature, citation and author tools over the INSPIRE REST API.
