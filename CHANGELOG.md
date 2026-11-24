# Changelog

All notable changes. Versions follow [semantic versioning](https://semver.org).

## 0.5.1

* Runs on Python 3.10 to 3.13 (was 3.12+): `uvx` failed on machines whose default Python is older.
  CI now tests all four versions.
* The concurrency test checks ordering instead of a wall-clock threshold, so it no longer flakes on a
  loaded machine.

## 0.5.0

Broader coverage and easier to try.

* New tools: `find_similar_papers`, `get_paper_figures`, `export_citations` (BibTeX, LaTeX EU and US,
  replaces `get_bibtex`), `search_collaboration`, `get_author_profile` (h-index, citations, distribution),
  `get_author_papers` (with `max_authors` to skip large collaborations), `search_conferences`,
  `search_seminars`, `search_jobs`, `search_institutions`, `search_experiments`, `ping`: 18 tools in all.
* `get_paper` adds keywords, the arXiv PDF and open-access full texts; citing papers and references
  are paginated.
* Command line: `inspire-mcp search`, `paper`, `doctor` and `config` work without any AI client.
* HTTP: host allowlist for DNS rebinding protection, per-client rate limit with `Retry-After`,
  `X-Forwarded-For` behind a trusted proxy, `PORT` honoured.
* Upcoming events with a keyword are filtered by date on the server side: INSPIRE combines free text
  and date ranges unreliably.
* README rewritten for first-time users.
* Reranker comparison on frozen candidates (`eval/rerank_lab.py`) and a held-out set of 40 papers
  frozen before any result: no reranker beats the default MiniLM-L6 on held-out data; bge-reranker-base
  is worse and seven times slower. Honest held-out quality: right paper first 60%.
* Agentic baseline (`eval/agent_eval.py`): with semantic_search, Claude Sonnet 5.5 finds the right paper
  first 85% of the time instead of 60% with INSPIRE search alone, in 1.3 tool calls instead of 5.8;
  confirmed on the held-out set (85% against 62%; Claude Haiku 5.5 82% against 53%).

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
