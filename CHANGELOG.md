# Changelog

All notable changes. Versions follow [semantic versioning](https://semver.org).

## 0.6.0

* Experimental local semantic index for services, off by default (more targets among the candidates on
  the development sets, no gain on a fresh test set: docs/evaluation.md): `inspire-mcp index download DIR` installs a prebuilt one
  (sha256-checked), `index build` / `index update` embed the INSPIRE records with an arXiv abstract
  (bge-small, fastembed; vectors and record ids only), and with
  `--dense-index DIR` (`INSPIRE_MCP_DENSE_INDEX`) half of semantic_search's candidates come from the
  records nearest to the question by meaning. The server reloads the index after an update. Kubernetes
  Job, CronJob and volume in `deploy/k8s/dense-index.yaml`.

* The cross-encoder scores only the 25 candidates BM25 ranks highest: about half the reranking time,
  with no loss on any development or test set (not with `prefer_influential`, whose candidates are
  there because lexical relevance buries them).
* `--reranker-model` (`INSPIRE_MCP_RERANKER_MODEL`): any fastembed cross-encoder, a Hugging Face
  repository with an ONNX export, or a local directory with one.
* INSPIRE's rate limit is kept as a window of 14 requests per 5 s instead of a fixed spacing, and the
  strict and relaxed candidate queries run together: candidates in 0.4 to 0.65 s instead of about 0.75 s.
* Evaluation: questions taken from the sentences in which other papers cite the target, a frozen test
  set of 86 and a development set of 64 (`eval/citation_contexts.py`, `judge_contexts.py`,
  `freeze_citations.py`); candidate recall (`recall_lab.py`); Claude Haiku as a reference reranker
  (`llm_rerank.py`). Training code for a physics cross-encoder in `train/`: the fine-tuned models
  beat the default on the development sets but not on the test sets, so the default is unchanged
  (docs/evaluation.md).
* Tool output costs an assistant about a quarter less per question (Claude Haiku agent, development
  questions: -24.6%, no measurable loss of accuracy): semantic_search returns the first 3 results in
  full and later ones as light records (an abstract excerpt of up to 400 characters, first author and
  author count), without keywords, categories or pdf_url (get_paper has them); results are compact JSON
  instead of indented;
  tool schemas no longer repeat each argument name as a title. The criterion fixed beforehand asked for
  -25%; the change is adopted by decision, as docs/evaluation.md records.
* Evaluation: `agent_eval.py --mcp` runs the agent on the real server in process, and
  `compare_runs.py` compares groups of runs.

## 0.5.3

* Network errors are retried with exponential backoff (1 s, 2 s, 4 s, at most 10 s) instead of
  immediately; the wait happens outside the request lock, so other requests are not held up.
* The User-Agent sent to INSPIRE carries the real version and this project's URL, so INSPIRE's
  administrators can tell who generates the traffic.
* Evaluation: `eval/abstract_overlap.py` measures how much vocabulary the questions share with the
  target abstracts; the bias it shows is documented under Limits, and the README now opens with the
  assistant results and the keyword baseline.

## 0.5.2

* `inspire-mcp doctor` reported Python 3.10 and 3.11 as unsupported; the minimum now comes from one
  constant, and a test keeps it equal to `requires-python`.

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
