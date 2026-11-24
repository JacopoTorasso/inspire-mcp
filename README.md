# inspire-mcp

An [MCP](https://modelcontextprotocol.io) server that lets AI assistants search
[INSPIRE-HEP](https://inspirehep.net), the literature database of high energy physics,
plus a small evaluation harness that measures how well different retrieval strategies
find the right paper for a natural-language question.

Personal project built on the public [INSPIRE REST API](https://github.com/inspirehep/rest-api-doc).
Not affiliated with CERN or INSPIRE.

## Tools

| Tool | What it does |
|---|---|
| `search_literature(query, size, sort)` | INSPIRE search syntax (`t higgs and cn atlas and date > 2020`) or plain words |
| `get_paper(identifier)` | one record by INSPIRE recid, arXiv id or DOI |
| `get_citing_papers(recid, size, sort)` | papers citing a record (`refersto:recid:`) |
| `get_references(recid, size)` | records a paper cites, most cited first |
| `search_authors(name, size)` | author profiles with INSPIRE BAI and ORCID |

All tools are read-only (`read_only_hint`), results are trimmed to the fields an assistant
needs (title, arXiv abstract, first five authors, collaboration, citations, link) to keep
token usage low, and the server instructions teach the model INSPIRE query syntax.

## Run

```bash
uv sync
uv run inspire-mcp                                   # stdio
uv run inspire-mcp --transport streamable-http       # http://127.0.0.1:8000/mcp

claude mcp add inspire -- uv --directory /path/to/inspire-mcp run inspire-mcp   # Claude Code
docker build -t inspire-mcp . && docker run -p 8000:8000 inspire-mcp
```

## Tests

```bash
uv run pytest      # offline: HTTP mocked with respx, tools exercised through an in-process MCP client
uv run ruff check .
```

## Retrieval evaluation

`eval/run_eval.py` runs a known-item retrieval benchmark: 30 natural-language questions
(`eval/queries.jsonl`), each pointing to one well-known paper identified by arXiv id.

Systems, all on the same fixed corpus (every INSPIRE record with 1500+ citations, 1396 papers):

* **bm25**: lexical baseline, Okapi BM25 on title + abstract
* **dense**: `BAAI/bge-small-en-v1.5` embeddings (384 dim, ONNX on CPU via fastembed)
* **hybrid**: reciprocal rank fusion of bm25 and dense
* **inspire_live**: the public INSPIRE search API queried with the raw question

```bash
uv run --extra eval python eval/run_eval.py
```

### Results (2026-11-24)

| system | R@1 | R@5 | R@10 | MRR@10 |
|---|---|---|---|---|
| bm25 | 0.57 | 0.90 | 0.97 | 0.71 |
| dense | 0.67 | 0.93 | 0.97 | 0.76 |
| hybrid | 0.60 | 0.97 | 0.97 | 0.76 |
| inspire_live (raw question) | 0.07 | 0.07 | 0.07 | 0.07 |

Dense cost on a laptop CPU (WSL2): index build about 280 s for 1396 abstracts, about 76 ms per query embedding.

### What this says, and what it does not

* **Raw questions do not work as INSPIRE queries**: 18 of 30 return zero hits, because the
  search engine combines words with AND. An assistant must translate intent into INSPIRE syntax
  (which the server instructions encourage) or a semantic layer is needed. The comparison with
  the other rows is not like for like: the live API searches the whole database, the others a
  1396-paper corpus.
* **Dense retrieval wins at the top of the ranking** (R@1 0.67 vs 0.57) and hybrid fusion gives
  the best R@5, but with 30 queries one query is worth 0.033, so differences between the three
  offline systems are indicative, not significant.
* **The task is easy for a small corpus of famous papers**: R@10 saturates at 0.97 for every
  offline system. The only miss (q13) asks for "version 8.2 of the general purpose Monte Carlo
  event generator" without naming PYTHIA: version numbers are a weak signal for both lexical and
  dense retrievers, and HERWIG and Sherpa win.
* Queries were written by the author, who knew the target papers: that biases them towards
  vocabulary in the abstracts.

### Next steps

* Agentic baseline: let an LLM use the MCP tools to rewrite the question into INSPIRE syntax and
  measure it with the same harness and metrics.
* Larger, harder corpus (all records of a subfield, not only top cited) and queries collected from
  real users or derived from citation contexts.
* Compare embedding models (size, domain-tuned vs general) on quality, latency and memory.
