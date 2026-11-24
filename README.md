# inspire-mcp

[![ci](https://github.com/JacopoTorasso/inspire-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/JacopoTorasso/inspire-mcp/actions/workflows/ci.yml)
[![eval](https://github.com/JacopoTorasso/inspire-mcp/actions/workflows/eval.yml/badge.svg)](https://github.com/JacopoTorasso/inspire-mcp/actions/workflows/eval.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Ask your AI assistant about high energy physics literature, in plain language.**

inspire-mcp connects assistants such as Claude to [INSPIRE-HEP](https://inspirehep.net), the database
of high energy physics papers, authors, conferences and jobs. Describe the paper you are looking for
in your own words: the server finds it and puts the right one first 70% of the time, against 7% for
the same question typed into INSPIRE's search box ([how this is measured](docs/evaluation.md)).

## What you can ask

* *"Find the paper that introduced GANs for calorimeter shower simulation, and who cites it lately"*
* *"What are Michela Paganini's most cited papers outside large collaborations? What is her h-index?"*
* *"Which neutrino conferences are coming up, and where?"*
* *"Are there open postdoc positions in lattice QCD in Europe?"*
* *"Show me the figures of arXiv:1705.02355 and give me its BibTeX"*
* *"Papers similar to the Randall–Sundrum paper"*
* *"Recent LHCb papers on pentaquarks"*

Every answer comes with INSPIRE links, so you can check the sources.

## Try it in one minute

You need [uv](https://docs.astral.sh/uv/), a small Python tool runner:

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then search straight from the terminal, no AI client needed:

```console
$ uvx --from "inspire-mcp[semantic]" inspire-mcp search "neural networks to speed up calorimeter shower simulation"

 1. Quantum Invertible Neural Networks for Fast Calorimetry as Quantum Generative Model with high
    interpretability at Future Colliders
     Corchia, Federico 2026  ·  0 citations  ·  https://inspirehep.net/literature/3221332
 2. CaloShowerGAN, a generative adversarial network model for fast calorimeter shower simulation
     Faucci Giannelli, Michele 2024  ·  33 citations  ·  https://inspirehep.net/literature/2697185
 3. On the use of neural networks for energy reconstruction in high-granularity calorimeters
     Akchurin, N. 2021  ·  38 citations  ·  https://inspirehep.net/literature/1889423
 ...
ranked by cross-encoder:ms-marco-MiniLM-L-6-v2 from 50 INSPIRE candidates in 2.7s
```

The first run downloads the package and a small ranking model (about 100 MB); later runs start in
seconds. Other commands: `inspire-mcp paper 1705.02355` shows one paper, `inspire-mcp doctor` checks
your setup.

## Use it with your AI assistant

**Claude Code**, one command:

```bash
claude mcp add inspire -- uvx --from "inspire-mcp[semantic]" inspire-mcp
```

**Claude Desktop**: open *Settings → Developer → Edit Config* (the file is
`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS and
`%APPDATA%\Claude\claude_desktop_config.json` on Windows), add the server and restart Claude:

```json
{
  "mcpServers": {
    "inspire": {
      "command": "uvx",
      "args": ["--from", "inspire-mcp[semantic]", "inspire-mcp"]
    }
  }
}
```

**Cursor, VS Code, and other MCP clients** use the same `command` and `args`. Run
`inspire-mcp config` to print the snippets.

> Until the first release is on PyPI, replace `inspire-mcp[semantic]` with
> `inspire-mcp[semantic] @ git+https://github.com/JacopoTorasso/inspire-mcp`.

## Tools

The assistant picks the right tool by itself; this is what it has available.

| | Tool | What it does |
|---|---|---|
| **Papers** | `semantic_search` | finds papers from a description in plain language |
| | `search_literature` | precise search in INSPIRE syntax (`t higgs and cn atlas and date > 2020`) |
| | `get_paper` | full record: abstract, authors, journal, keywords, arXiv PDF, open-access full text |
| | `find_similar_papers` | papers on the same subject as a given one |
| | `get_citing_papers` | who cites a paper, newest or most cited first |
| | `get_references` | what a paper cites |
| | `get_paper_figures` | figures with captions and download links |
| | `export_citations` | BibTeX or LaTeX (`\bibitem`, EU or US style) entries |
| | `search_collaboration` | papers of ATLAS, CMS, LHCb, ALICE, Belle II... |
| **People** | `search_authors` | find an author profile |
| | `get_author_profile` | papers, citations, h-index, citation distribution |
| | `get_author_papers` | an author's papers, optionally only those written in small teams |
| **Community** | `search_conferences` | upcoming (or past) conferences and workshops |
| | `search_seminars` | upcoming seminars and online talks, with link to join |
| | `search_jobs` | open positions: postdocs, faculty, PhD, staff |
| | `search_institutions` | research institutions, location and output |
| | `search_experiments` | experiments: accelerator, dates, number of papers |
| | `ping` | checks the server is up |

Lists are paginated and can return compact results to save the assistant's context. Also included:
a resource, `inspire://literature/{recid}`, and a prompt, `literature_review`, that writes a short cited
overview of a topic.

## How good is the search?

Measured on 40 papers picked at random from INSPIRE, each described by a question that does not
repeat its title:

| Approach | Right paper first | In the top 10 |
|---|---|---|
| Question typed into INSPIRE's search | 7% | 10% |
| Keywords of the question, INSPIRE's ranking | 38% | 72% |
| **inspire-mcp** (keywords, then a cross-encoder re-reads every abstract) | **70%** | **85%** |

A second benchmark of 60 famous papers, the significance tests and the design choices that were
tried and rejected are in [docs/evaluation.md](docs/evaluation.md).

## How it works

```mermaid
flowchart LR
    Q[your question] --> K["keywords joined with OR<br/>(INSPIRE needs all words to match otherwise)"]
    K --> P[(50 candidate papers<br/>from live INSPIRE)]
    P --> R["cross-encoder reads the question<br/>with each title and abstract"]
    R --> A[best matches, with links<br/>and how they were found]
```

Nothing is copied or indexed: every search runs against the live INSPIRE database, and only the
ranking happens locally, on CPU. For questions about foundational work the assistant can ask for
highly cited candidates (`prefer_influential`), which helps find famous papers buried under their
follow-ups.

<details>
<summary><b>Run it as a service</b> (Docker, Kubernetes, configuration)</summary>

```bash
docker build --build-arg EXTRAS=semantic -t inspire-mcp:semantic .   # models included, no download at runtime
docker run -p 8000:8000 -e INSPIRE_MCP_ALLOWED_HOSTS=mcp.example.org inspire-mcp:semantic
kubectl apply -f deploy/k8s/deployment.yaml                           # non-root, read-only filesystem, probes
```

Release tags publish `ghcr.io/jacopotorasso/inspire-mcp:<version>` (no models, 364 MB) and
`:<version>-semantic` (about 1 GB).

| Endpoint | Purpose |
|---|---|
| `/mcp` | MCP over Streamable HTTP, rate limited per client |
| `/healthz` | liveness |
| `/readyz` | readiness: 503 until the ranking model is loaded |
| `/metrics` | Prometheus: calls by tool and outcome, latency histogram, readiness |

Every option is a flag or an environment variable:

| Variable | Default | |
|---|---|---|
| `INSPIRE_MCP_TRANSPORT` | `stdio` | `streamable-http` to serve over HTTP |
| `INSPIRE_MCP_HOST`, `INSPIRE_MCP_PORT` | `127.0.0.1`, `8000` | also honours `PORT` |
| `INSPIRE_MCP_ALLOWED_HOSTS` | | public host names (DNS rebinding protection when binding to `0.0.0.0`) |
| `INSPIRE_MCP_RATE_LIMIT` | `120` | MCP requests per minute per client, `0` disables |
| `INSPIRE_MCP_TRUST_PROXY` | `false` | identify clients by `X-Forwarded-For` behind a trusted proxy |
| `INSPIRE_MCP_RERANKER` | `auto` | `cross-encoder`, `dense` or `bm25` |
| `INSPIRE_MCP_CACHE_TTL` | `300` | seconds INSPIRE answers stay cached in memory |
| `INSPIRE_MCP_MIN_INTERVAL` | `0.35` | seconds between requests to INSPIRE |
| `INSPIRE_MCP_CACHE` | `~/.cache/inspire-mcp` | embedding cache |
| `INSPIRE_MCP_LOG_FORMAT`, `INSPIRE_MCP_LOG_LEVEL` | `text`, `INFO` | `json` for log aggregation |

The HTTP endpoint has no authentication; see [SECURITY.md](SECURITY.md).

</details>

<details>
<summary><b>Troubleshooting</b></summary>

* **`uvx: command not found`**: install uv (above) and open a new terminal.
* **The first search is slow**: the ranking model is downloaded once (about 100 MB).
* **The assistant does not see the tools**: restart the client after editing its configuration, then
  run `inspire-mcp doctor` in a terminal.
* **Behind a corporate proxy**: set `HTTPS_PROXY`; the server needs to reach `https://inspirehep.net`.
* **Lighter install**: `uvx inspire-mcp` (without `[semantic]`) skips the models and ranks with BM25,
  which is noticeably less accurate.

</details>

## Development

```bash
uv sync --extra semantic
uv run pytest                                  # 68 tests, offline, about 90% coverage
uv run ruff check . && uv run mypy
uv run --extra eval python eval/run_eval.py    # benchmark against live INSPIRE
```

CI runs lint, types, tests, both container images with a health probe and Kubernetes manifest
validation; the benchmark runs weekly to catch regressions. See [CONTRIBUTING.md](CONTRIBUTING.md)
and [CHANGELOG.md](CHANGELOG.md).

## License

MIT. Data comes from [INSPIRE-HEP](https://inspirehep.net) under its terms of use. This is an
independent project, not affiliated with CERN or INSPIRE.
