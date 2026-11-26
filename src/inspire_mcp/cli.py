"""Command line: try the search from a terminal, check the setup, print client configuration.

    inspire-mcp search "neural networks for calorimeter simulation"
    inspire-mcp paper 1705.02355
    inspire-mcp doctor
    inspire-mcp config

Without a subcommand (or with `serve`), inspire-mcp runs the MCP server, which is what AI clients call.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import os
import shutil
import sys
import textwrap
import time
from pathlib import Path

from . import __version__
from .client import InspireClient, InspireError

COMMANDS = ("search", "paper", "doctor", "config", "index")
MIN_PYTHON = (3, 10)  # keep in sync with requires-python in pyproject.toml (checked by a test)
WIDTH = min(shutil.get_terminal_size((100, 20)).columns, 110)


def _wrap(text: str, indent: str = "     ") -> str:
    return textwrap.fill(text, width=WIDTH, initial_indent=indent, subsequent_indent=indent)


def _semantic_installed() -> bool:
    return importlib.util.find_spec("fastembed") is not None


async def _search(question: str, size: int, as_json: bool) -> int:
    from .semantic import default_reranker, semantic_search

    if not _semantic_installed():
        print(
            "note: models not installed, ranking with BM25. For better results:\n"
            "      uvx --from 'inspire-mcp[semantic]' inspire-mcp search ...\n",
            file=sys.stderr,
        )
    else:
        print("loading the ranking model (a few seconds the first time)...", file=sys.stderr)
    reranker = await asyncio.to_thread(default_reranker)
    async with InspireClient() as client:
        start = time.perf_counter()
        result = await semantic_search(client, question, k=size, reranker=reranker)
        elapsed = time.perf_counter() - start
    if as_json:
        print(json.dumps([p.to_dict() for p in result.papers], indent=2, ensure_ascii=False))
        return 0
    if not result.papers:
        print("No papers found. Try fewer or different words.")
        return 1
    print(f"\n{len(result.papers)} papers for: {question}\n")
    for i, p in enumerate(result.papers, 1):
        who = p.collaborations[0] if p.collaborations else (p.authors[0] if p.authors else "")
        print(f"{i:>2}. {p.title}")
        print(f"     {who} {p.year or ''}  ·  {p.citation_count or 0} citations  ·  {p.url}")
    print(f"\nranked by {result.reranker} from {result.pool_size} INSPIRE candidates in {elapsed:.1f}s")
    return 0


async def _paper(identifier: str, as_json: bool) -> int:
    async with InspireClient() as client:
        paper = await client.get_paper(identifier)
        if paper is None:
            print(f"No INSPIRE record for {identifier!r}")
            return 1
        files = await client.documents(str(paper.recid))
    if as_json:
        print(json.dumps({**paper.to_dict(), "fulltext": files}, indent=2, ensure_ascii=False))
        return 0
    authors = ", ".join(paper.authors) + (
        f" and {paper.author_count - len(paper.authors)} more"
        if paper.author_count and paper.author_count > len(paper.authors)
        else ""
    )
    print(f"\n{paper.title}\n")
    print(_wrap(authors, ""))
    print(f"{paper.journal or 'preprint'} ({paper.year})  ·  {paper.citation_count or 0} citations")
    if paper.abstract:
        print("\n" + _wrap(paper.abstract, ""))
    print(f"\nINSPIRE  {paper.url}")
    if paper.pdf_url:
        print(f"arXiv    {paper.pdf_url}")
    for f in files:
        print(f"full text {f['url']} ({f.get('description') or f.get('source')})")
    return 0


async def _doctor() -> int:
    ok = True

    def line(passed: bool, label: str, detail: str = "", fix: str = "") -> None:
        nonlocal ok
        ok &= passed
        print(f"{'✓' if passed else '✗'} {label}" + (f": {detail}" if detail else ""))
        if fix and not passed:
            print(f"    fix: {fix}")

    print(f"inspire-mcp {__version__}\n")
    minimum = ".".join(map(str, MIN_PYTHON))
    line(
        sys.version_info >= MIN_PYTHON,
        "Python",
        sys.version.split()[0],
        f"install Python {minimum} or newer, or use uvx",
    )
    try:
        async with InspireClient(cache_ttl=0) as client:
            start = time.perf_counter()
            total, _ = await client.search_literature("t higgs", size=1)
            line(total > 0, "INSPIRE API reachable", f"{(time.perf_counter() - start) * 1000:.0f} ms")
    except InspireError as exc:
        line(
            False,
            "INSPIRE API reachable",
            str(exc),
            "check the network or proxy; the server needs https://inspirehep.net",
        )
    semantic = _semantic_installed()
    line(
        True,
        "Ranking",
        "cross-encoder (semantic extra installed)"
        if semantic
        else "BM25 only; install the semantic extra for much better ranking: inspire-mcp[semantic]",
    )
    if semantic:
        models = Path(
            os.environ.get("FASTEMBED_CACHE_PATH", Path(os.environ.get("TMPDIR", "/tmp")) / "fastembed_cache")
        )
        cached = models.exists() and any(models.iterdir())
        line(True, "Models", f"cached in {models}" if cached else "downloaded on first use (about 100 MB)")
    cache = Path(os.environ.get("INSPIRE_MCP_CACHE", Path.home() / ".cache" / "inspire-mcp"))
    try:
        cache.mkdir(parents=True, exist_ok=True)
        probe = cache / ".write-test"
        probe.write_text("ok")
        probe.unlink()
        line(True, "Cache directory writable", str(cache))
    except OSError as exc:
        line(False, "Cache directory writable", f"{cache}: {exc}", "set INSPIRE_MCP_CACHE to a writable directory")
    print("\nAll good: add the server to your AI client with `inspire-mcp config`." if ok else "\nSome checks failed.")
    return 0 if ok else 1


def _config() -> int:
    server = {"command": "uvx", "args": ["--from", "inspire-mcp[semantic]", "inspire-mcp"]}
    print("Claude Code (run in a terminal):\n")
    print("  claude mcp add inspire -- uvx --from 'inspire-mcp[semantic]' inspire-mcp\n")
    print("Claude Desktop, Cursor, VS Code and other clients (add to the client's MCP configuration):\n")
    print(textwrap.indent(json.dumps({"mcpServers": {"inspire": server}}, indent=2), "  "))
    print("\nNeeds uv (https://docs.astral.sh/uv/). Restart the client after editing its configuration.")
    return 0


async def _index(action: str, directory: Path, model: str | None) -> int:
    from .dense import DEFAULT_MODEL, build

    if action == "update" and not (directory / "index.json").exists():
        print(f"{directory} holds no index: run 'inspire-mcp index build {directory}' first", file=sys.stderr)
        return 1
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    added = await build(directory, model or DEFAULT_MODEL, update=action == "update")
    print(f"{added} records embedded into {directory}")
    return 0


def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="inspire-mcp", description="INSPIRE-HEP search for AI assistants")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("search", help="find papers for a question")
    s.add_argument("question", nargs="+")
    s.add_argument("-n", "--size", type=int, default=10)
    s.add_argument("--json", action="store_true")
    p = sub.add_parser("paper", help="show one paper (INSPIRE id, arXiv id or DOI)")
    p.add_argument("identifier")
    p.add_argument("--json", action="store_true")
    sub.add_parser("doctor", help="check the setup")
    sub.add_parser("config", help="print configuration for AI clients")
    ix = sub.add_parser("index", help="build or update a local semantic index for a service (--dense-index)")
    ix.add_argument("action", choices=["build", "update"], help="build: every record; update: the newest ones")
    ix.add_argument("directory", type=Path)
    ix.add_argument("--model", default=None, help="fastembed text embedding model (build only)")
    args = parser.parse_args(argv)
    if args.command == "search":
        return asyncio.run(_search(" ".join(args.question), max(1, min(args.size, 25)), args.json))
    if args.command == "paper":
        return asyncio.run(_paper(args.identifier, args.json))
    if args.command == "doctor":
        return asyncio.run(_doctor())
    if args.command == "index":
        return asyncio.run(_index(args.action, args.directory, args.model))
    return _config()
