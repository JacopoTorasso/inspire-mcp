"""MCP server exposing INSPIRE-HEP literature and author search as tools."""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from . import __version__
from .client import InspireClient, InspireError
from .semantic import (
    BM25Reranker,
    CrossEncoderReranker,
    DenseReranker,
    Reranker,
    default_reranker,
    semantic_search,
)

log = logging.getLogger("inspire_mcp")

INSTRUCTIONS = """\
Tools over INSPIRE-HEP, the literature database of high energy physics.
For natural-language questions ("the paper that proposed ...") use semantic_search.
For precise requests use search_literature with INSPIRE syntax:
  t <words>        title words          a <name> / a E.Witten.1   author (name or INSPIRE BAI)
  cn <name>        collaboration        j <journal>               journal
  date > 2020      date filter          topcite 500+              citation count
  arxiv:<id>       arXiv eprint         refersto:recid:<id>       papers citing a record
Plain words in search_literature are combined with AND, so long sentences often return nothing.
Always cite papers with their INSPIRE url so the user can verify them, and never invent references.
"""

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=True)
SortArg = Literal["mostrecent", "mostcited"] | None


def _instrumented[**P, R](func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    """Log latency and outcome of every tool call; turn API failures into MCP tool errors."""

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        outcome = "ok"
        try:
            return await func(*args, **kwargs)
        except InspireError as exc:
            outcome = "error"
            raise ToolError(str(exc)) from exc
        finally:
            log.info("tool=%s outcome=%s latency_ms=%.0f", func.__name__, outcome, (time.perf_counter() - start) * 1000)

    return wrapper


def build_server(client: InspireClient | None = None, reranker: Reranker | None = None) -> MCPServer:
    inspire = client or InspireClient()
    state: dict[str, Reranker] = {}

    def get_reranker() -> Reranker:
        # Loading the embedding model takes seconds, so do it on first use, not at startup.
        if "r" not in state:
            state["r"] = reranker or default_reranker()
        return state["r"]

    server = MCPServer("inspire-hep", instructions=INSTRUCTIONS, version=__version__)

    @server.tool(annotations=READ_ONLY)
    @_instrumented
    async def search_literature(query: str, size: int = 10, sort: SortArg = None) -> dict[str, Any]:
        """Search INSPIRE-HEP papers with INSPIRE search syntax.

        Args:
            query: INSPIRE query, e.g. "t higgs and cn atlas and date > 2020". Plain words are ANDed.
            size: number of results to return (1-25).
            sort: "mostrecent", "mostcited", or null for the INSPIRE default ranking.
        """
        total, papers = await inspire.search_literature(query, size=_clamp(size, 1, 25), sort=sort)
        return {"total": total, "results": [p.to_dict() for p in papers]}

    @server.tool(annotations=READ_ONLY)
    @_instrumented
    async def semantic_search(
        question: str, size: int = 10, pool: int = 50, prefer_influential: bool = False
    ) -> dict[str, Any]:
        """Find papers for a natural-language question, e.g. "the paper that proposed warped extra dimensions".

        Retrieves up to `pool` candidates from INSPIRE with a relaxed query, reranks them by semantic
        similarity to the question and fuses that order with INSPIRE's ranking.

        Args:
            question: the question in plain English.
            size: number of results to return (1-25).
            pool: candidates to rerank (10-100). Larger finds more, but is slower the first time.
            prefer_influential: true when the user asks for a foundational, seminal or well-known paper:
                half of the candidates then come from papers with 250+ citations. Leave false for recent
                or niche work, which this would push out.
        """
        result = await _semantic(inspire, question, size, pool, get_reranker(), prefer_influential)
        return {
            "results": [{**p.to_dict(), "similarity": result.scores.get(p.recid)} for p in result.papers],
            "retrieval": {
                "inspire_query": result.inspire_query,
                "strict_query_hits": result.strict_hits,
                "candidates_reranked": result.pool_size,
                "reranker": result.reranker,
                "fusion": result.fusion,
            },
        }

    @server.tool(annotations=READ_ONLY)
    @_instrumented
    async def get_paper(identifier: str) -> dict[str, Any]:
        """Get one paper by INSPIRE record id, arXiv id (e.g. 1207.7214, hep-th/9711200) or DOI."""
        paper = await inspire.get_paper(identifier)
        if paper is None:
            raise ToolError(f"No INSPIRE record found for {identifier!r}")
        return paper.to_dict()

    @server.tool(annotations=READ_ONLY)
    @_instrumented
    async def get_citing_papers(recid: int, size: int = 10, sort: SortArg = "mostrecent") -> dict[str, Any]:
        """List papers that cite the given INSPIRE record (use get_paper first to find the recid)."""
        total, papers = await inspire.citing_papers(recid, size=_clamp(size, 1, 25), sort=sort)
        return {"total_citations": total, "results": [p.to_dict() for p in papers]}

    @server.tool(annotations=READ_ONLY)
    @_instrumented
    async def get_references(recid: int, size: int = 25) -> dict[str, Any]:
        """List the INSPIRE records referenced by the given paper, most cited first."""
        total, papers = await inspire.references(recid, size=_clamp(size, 1, 50))
        return {"total_references": total, "results": [p.to_dict() for p in papers]}

    @server.tool(annotations=READ_ONLY)
    @_instrumented
    async def get_bibtex(identifiers: list[str]) -> str:
        """BibTeX entries generated by INSPIRE for up to 25 recids, arXiv ids or DOIs."""
        if not identifiers or len(identifiers) > 25:
            raise ToolError("Pass between 1 and 25 identifiers")
        text = await inspire.bibtex(identifiers)
        if not text.strip():
            raise ToolError("No INSPIRE records found for these identifiers")
        return text

    @server.tool(annotations=READ_ONLY)
    @_instrumented
    async def search_authors(name: str, size: int = 5) -> dict[str, Any]:
        """Find author profiles by name. Returns the INSPIRE BAI to use in `a <BAI>` literature queries."""
        authors = await inspire.search_authors(name, size=_clamp(size, 1, 10))
        return {"results": [a.to_dict() for a in authors]}

    @server.resource(
        "inspire://literature/{recid}",
        name="paper",
        description="Metadata and abstract of one INSPIRE literature record",
        mime_type="application/json",
    )
    async def paper_resource(recid: str) -> dict[str, Any]:
        paper = await inspire.get_paper(recid)
        if paper is None:
            raise ValueError(f"No INSPIRE record {recid}")
        return paper.to_dict()

    @server.prompt(description="Survey the literature on a topic using the INSPIRE tools")
    def literature_review(topic: str, max_papers: int = 10) -> str:
        return (
            f"Write a short literature overview on: {topic}.\n"
            f"1. Use semantic_search to find the foundational papers, then search_literature with "
            f"'topcite' and 'date' filters for influential and recent work. Keep at most {max_papers} papers.\n"
            "2. For the most central paper, use get_citing_papers to see how the field developed.\n"
            "3. Group papers by idea, give one sentence each with year and collaboration or first author, "
            "and cite every claim with its INSPIRE url. Say explicitly when the tools found nothing relevant.\n"
            "4. End with BibTeX from get_bibtex for the papers you cited."
        )

    @server.custom_route("/healthz", methods=["GET"], include_in_schema=False)
    async def healthz(request: Request) -> Response:
        return JSONResponse({"status": "ok", "version": __version__})

    return server


async def _semantic(
    inspire: InspireClient, question: str, size: int, pool: int, reranker: Reranker, prefer_influential: bool
):
    pool = _clamp(pool, 10, 100)
    kwargs = {"k": _clamp(size, 1, 25), "pool": pool, "influential": pool // 2 if prefer_influential else 0}
    try:
        return await semantic_search(inspire, question, reranker=reranker, **kwargs)
    except RuntimeError:
        # e.g. the model could not be loaded: degrade to lexical reranking instead of failing the call
        log.warning("semantic reranker unavailable, falling back to BM25")
        return await semantic_search(inspire, question, reranker=BM25Reranker(), **kwargs)


def make_reranker(kind: str) -> Reranker | None:
    """None means auto: decided lazily on the first semantic_search call."""
    factories = {"bm25": BM25Reranker, "dense": DenseReranker, "cross-encoder": CrossEncoderReranker}
    return factories[kind]() if kind in factories else None


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def main() -> None:
    parser = argparse.ArgumentParser(description="INSPIRE-HEP MCP server")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--reranker",
        choices=["auto", "bm25", "dense", "cross-encoder"],
        default=os.environ.get("INSPIRE_MCP_RERANKER", "auto"),
        help="semantic_search reranker: auto = dense if the 'semantic' extra is installed, else bm25",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    server = build_server(reranker=make_reranker(args.reranker))
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
