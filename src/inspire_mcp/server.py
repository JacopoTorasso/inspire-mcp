"""MCP server exposing INSPIRE-HEP literature and author search as tools."""

from __future__ import annotations

import argparse
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from .client import InspireClient, InspireError

INSTRUCTIONS = """\
Tools over INSPIRE-HEP, the literature database of high energy physics.
Prefer INSPIRE search syntax in queries when the user intent is precise:
  t <words>        title words          a <name> / a E.Witten.1   author (name or INSPIRE BAI)
  cn <name>        collaboration        j <journal>               journal
  date > 2020      date filter          topcite 500+              citation count
  arxiv:<id>       arXiv eprint         refersto:recid:<id>       papers citing a record
Plain words also work and search titles, abstracts and keywords.
Always cite papers with their INSPIRE url so the user can verify them.
"""

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=True)
SortArg = Literal["mostrecent", "mostcited"] | None


def build_server(client: InspireClient | None = None) -> MCPServer:
    inspire = client or InspireClient()
    server = MCPServer("inspire-hep", instructions=INSTRUCTIONS)

    @server.tool(annotations=READ_ONLY)
    async def search_literature(query: str, size: int = 10, sort: SortArg = None) -> dict[str, Any]:
        """Search INSPIRE-HEP papers.

        Args:
            query: INSPIRE search query, e.g. "t higgs and cn atlas and date > 2020" or plain words.
            size: number of results to return (1-25).
            sort: "mostrecent", "mostcited", or null for the INSPIRE default ranking.
        """
        total, papers = await inspire.search_literature(query, size=_clamp(size, 1, 25), sort=sort)
        return {"total": total, "results": [p.to_dict() for p in papers]}

    @server.tool(annotations=READ_ONLY)
    async def get_paper(identifier: str) -> dict[str, Any]:
        """Get one paper by INSPIRE record id, arXiv id (e.g. 1207.7214, hep-th/9711200) or DOI."""
        try:
            paper = await inspire.get_paper(identifier)
        except InspireError as exc:
            return {"error": str(exc)}
        return paper.to_dict() if paper else {"error": f"No INSPIRE record found for {identifier!r}"}

    @server.tool(annotations=READ_ONLY)
    async def get_citing_papers(recid: int, size: int = 10, sort: SortArg = "mostrecent") -> dict[str, Any]:
        """List papers that cite the given INSPIRE record (use get_paper first to find the recid)."""
        total, papers = await inspire.citing_papers(recid, size=_clamp(size, 1, 25), sort=sort)
        return {"total_citations": total, "results": [p.to_dict() for p in papers]}

    @server.tool(annotations=READ_ONLY)
    async def get_references(recid: int, size: int = 25) -> dict[str, Any]:
        """List the INSPIRE records referenced by the given paper, most cited first."""
        total, papers = await inspire.references(recid, size=_clamp(size, 1, 50))
        return {"total_references": total, "results": [p.to_dict() for p in papers]}

    @server.tool(annotations=READ_ONLY)
    async def search_authors(name: str, size: int = 5) -> dict[str, Any]:
        """Find author profiles by name. Returns the INSPIRE BAI to use in `a <BAI>` literature queries."""
        authors = await inspire.search_authors(name, size=_clamp(size, 1, 10))
        return {"results": [a.to_dict() for a in authors]}

    return server


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def main() -> None:
    parser = argparse.ArgumentParser(description="INSPIRE-HEP MCP server")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = build_server()
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
