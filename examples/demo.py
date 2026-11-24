"""A research session through the MCP protocol, as an assistant would run it.

    uv run --extra semantic python examples/demo.py ["your question"]

Starts the server as a subprocess over stdio (exactly what Claude Desktop or Claude Code do),
then: semantic_search for the question, get_paper on the top hit, get_citing_papers, export_citations.
"""

import asyncio
import sys

from mcp import Client, StdioServerParameters

QUESTION = "using neural networks to accelerate the simulation of calorimeter showers"


async def main(question: str) -> None:
    server = StdioServerParameters(command=sys.executable, args=["-m", "inspire_mcp", "--log-level", "WARNING"])
    async with Client(server) as mcp:
        print(f"> semantic_search: {question!r}")
        found = await mcp.call_tool("semantic_search", {"question": question, "size": 3})
        retrieval = found.structured_content["retrieval"]
        print(f"  {retrieval['candidates_reranked']} candidates from INSPIRE, reranked by {retrieval['reranker']}")
        for i, p in enumerate(found.structured_content["results"], 1):
            print(f"  {i}. {p['title']} ({p['year']}, {p['citation_count']} citations) {p['url']}")

        top = found.structured_content["results"][0]
        print(f"\n> get_paper: {top['arxiv_id']}")
        paper = (await mcp.call_tool("get_paper", {"identifier": top["arxiv_id"]})).structured_content
        print(f"  {', '.join(paper['authors'])}; {paper['journal']}")
        print(f"  {paper['abstract'][:240]}...")

        print(f"\n> get_citing_papers: recid {top['recid']}, most recent")
        citing = (await mcp.call_tool("get_citing_papers", {"recid": top["recid"], "size": 3})).structured_content
        print(f"  {citing['total_citations']} citing papers, latest:")
        for p in citing["results"]:
            print(f"  - {p['title'][:90]} ({p['year']})")

        print("\n> export_citations")
        bibtex = (await mcp.call_tool("export_citations", {"identifiers": [top["arxiv_id"]]})).content[0].text
        print("  " + "\n  ".join(bibtex.strip().splitlines()[:4]) + "\n  ...")


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or QUESTION))
