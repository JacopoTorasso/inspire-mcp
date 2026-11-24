import httpx
import respx
from mcp import Client

from inspire_mcp import InspireClient, build_server
from inspire_mcp.client import BASE_URL

from .conftest import ATLAS_HIGGS, WITTEN, hits

EXPECTED_TOOLS = {"search_literature", "get_paper", "get_citing_papers", "get_references", "search_authors"}


async def test_tools_are_listed_and_read_only():
    async with Client(build_server(InspireClient(min_interval=0))) as client:
        tools = (await client.list_tools()).tools
    assert {t.name for t in tools} == EXPECTED_TOOLS
    assert all(t.annotations and t.annotations.read_only_hint for t in tools)


@respx.mock
async def test_get_paper_by_arxiv_id():
    route = respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(200, json=hits(ATLAS_HIGGS)))
    async with Client(build_server(InspireClient(min_interval=0))) as client:
        result = await client.call_tool("get_paper", {"identifier": "1207.7214"})
    assert route.calls.last.request.url.params["q"] == "arxiv:1207.7214"
    assert result.structured_content["recid"] == 1124337


@respx.mock
async def test_get_paper_not_found_and_invalid_identifier():
    respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(200, json=hits()))
    async with Client(build_server(InspireClient(min_interval=0))) as client:
        missing = await client.call_tool("get_paper", {"identifier": "9999999999"})
        invalid = await client.call_tool("get_paper", {"identifier": "not an id"})
    assert "No INSPIRE record" in missing.structured_content["error"]
    assert "Unrecognised identifier" in invalid.structured_content["error"]


@respx.mock
async def test_search_size_is_clamped():
    route = respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(200, json=hits(ATLAS_HIGGS)))
    async with Client(build_server(InspireClient(min_interval=0))) as client:
        await client.call_tool("search_literature", {"query": "higgs", "size": 500})
    assert route.calls.last.request.url.params["size"] == "25"


@respx.mock
async def test_search_authors():
    respx.get(f"{BASE_URL}/authors").mock(return_value=httpx.Response(200, json=hits(WITTEN)))
    async with Client(build_server(InspireClient(min_interval=0))) as client:
        result = await client.call_tool("search_authors", {"name": "witten"})
    assert result.structured_content["results"][0]["bai"] == "Edward.Witten.1"
