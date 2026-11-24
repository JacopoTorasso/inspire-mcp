import httpx
import respx
from mcp import Client

from inspire_mcp import InspireClient, build_server
from inspire_mcp.client import BASE_URL
from inspire_mcp.semantic import BM25Reranker

from .conftest import ATLAS_HIGGS, WITTEN, hits

EXPECTED_TOOLS = {
    "search_literature",
    "semantic_search",
    "get_paper",
    "get_citing_papers",
    "get_references",
    "get_bibtex",
    "search_authors",
}


def server():
    return build_server(InspireClient(min_interval=0), reranker=BM25Reranker())


async def test_tools_are_listed_and_read_only():
    async with Client(server()) as client:
        tools = (await client.list_tools()).tools
    assert {t.name for t in tools} == EXPECTED_TOOLS
    assert all(t.annotations and t.annotations.read_only_hint for t in tools)


@respx.mock
async def test_get_paper_by_arxiv_id():
    route = respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(200, json=hits(ATLAS_HIGGS)))
    async with Client(server()) as client:
        result = await client.call_tool("get_paper", {"identifier": "1207.7214"})
    assert route.calls.last.request.url.params["q"] == "arxiv:1207.7214"
    assert result.structured_content["recid"] == 1124337


@respx.mock
async def test_get_paper_not_found_and_invalid_identifier():
    respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(200, json=hits()))
    async with Client(server()) as client:
        missing = await client.call_tool("get_paper", {"identifier": "9999999999"})
        invalid = await client.call_tool("get_paper", {"identifier": "not an id"})
    assert missing.is_error and "No INSPIRE record" in missing.content[0].text
    assert invalid.is_error and "Unrecognised identifier" in invalid.content[0].text


@respx.mock
async def test_search_size_is_clamped():
    route = respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(200, json=hits(ATLAS_HIGGS)))
    async with Client(server()) as client:
        await client.call_tool("search_literature", {"query": "higgs", "size": 500})
    assert route.calls.last.request.url.params["size"] == "25"


@respx.mock
async def test_search_authors():
    respx.get(f"{BASE_URL}/authors").mock(return_value=httpx.Response(200, json=hits(WITTEN)))
    async with Client(server()) as client:
        result = await client.call_tool("search_authors", {"name": "witten"})
    assert result.structured_content["results"][0]["bai"] == "Edward.Witten.1"


@respx.mock
async def test_api_failure_becomes_tool_error():
    respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(404))
    async with Client(server()) as client:
        result = await client.call_tool("search_literature", {"query": "higgs"})
    assert result.is_error and "HTTP 404" in result.content[0].text


@respx.mock
async def test_semantic_search_tool_reports_retrieval_details():
    respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(200, json=hits(ATLAS_HIGGS)))
    async with Client(server()) as client:
        result = await client.call_tool("semantic_search", {"question": "discovery of the higgs boson", "pool": 500})
    sc = result.structured_content
    assert sc["results"][0]["recid"] == 1124337 and "similarity" in sc["results"][0]
    assert sc["retrieval"]["reranker"] == "bm25" and sc["retrieval"]["inspire_query"] == "discovery or higgs or boson"


@respx.mock
async def test_get_bibtex_requests_bibtex_format():
    route = respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(200, text="@article{ATLAS:2012yve,}"))
    async with Client(server()) as client:
        result = await client.call_tool("get_bibtex", {"identifiers": ["1207.7214", "hep-th/9711200"]})
    req = route.calls.last.request
    assert req.url.params["q"] == "arxiv:1207.7214 or arxiv:hep-th/9711200"
    assert req.headers["accept"] == "application/x-bibtex"
    assert result.content[0].text.startswith("@article")


async def test_bibtex_rejects_empty_list():
    async with Client(server()) as client:
        result = await client.call_tool("get_bibtex", {"identifiers": []})
    assert result.is_error


@respx.mock
async def test_paper_resource_and_prompt():
    respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(200, json=hits(ATLAS_HIGGS)))
    async with Client(server()) as client:
        res = await client.read_resource("inspire://literature/1124337")
        prompt = await client.get_prompt("literature_review", {"topic": "pentaquarks"})
    assert '"recid": 1124337' in res.contents[0].text
    assert "pentaquarks" in prompt.messages[0].content.text


@respx.mock
async def test_responses_are_cached():
    route = respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(200, json=hits(ATLAS_HIGGS)))
    async with InspireClient(min_interval=0) as client:
        await client.search_literature("higgs")
        await client.search_literature("higgs")
    assert route.call_count == 1


@respx.mock
async def test_prefer_influential_adds_citation_filtered_query():
    route = respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(200, json=hits(ATLAS_HIGGS)))
    async with Client(server()) as client:
        await client.call_tool("semantic_search", {"question": "higgs discovery", "prefer_influential": True})
    queries = [c.request.url.params["q"] for c in route.calls]
    assert "(higgs or discovery) and topcite 250+" in queries
