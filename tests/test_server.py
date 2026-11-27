import json

import httpx
import respx
from mcp import Client

from inspire_mcp import InspireClient, build_server
from inspire_mcp.client import BASE_URL
from inspire_mcp.semantic import BM25Reranker

from .conftest import ATLAS_HIGGS, WITTEN, hits

EXPECTED_TOOLS = {
    "semantic_search",
    "search_literature",
    "get_paper",
    "find_similar_papers",
    "get_citing_papers",
    "get_references",
    "get_paper_figures",
    "export_citations",
    "search_collaboration",
    "search_authors",
    "get_author_profile",
    "get_author_papers",
    "search_conferences",
    "search_seminars",
    "search_jobs",
    "search_institutions",
    "search_experiments",
    "ping",
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
    assert route.calls[0].request.url.params["q"] == "arxiv:1207.7214"
    assert result.structured_content["recid"] == 1124337
    assert result.structured_content["pdf_url"] == "https://arxiv.org/pdf/1207.7214"


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
    assert sc["results"][0]["recid"] == 1124337 and "score" in sc["results"][0]
    assert sc["retrieval"]["reranker"] == "bm25" and sc["retrieval"]["inspire_query"] == "discovery or higgs or boson"


@respx.mock
async def test_export_citations_requests_bibtex_format():
    route = respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(200, text="@article{ATLAS:2012yve,}"))
    async with Client(server()) as client:
        result = await client.call_tool("export_citations", {"identifiers": ["1207.7214", "hep-th/9711200"]})
    req = route.calls.last.request
    assert req.url.params["q"] == "arxiv:1207.7214 or arxiv:hep-th/9711200"
    assert req.headers["accept"] == "*/*" and req.url.params["format"] == "bibtex"
    assert result.content[0].text.startswith("@article")


async def test_bibtex_rejects_empty_list():
    async with Client(server()) as client:
        result = await client.call_tool("export_citations", {"identifiers": []})
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


def test_options_read_environment(monkeypatch):
    from inspire_mcp.server import parse_args

    monkeypatch.setenv("INSPIRE_MCP_TRANSPORT", "streamable-http")
    monkeypatch.setenv("INSPIRE_MCP_PORT", "9001")
    monkeypatch.setenv("INSPIRE_MCP_RERANKER", "bm25")
    args = parse_args([])
    assert (args.transport, args.port, args.reranker) == ("streamable-http", 9001, "bm25")
    assert args.reranker_model is None
    monkeypatch.setenv("INSPIRE_MCP_RERANKER_MODEL", "/models/physics")
    assert parse_args([]).reranker_model == "/models/physics"
    assert parse_args(["--port", "7000"]).port == 7000  # flags win over environment


def test_json_log_format():
    import json
    import logging

    from inspire_mcp.server import _JsonFormatter

    record = logging.LogRecord("inspire_mcp", logging.INFO, __file__, 1, "tool=%s", ("get_paper",), None)
    entry = json.loads(_JsonFormatter().format(record))
    assert entry["message"] == "tool=get_paper" and entry["level"] == "INFO"


@respx.mock
async def test_search_pagination_and_compact_results():
    route = respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(200, json=hits(ATLAS_HIGGS, total=23)))
    async with Client(server()) as client:
        first = await client.call_tool("search_literature", {"query": "higgs", "size": 10, "compact": True})
        last = await client.call_tool("search_literature", {"query": "higgs", "size": 10, "page": 3})
    sc = first.structured_content
    assert (sc["page"], sc["has_more"], sc["next_page"]) == (1, True, 2)
    assert "abstract" not in sc["results"][0] and sc["results"][0]["citation_count"] == 17848
    assert last.structured_content["has_more"] is False and last.structured_content["next_page"] is None
    assert route.calls.last.request.url.params["page"] == "3"


def test_semantic_results_keep_full_abstracts_only_at_the_top():
    from inspire_mcp.client import Paper
    from inspire_mcp.server import ABSTRACT_EXCERPT, FULL_ABSTRACTS, _ranked_result

    long = "word " * 200
    paper = Paper(1, "t", long, ["A, B", "C, D"], 2, [], 2020, "J", "2001.00001", "d", 5, ["Th"], "u", ["k"], "p")
    top = _ranked_result(paper, FULL_ABSTRACTS - 1, compact=False)
    later = _ranked_result(paper, FULL_ABSTRACTS, compact=False)
    assert top["abstract"] == long and top["authors"] == ["A, B", "C, D"] and top["doi"] == "d"
    assert later["authors"] == ["A, B"] and later["author_count"] == 2 and later["arxiv_id"] == "2001.00001"
    assert not {"journal", "doi", "collaborations"} & later.keys()
    assert later["abstract"].endswith(" …") and len(later["abstract"]) <= ABSTRACT_EXCERPT + 2
    assert not {"keywords", "categories", "pdf_url"} & (top.keys() | later.keys()) and later["url"] == "u"
    assert "abstract" not in _ranked_result(paper, 0, compact=True)


@respx.mock
async def test_tool_results_are_compact_json_with_the_same_structured_content():
    respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(200, json=hits(ATLAS_HIGGS)))
    async with Client(server()) as client:
        result = await client.call_tool("search_literature", {"query": "higgs"})
    text = result.content[0].text
    assert "\n" not in text and ", " not in text.split('"abstract"')[0]
    assert json.loads(text) == result.structured_content and result.structured_content["total"] == 1


async def test_listed_tool_schemas_carry_no_pydantic_titles():
    from inspire_mcp.server import _without_titles

    async with Client(server()) as client:
        tools = (await client.list_tools()).tools
    assert all('"title"' not in json.dumps(t.input_schema) for t in tools)
    semantic = next(t for t in tools if t.name == "semantic_search")
    assert (
        semantic.input_schema["required"] == ["question"]
        and semantic.input_schema["properties"]["size"]["default"] == 10
    )
    kept = {"properties": {"title": {"title": "Title", "type": "string"}}, "title": "fArguments"}
    assert _without_titles(kept) == {"properties": {"title": {"type": "string"}}}
