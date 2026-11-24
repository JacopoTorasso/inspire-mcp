import httpx
import pytest
import respx

from inspire_mcp.client import BASE_URL, InspireClient, InspireError, identifier_query, parse_author, parse_paper

from .conftest import ATLAS_HIGGS, MATHML_TITLE, WITTEN, hits


@pytest.mark.parametrize(
    ("ident", "query"),
    [
        ("1124337", "recid:1124337"),
        ("1207.7214", "arxiv:1207.7214"),
        ("arXiv:1207.7214v2", "arxiv:1207.7214"),
        ("hep-th/9711200", "arxiv:hep-th/9711200"),
        ("10.1016/j.physletb.2012.08.020", 'doi:"10.1016/j.physletb.2012.08.020"'),
        ("doi:10.1103/PhysRevLett.116.061102", 'doi:"10.1103/PhysRevLett.116.061102"'),
    ],
)
def test_identifier_query(ident, query):
    assert identifier_query(ident) == query


def test_identifier_query_rejects_free_text():
    with pytest.raises(InspireError):
        identifier_query("higgs boson")


def test_parse_paper_prefers_arxiv_abstract_and_truncates_authors():
    paper = parse_paper(ATLAS_HIGGS)
    assert paper.abstract.startswith("A search for the Standard Model")
    assert len(paper.authors) == 5 and paper.author_count == 2932
    assert paper.journal == "Phys.Lett.B 716 1" and paper.year == 2012
    assert paper.arxiv_id == "1207.7214" and paper.collaborations == ["ATLAS"]
    assert paper.url == "https://inspirehep.net/literature/1124337"


def test_parse_paper_strips_mathml_and_falls_back_to_earliest_date():
    paper = parse_paper(MATHML_TITLE)
    assert paper.title == "Observation of a J/ψ state"
    assert paper.year == 2023 and paper.abstract is None


def test_parse_author_current_institution():
    author = parse_author(WITTEN)
    assert (author.bai, author.current_institution) == ("Edward.Witten.1", "Princeton, Inst. Advanced Study")


@respx.mock
async def test_search_sends_query_fields_and_sort():
    route = respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(200, json=hits(ATLAS_HIGGS, total=42)))
    async with InspireClient(min_interval=0) as client:
        total, papers = await client.search_literature("cn atlas", size=3, sort="mostcited")
    params = route.calls.last.request.url.params
    assert (params["q"], params["size"], params["sort"]) == ("cn atlas", "3", "mostcited")
    assert "citation_count" in params["fields"]
    assert total == 42 and papers[0].recid == 1124337


@respx.mock
async def test_retries_on_rate_limit(monkeypatch):
    monkeypatch.setattr("inspire_mcp.client._retry_after", lambda resp, attempt: 0)
    respx.get(f"{BASE_URL}/literature").mock(
        side_effect=[httpx.Response(429), httpx.Response(200, json=hits(ATLAS_HIGGS))]
    )
    async with InspireClient(min_interval=0) as client:
        _, papers = await client.search_literature("x")
    assert len(papers) == 1


@respx.mock
async def test_http_error_raises():
    respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(500))
    async with InspireClient(min_interval=0) as client:
        with pytest.raises(InspireError, match="500"):
            await client.search_literature("x")


@respx.mock
async def test_citing_papers_uses_refersto():
    route = respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(200, json=hits(total=17848)))
    async with InspireClient(min_interval=0) as client:
        total, _ = await client.citing_papers(1124337, size=2)
    assert route.calls.last.request.url.params["q"] == "refersto:recid:1124337" and total == 17848
