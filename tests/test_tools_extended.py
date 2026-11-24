"""Tools beyond literature search: figures, citations, authors, collections, similar papers."""

from datetime import date, timedelta

import httpx
import respx
from mcp import Client

from inspire_mcp import InspireClient, build_server
from inspire_mcp.client import BASE_URL
from inspire_mcp.community import parse_experiment, parse_institution, parse_job
from inspire_mcp.semantic import BM25Reranker

from .conftest import ATLAS_HIGGS, WITTEN, hits

FIGURES = {
    "control_number": 1598455,
    "figures": [
        {
            "url": "https://inspirehep.net/files/a",
            "caption": r"Energy from \textsc{Geant4} in $\gamma$ showers",
            "filename": "fig1.png",
        },
        {"url": "https://inspirehep.net/files/b", "caption": "Second"},
    ],
}
FACETS = {
    "aggregations": {
        "citation_summary": {
            "doc_count": 197,
            "h-index": {"value": {"all": 118, "published": 110}},
            "citations": {
                "buckets": {
                    "all": {
                        "citations_count": {"value": 98201.0},
                        "citation_buckets": {
                            "buckets": [{"key": "0--0", "doc_count": 7}, {"key": "500--", "doc_count": 39}]
                        },
                    }
                }
            },
        }
    }
}
CONFERENCE = {
    "control_number": 3222759,
    "titles": [{"title": "Machine Learning Physics across Scales"}],
    "opening_date": "2026-12-01",
    "closing_date": "2026-12-03",
    "addresses": [{"cities": ["Tokyo"], "country": "Japan"}],
    "urls": [{"value": "https://indico2.riken.jp/event/5793/"}],
    "inspire_categories": [{"term": "Computing"}],
}
SEMINAR = {
    "control_number": 3217038,
    "title": {"title": "Heavy-quark masses and HVP"},
    "start_datetime": "2026-11-25T11:00:00",
    "timezone": "Europe/Madrid",
    "speakers": [{"name": "Rivera Vargas, Antonio", "affiliations": [{"value": "Barcelona, IFAE"}]}],
    "join_urls": [{"value": "https://zoom.example/j/1"}],
    "series": [{"name": "ICCUB"}],
}
JOB = {
    "control_number": 3231470,
    "position": "Postdoctoral positions in Theoretical Physics",
    "institutions": [{"value": "U. Pisa (main)"}],
    "regions": ["Europe"],
    "ranks": ["POSTDOC"],
    "arxiv_categories": ["hep-th"],
    "deadline_date": "2026-12-18",
    "status": "open",
    "description": "<div>Several <b>postdoctoral</b> positions &amp; more</div>",
    "urls": [{"value": "https://apply.example/1"}],
    "contact_details": [{"name": "Somebody", "email": "somebody@example.org"}],
}


def server():
    return build_server(InspireClient(min_interval=0, cache_ttl=0), reranker=BM25Reranker())


@respx.mock
async def test_figures_have_readable_captions_and_limit():
    respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(200, json=hits(FIGURES)))
    async with Client(server()) as client:
        sc = (await client.call_tool("get_paper_figures", {"identifier": "1705.02355", "limit": 1})).structured_content
    assert sc["total"] == 2 and len(sc["figures"]) == 1
    assert sc["figures"][0]["caption"] == r"Energy from Geant4 in \gamma showers"


@respx.mock
async def test_export_citations_latex_style():
    route = respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(200, text="\\bibitem{ATLAS:2012yve}"))
    async with Client(server()) as client:
        result = await client.call_tool("export_citations", {"identifiers": ["1207.7214"], "style": "latex-us"})
    assert route.calls.last.request.url.params["format"] == "latex-us"
    assert result.content[0].text.startswith("\\bibitem")


@respx.mock
async def test_author_profile_resolves_bai_and_reads_metrics():
    authors = respx.get(f"{BASE_URL}/authors").mock(return_value=httpx.Response(200, json=hits(WITTEN)))
    facets = respx.get(f"{BASE_URL}/literature/facets").mock(return_value=httpx.Response(200, json=FACETS))
    async with Client(server()) as client:
        sc = (await client.call_tool("get_author_profile", {"author": "Edward.Witten.1"})).structured_content
    assert authors.calls.last.request.url.params["q"] == "ids.value:Edward.Witten.1"
    assert facets.calls.last.request.url.params["q"] == "a Edward.Witten.1"
    assert sc["metrics"] == {
        "papers": 197,
        "citations": 98201,
        "h_index": 118,
        "h_index_published": 110,
        "citation_distribution": {"0-0": 7, "500-": 39},
    }


@respx.mock
async def test_author_papers_by_orcid_and_unknown_author():
    respx.get(f"{BASE_URL}/authors").mock(
        side_effect=[httpx.Response(200, json=hits(WITTEN)), httpx.Response(200, json=hits())]
    )
    lit = respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(200, json=hits(ATLAS_HIGGS, total=30)))
    async with Client(server()) as client:
        found = await client.call_tool("get_author_papers", {"author": "0000-0002-7752-6073", "size": 10})
        missing = await client.call_tool("get_author_papers", {"author": "Nobody Atall"})
    assert lit.calls.last.request.url.params["q"] == "a Edward.Witten.1"
    assert found.structured_content["next_page"] == 2
    assert missing.is_error and "search_authors" in missing.content[0].text


@respx.mock
async def test_collaboration_query_is_combined():
    route = respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(200, json=hits(ATLAS_HIGGS)))
    async with Client(server()) as client:
        await client.call_tool("search_collaboration", {"collaboration": "ATLAS", "query": "t dark matter"})
    assert route.calls.last.request.url.params["q"] == 'cn "ATLAS" and (t dark matter)'


@respx.mock
async def test_upcoming_events_filter_by_date():
    past = {**CONFERENCE, "control_number": 1, "opening_date": "2001-01-01"}
    later = {**CONFERENCE, "control_number": 2, "opening_date": (date.today() + timedelta(days=60)).isoformat()}
    conf = respx.get(f"{BASE_URL}/conferences").mock(
        return_value=httpx.Response(200, json=hits(later, CONFERENCE, past))
    )
    sem = respx.get(f"{BASE_URL}/seminars").mock(return_value=httpx.Response(200, json=hits(SEMINAR)))
    async with Client(server()) as client:
        c = (await client.call_tool("search_conferences", {"query": "machine learning"})).structured_content
        await client.call_tool("search_seminars", {})
    q = conf.calls.last.request.url.params
    # keyword search sorted newest first, dates filtered here: INSPIRE mixes free text and ranges unreliably
    assert (q["q"], q["sort"]) == ("machine learning", "datedesc")
    assert [r["recid"] for r in c["results"]] == [3222759, 2] and c["total"] == 2
    assert c["results"][0]["place"] == "Tokyo, Japan"
    s = sem.calls.last.request.url.params
    assert s["q"].startswith("start_datetime:>") and s["sort"] == "dateasc"


@respx.mock
async def test_author_papers_can_skip_large_collaborations():
    respx.get(f"{BASE_URL}/authors").mock(return_value=httpx.Response(200, json=hits(WITTEN)))
    lit = respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(200, json=hits(ATLAS_HIGGS)))
    async with Client(server()) as client:
        await client.call_tool("get_author_papers", {"author": "Edward.Witten.1", "max_authors": 10})
    assert lit.calls.last.request.url.params["q"] == "a Edward.Witten.1 and ac 1->10"


@respx.mock
async def test_jobs_are_open_only_and_drop_contact_details():
    route = respx.get(f"{BASE_URL}/jobs").mock(return_value=httpx.Response(200, json=hits(JOB)))
    async with Client(server()) as client:
        sc = (await client.call_tool("search_jobs", {"query": "postdoc"})).structured_content
    assert route.calls.last.request.url.params["q"] == "status:open and (postdoc)"
    job = sc["results"][0]
    assert job["description"] == "Several postdoctoral positions & more"
    assert "contact_details" not in job and "somebody@example.org" not in str(job)


def test_seminar_speaker_without_affiliation():
    from inspire_mcp.community import parse_seminar

    assert parse_seminar({"control_number": 1, "speakers": [{"name": "Doe, J."}]}).speakers == ["Doe, J."]


def test_institution_and_experiment_parsers():
    inst = parse_institution(
        {
            "control_number": 902725,
            "legacy_ICN": "CERN",
            "number_of_papers": 79049,
            "addresses": [
                {"cities": ["Prévessin"], "country": "France"},
                {"cities": ["Geneva"], "country": "Switzerland"},
            ],
        }
    )
    assert inst.place == "Prévessin, France; Geneva, Switzerland"
    exp = parse_experiment(
        {
            "control_number": 1,
            "legacy_name": "CERN-LHC-LHCb",
            "accelerator": {"value": "LHC"},
            "date_started": "2009-11-23",
            "date_completed": "2008-09",
        }
    )
    assert exp.accelerator == "LHC" and exp.completed == "2008-09" and not hasattr(exp, "status")
    assert parse_job({"control_number": 2}).description is None


@respx.mock
async def test_similar_papers_exclude_the_seed_paper():
    seed = {**ATLAS_HIGGS, "keywords": [{"value": "Higgs particle"}]}
    other = {**ATLAS_HIGGS, "control_number": 2, "titles": [{"title": "Another Higgs observation"}]}
    respx.get(f"{BASE_URL}/literature").mock(
        side_effect=[httpx.Response(200, json=hits(seed))] + [httpx.Response(200, json=hits(seed, other))] * 4
    )
    async with Client(server()) as client:
        sc = (await client.call_tool("find_similar_papers", {"identifier": "1207.7214"})).structured_content
    assert sc["paper"]["recid"] == 1124337
    assert [p["recid"] for p in sc["results"]] == [2]


async def test_ping_needs_no_network():
    async with Client(server()) as client:
        sc = (await client.call_tool("ping", {})).structured_content
    assert sc["status"] == "ok"
