import json

import httpx
import pytest
import respx

from inspire_mcp import cli
from inspire_mcp.client import BASE_URL
from inspire_mcp.server import main

from .conftest import ATLAS_HIGGS, hits


@pytest.fixture(autouse=True)
def lexical_only(monkeypatch):
    # Keep CLI tests offline and fast: no model loading.
    from inspire_mcp import semantic

    monkeypatch.setattr(semantic, "default_reranker", lambda: semantic.BM25Reranker())


def test_config_prints_ready_to_paste_json(capsys):
    assert cli.run(["config"]) == 0
    out = capsys.readouterr().out
    assert "claude mcp add inspire" in out
    snippet = out[out.index("{") : out.rindex("}") + 1]
    assert json.loads(snippet)["mcpServers"]["inspire"]["command"] == "uvx"


@respx.mock
def test_search_prints_ranked_results(capsys):
    respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(200, json=hits(ATLAS_HIGGS)))
    assert cli.run(["search", "higgs", "discovery", "-n", "3"]) == 0
    out = capsys.readouterr().out
    assert "1. Observation of a new particle" in out and "17848 citations" in out and "ATLAS 2012" in out


@respx.mock
def test_search_json_and_no_results(capsys):
    def by_query(request):
        found = "higgs" in request.url.params["q"]
        return httpx.Response(200, json=hits(ATLAS_HIGGS) if found else hits())

    respx.get(f"{BASE_URL}/literature").mock(side_effect=by_query)
    assert cli.run(["search", "higgs", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["recid"] == 1124337
    assert cli.run(["search", "nothing", "here"]) == 1


@respx.mock
def test_paper_shows_details_and_links(capsys):
    respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(200, json=hits(ATLAS_HIGGS)))
    assert cli.run(["paper", "1207.7214"]) == 0
    out = capsys.readouterr().out
    assert "and 2927 more" in out and "https://arxiv.org/pdf/1207.7214" in out


@respx.mock
def test_doctor_reports_unreachable_inspire(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("INSPIRE_MCP_CACHE", str(tmp_path))
    monkeypatch.setattr("inspire_mcp.client._retry_after", lambda resp, attempt: 0)
    respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(503))
    assert cli.run(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "✗ INSPIRE API reachable" in out and "fix:" in out and "✓ Cache directory writable" in out


def test_main_dispatches_subcommands_and_keeps_server_default(monkeypatch):
    called = {}
    monkeypatch.setattr(cli, "run", lambda argv: called.setdefault("cli", argv) and 0)
    with pytest.raises(SystemExit):
        main(["config"])
    assert called["cli"] == ["config"]
