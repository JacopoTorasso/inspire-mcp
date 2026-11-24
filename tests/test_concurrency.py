import asyncio
import time

import httpx
import respx
from mcp import Client

from inspire_mcp import InspireClient, build_server
from inspire_mcp.client import BASE_URL

from .conftest import ATLAS_HIGGS, hits


class SlowReranker:
    """CPU-bound like a real model: blocks its thread for half a second."""

    name = "slow"
    fuse_with_inspire = False

    def scores(self, query, docs):
        time.sleep(0.5)
        return [0.0 for _ in docs]


@respx.mock
async def test_reranking_does_not_block_other_requests():
    respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(200, json=hits(ATLAS_HIGGS)))
    server = build_server(InspireClient(min_interval=0, cache_ttl=0), reranker=SlowReranker())
    async with Client(server) as client:
        done: dict[str, float] = {}
        start = time.perf_counter()

        async def slow():
            await client.call_tool("semantic_search", {"question": "higgs boson"})
            done["semantic"] = time.perf_counter() - start

        async def fast():
            await asyncio.sleep(0.05)  # let the slow call reach the reranker first
            await client.call_tool("search_literature", {"query": "higgs"})
            done["search"] = time.perf_counter() - start

        await asyncio.gather(slow(), fast())
    assert done["search"] < 0.3, done
    assert done["semantic"] >= 0.5


def test_readyz_waits_for_warmup_and_metrics_count_calls():
    from starlette.testclient import TestClient

    from inspire_mcp.metrics import Metrics

    metrics = Metrics()
    server = build_server(InspireClient(min_interval=0), reranker=SlowReranker(), warmup=True, metrics=metrics)
    app = server.streamable_http_app()
    with TestClient(app) as http:
        assert http.get("/healthz").status_code == 200
        first = http.get("/readyz")
        assert first.status_code == 503 and first.json()["ready"] is False
        deadline = time.time() + 5
        while http.get("/readyz").status_code != 200 and time.time() < deadline:
            time.sleep(0.05)
        assert http.get("/readyz").json() == {"ready": True, "reranker": "slow"}
        metrics.observe("get_paper", "ok", 0.07)
        body = http.get("/metrics").text
    assert 'inspire_mcp_tool_calls_total{tool="get_paper",outcome="ok"} 1' in body
    assert 'inspire_mcp_tool_latency_seconds_bucket{tool="get_paper",le="0.1"} 1' in body
    assert 'inspire_mcp_ready{reranker="slow"} 1' in body


@respx.mock
async def test_tool_calls_are_recorded_in_metrics():
    from inspire_mcp.metrics import Metrics

    respx.get(f"{BASE_URL}/literature").mock(return_value=httpx.Response(500))
    metrics = Metrics()
    async with Client(build_server(InspireClient(min_interval=0), metrics=metrics)) as client:
        await client.call_tool("search_literature", {"query": "higgs"})
        await client.call_tool("export_citations", {"identifiers": []})
    body = metrics.render()
    assert 'tool="search_literature",outcome="upstream_error"} 1' in body
    assert 'tool="export_citations",outcome="error"} 1' in body
