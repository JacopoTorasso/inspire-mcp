from starlette.testclient import TestClient

from inspire_mcp import InspireClient, build_server
from inspire_mcp.http import RateLimitMiddleware, build_http_app
from inspire_mcp.semantic import BM25Reranker

INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}},
}
HEADERS = {"accept": "application/json, text/event-stream", "content-type": "application/json"}


def server():
    return build_server(InspireClient(min_interval=0), reranker=BM25Reranker())


def test_rate_limit_answers_429_with_retry_after_and_spares_probes():
    app = build_http_app(server(), host="0.0.0.0", rate_limit=2)
    assert isinstance(app, RateLimitMiddleware)
    with TestClient(app) as http:
        codes = [http.post("/mcp", json=INIT, headers=HEADERS).status_code for _ in range(3)]
        assert codes[:2] != [429, 429] and codes[2] == 429
        limited = http.post("/mcp", json=INIT, headers=HEADERS)
        assert limited.status_code == 429 and int(limited.headers["retry-after"]) >= 1
        assert http.get("/healthz").status_code == 200  # probes are never rate limited


def test_rate_limit_keys_on_forwarded_client_only_behind_proxy():
    mw = RateLimitMiddleware(None, 60, trust_proxy=True)  # type: ignore[arg-type]
    scope = {"client": ("10.0.0.1", 1), "headers": [(b"x-forwarded-for", b"203.0.113.7, 10.0.0.1")]}
    assert mw._client(scope) == "203.0.113.7"
    assert RateLimitMiddleware(None, 60)._client(scope) == "10.0.0.1"  # type: ignore[arg-type]


def test_allowed_hosts_reject_unknown_host_header():
    def app():
        return build_http_app(server(), host="0.0.0.0", allowed_hosts=["mcp.example.org"])

    with TestClient(app(), base_url="http://evil.example") as http:
        assert http.post("/mcp", json=INIT, headers=HEADERS).status_code in (400, 403, 421)
    with TestClient(app(), base_url="http://mcp.example.org") as http:
        assert http.post("/mcp", json=INIT, headers=HEADERS).status_code == 200


def test_port_falls_back_to_platform_variable(monkeypatch):
    from inspire_mcp.server import parse_args

    monkeypatch.delenv("INSPIRE_MCP_PORT", raising=False)
    monkeypatch.setenv("PORT", "10000")
    assert parse_args([]).port == 10000
