"""Streamable HTTP serving: host allowlist (DNS rebinding protection) and per-client rate limiting.

The SDK only turns DNS rebinding protection on when binding to localhost; a public deployment binds
to 0.0.0.0, so the allowed host names are passed explicitly here. The rate limit protects both this
server and the INSPIRE API behind it from a single noisy client.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


class RateLimitMiddleware:
    """Token bucket per client: `per_minute` requests on average, bursts up to the same number."""

    def __init__(self, app: ASGIApp, per_minute: int, paths: tuple[str, ...] = ("/mcp",), trust_proxy: bool = False):
        self.app = app
        self.rate = per_minute / 60.0
        self.capacity = float(per_minute)
        self.paths = paths
        self.trust_proxy = trust_proxy
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def _client(self, scope: Scope) -> str:
        if self.trust_proxy:
            for name, value in scope.get("headers", []):
                if name == b"x-forwarded-for":
                    return value.decode().split(",")[0].strip()
        client = scope.get("client")
        return client[0] if client else "unknown"

    def _take(self, key: str) -> float:
        """Consume a token; return 0 if allowed, else the seconds until one is available."""
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens >= 1:
                self._buckets[key] = (tokens - 1, now)
                return 0.0
            self._buckets[key] = (tokens, now)
            if len(self._buckets) > 10_000:  # forget idle clients
                self._buckets = {k: v for k, v in self._buckets.items() if now - v[1] < 600}
            return (1 - tokens) / self.rate

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"].startswith(self.paths):
            wait = self._take(self._client(scope))
            if wait:
                body = b'{"error": "rate limit exceeded, retry later"}'
                await send(
                    {
                        "type": "http.response.start",
                        "status": 429,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"retry-after", str(math.ceil(wait)).encode()),
                            (b"content-length", str(len(body)).encode()),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)


def build_http_app(
    server: MCPServer,
    host: str = "127.0.0.1",
    allowed_hosts: list[str] | None = None,
    rate_limit: int = 0,
    trust_proxy: bool = False,
) -> Starlette | RateLimitMiddleware:
    security = None
    if allowed_hosts:
        hosts = [h for name in allowed_hosts for h in (name, f"{name}:*")]
        origins = [f"{scheme}://{h}" for h in hosts for scheme in ("http", "https")]
        security = TransportSecuritySettings(allowed_hosts=hosts, allowed_origins=origins)
    app = server.streamable_http_app(transport_security=security, host=host)
    return RateLimitMiddleware(app, rate_limit, trust_proxy=trust_proxy) if rate_limit > 0 else app


def serve_http(app: Starlette | RateLimitMiddleware, host: str, port: int, log_level: str = "info") -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level=log_level.lower(), proxy_headers=False)
