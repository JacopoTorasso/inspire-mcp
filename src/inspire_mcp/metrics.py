"""Minimal Prometheus metrics, rendered in the text exposition format on /metrics.

Kept dependency free on purpose: three metric families are all the server needs.
"""

from __future__ import annotations

import threading
from collections import defaultdict

BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: dict[tuple[str, str], int] = defaultdict(int)
        self._buckets: dict[str, list[int]] = defaultdict(lambda: [0] * len(BUCKETS))
        self._sum: dict[str, float] = defaultdict(float)
        self._count: dict[str, int] = defaultdict(int)
        self.reranker = "none"
        self.ready = False

    def observe(self, tool: str, outcome: str, seconds: float) -> None:
        with self._lock:
            self._calls[(tool, outcome)] += 1
            self._sum[tool] += seconds
            self._count[tool] += 1
            for i, bound in enumerate(BUCKETS):
                if seconds <= bound:
                    self._buckets[tool][i] += 1

    def render(self) -> str:
        lines = [
            "# HELP inspire_mcp_tool_calls_total Tool calls by tool and outcome.",
            "# TYPE inspire_mcp_tool_calls_total counter",
        ]
        with self._lock:
            for (tool, outcome), n in sorted(self._calls.items()):
                lines.append(f'inspire_mcp_tool_calls_total{{tool="{tool}",outcome="{outcome}"}} {n}')
            lines += [
                "# HELP inspire_mcp_tool_latency_seconds Tool call latency.",
                "# TYPE inspire_mcp_tool_latency_seconds histogram",
            ]
            for tool in sorted(self._count):
                for bound, n in zip(BUCKETS, self._buckets[tool], strict=True):
                    lines.append(f'inspire_mcp_tool_latency_seconds_bucket{{tool="{tool}",le="{bound}"}} {n}')
                lines.append(f'inspire_mcp_tool_latency_seconds_bucket{{tool="{tool}",le="+Inf"}} {self._count[tool]}')
                lines.append(f'inspire_mcp_tool_latency_seconds_sum{{tool="{tool}"}} {self._sum[tool]:.6f}')
                lines.append(f'inspire_mcp_tool_latency_seconds_count{{tool="{tool}"}} {self._count[tool]}')
        lines += [
            "# HELP inspire_mcp_ready 1 when the reranker is loaded and the server is ready.",
            "# TYPE inspire_mcp_ready gauge",
            f'inspire_mcp_ready{{reranker="{self.reranker}"}} {int(self.ready)}',
        ]
        return "\n".join(lines) + "\n"
