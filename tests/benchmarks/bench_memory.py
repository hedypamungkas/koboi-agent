"""Memory-regression benchmarks (peak allocated bytes via tracemalloc).

Latency benchmarks (bench_core/hooks/rag/tui/loop/server) catch speed
regressions but miss leaks -- a change that holds onto 10x more transient
memory can keep the same wall-time yet still be a regression (long-running
server sessions, large RAG corpora). These measure peak allocated bytes for
representative operations and assert a generous ceiling so an egregious leak
fails loudly without flaking on normal variance.

NOTE: tracemalloc instruments every allocation, so the *time* numbers here are
inflated and meaningless -- read ``benchmark.extra_info["peak_kb"]``, not the
time column. The peak-bytes metric itself is low-variance and deterministic
enough to gate absolutely.
"""

from __future__ import annotations

import asyncio
import tracemalloc

from koboi.config import Config
from koboi.loop import AgentCore
from koboi.memory import ConversationMemory
from koboi.rag.retriever import KeywordRetriever
from koboi.server import create_app
from tests.conftest import MockClient, make_mock_response, make_mock_tool_call, make_tool_registry

# Generous ceilings (bytes) -- catch egregious leaks, not minor variance.
# A turn / index / boot should never approach these; tune down once CI baseline
# (W1.4/W1.5) characterizes the real footprint.
_CEIL_TURN = 50 * 1024 * 1024  # 50 MB
_CEIL_RAG = 50 * 1024 * 1024
_CEIL_BOOT = 100 * 1024 * 1024


def _peak_bytes(fn) -> int:
    """Run fn under tracemalloc and return its peak allocated bytes."""
    tracemalloc.start()
    try:
        fn()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def _tool_turn_agent() -> AgentCore:
    tc = make_mock_tool_call("get_weather", {"city": "Jakarta"})
    client = MockClient([make_mock_response(None, [tc]), make_mock_response("Weather in Jakarta: sunny")])
    return AgentCore(client=client, memory=ConversationMemory(), tools=make_tool_registry(), max_iterations=5)


def _server_config() -> Config:
    return Config.from_dict(
        {
            "agent": {"name": "bench-mem", "system_prompt": "h", "max_iterations": 3},
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "test",
                "base_url": "http://localhost:8080/v1",
            },
            "memory": {"backend": "in_memory"},
            "sandbox": {"backend": "restricted"},
            "server": {"auth_required": False},
        },
        validate=True,
    )


def test_mem_agentcore_turn(benchmark):
    """Peak memory for one tool-call turn (loop + 8-step pipeline allocation)."""

    def run_once():
        return _peak_bytes(lambda: asyncio.run(_tool_turn_agent().run("weather in Jakarta?")))

    peak = benchmark(run_once)
    benchmark.extra_info["peak_kb"] = round(peak / 1024, 1)
    assert peak < _CEIL_TURN


def test_mem_rag_index(benchmark, sample_chunks):
    """Peak memory to build a KeywordRetriever index over 100 chunks."""

    def run_once():
        return _peak_bytes(lambda: KeywordRetriever(chunks=sample_chunks))

    peak = benchmark(run_once)
    benchmark.extra_info["peak_kb"] = round(peak / 1024, 1)
    assert peak < _CEIL_RAG


def test_mem_server_app_boot(benchmark, tmp_path):
    """Peak memory to boot a server app (routes + pool + stores + middleware)."""

    def _factory():
        return MockClient([make_mock_response(content="hello")])

    def run_once():
        return _peak_bytes(
            lambda: create_app(
                _server_config(),
                client_factory=_factory,
                enable_cors=False,
                api_keys=None,
                workspace_root=str(tmp_path / "ws"),
            )
        )

    peak = benchmark(run_once)
    benchmark.extra_info["peak_kb"] = round(peak / 1024, 1)
    assert peak < _CEIL_BOOT
