"""Agent-loop + tool-pipeline performance benchmarks.

Measures the framework's hot path -- ``AgentCore.run`` (a full turn) and the
8-step ``ToolExecutionPipeline`` (one tool-call round-trip) -- with a canned
``MockClient`` (no network, no real LLM). These are higher-variance than the
pure-CPU micro-benmarks in bench_core/hooks/rag/tui, so they are REPORTED in
the CI summary + artifact but gated only loosely (the relative-compare layer,
not the absolute NFR threshold, is the right gate here -- see
docs/performance-benchmarking.md).

Async-under-benchmark convention (mirrors bench_core.py:135-142): wrap
``asyncio.run(coro)`` in a SYNC closure passed to ``benchmark(...)``; do NOT
mark the bench function async (the ``benchmark`` fixture is synchronous).
"""

from __future__ import annotations

import asyncio

import pytest

from koboi.hooks.chain import Hook, HookChain, HookContext, HookEvent
from koboi.loop import AgentCore
from koboi.loop_pipeline import ToolExecutionPipeline
from koboi.memory import ConversationMemory
from koboi.tools.registry import ToolRegistry
from tests.conftest import MockClient, make_mock_response, make_mock_tool_call, make_tool_registry


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def _plain_agent() -> AgentCore:
    """One-iteration agent: a single text completion, no tools."""
    client = MockClient([make_mock_response("Hello!")])
    return AgentCore(client=client, memory=ConversationMemory(), tools=ToolRegistry(), max_iterations=5)


def _tool_turn_agent() -> AgentCore:
    """Two-iteration agent: LLM requests a tool, then completes."""
    tc = make_mock_tool_call("get_weather", {"city": "Jakarta"})
    client = MockClient(
        [
            make_mock_response(None, [tc]),  # iter 1: tool call
            make_mock_response("Weather in Jakarta: sunny"),  # iter 2: completion
        ]
    )
    return AgentCore(client=client, memory=ConversationMemory(), tools=make_tool_registry(), max_iterations=5)


def _n_turn_agent(n: int) -> AgentCore:
    """Agent scripted for n tool-call turns (2n responses)."""
    pair = [
        make_mock_response(None, [make_mock_tool_call("get_weather", {"city": "x"})]),
        make_mock_response("done"),
    ]
    client = MockClient(pair * n)
    return AgentCore(
        client=client,
        memory=ConversationMemory(),
        tools=make_tool_registry(),
        max_iterations=n * 2 + 1,
    )


class _NullHook(Hook):
    """No-op hook subscribed to the 4 per-iteration hot events."""

    def handles(self):
        return [HookEvent.PRE_LLM_CALL, HookEvent.POST_LLM_CALL, HookEvent.PRE_TOOL_USE, HookEvent.POST_TOOL_USE]

    async def execute(self, ctx: HookContext) -> HookContext:
        return ctx


def _tool_turn_agent_with_hooks(n_hooks: int) -> AgentCore:
    tc = make_mock_tool_call("get_weather", {"city": "Jakarta"})
    client = MockClient([make_mock_response(None, [tc]), make_mock_response("done")])
    chain = HookChain([_NullHook() for _ in range(n_hooks)]) if n_hooks else None
    return AgentCore(
        client=client,
        memory=ConversationMemory(),
        tools=make_tool_registry(),
        max_iterations=5,
        hook_chain=chain,
    )


# ---------------------------------------------------------------------------
# benchmarks
# ---------------------------------------------------------------------------


def test_loop_plain_turn(benchmark):
    """One plain completion turn (1 iteration, no tools)."""

    def run_once():
        return asyncio.run(_plain_agent().run("Hi"))

    result = benchmark(run_once)
    assert result.content == "Hello!"


def test_loop_tool_call_turn(benchmark):
    """One tool-call turn (LLM -> tool -> LLM complete; 2 iterations)."""

    def run_once():
        return asyncio.run(_tool_turn_agent().run("weather in Jakarta?"))

    result = benchmark(run_once)
    assert result.success and result.iterations_used == 2


def test_loop_n_turns_10(benchmark):
    """Throughput: 10 sequential tool-call turns on one persistent agent.

    A single run() stops at the first completion, so N turns = N separate run()
    calls sharing one agent (MockClient advances its canned-response index
    across calls; 2 responses consumed per turn).
    """

    def run_once():
        agent = _n_turn_agent(10)  # 20 canned responses -> 2 per turn
        return [asyncio.run(agent.run("go")) for _ in range(10)]

    results = benchmark(run_once)
    assert len(results) == 10 and all(r.success for r in results)


def test_pipeline_one_round_trip(benchmark):
    """The 8-step ToolExecutionPipeline for one tool call (isolated from the loop)."""
    registry = make_tool_registry()
    tc = make_mock_tool_call("get_weather", {"city": "Jakarta"})

    # Force fresh state per round so memory accumulation doesn't skew timing.
    def run_once():
        pipe = ToolExecutionPipeline(tools=registry, memory=ConversationMemory())
        return asyncio.run(pipe.execute_tool_call(tc, iteration=0))

    result = benchmark(run_once)
    assert result.tool_name == "get_weather" and not result.skipped


@pytest.mark.parametrize("n_hooks", [0, 5, 10])
def test_loop_turn_with_hooks(benchmark, n_hooks):
    """Per-turn loop cost as hook-chain depth grows (measures in-loop hook overhead)."""

    def run_once():
        return asyncio.run(_tool_turn_agent_with_hooks(n_hooks).run("weather in Jakarta?"))

    result = benchmark(run_once)
    assert result.success and result.iterations_used == 2
