"""Tier 1 CI-safe mechanics smoke for deep_research: fetch robustness.

Env-free + deterministic (DispatchingClient-style fake). Budget hard-stop, resume, and
session-message durability are already covered (see test_deep_research.py +
test_server_pool.py); this file covers the one genuinely missing mechanic: a FAILING fetch
provider must not crash the run. The ``web_fetch`` tool catches provider exceptions /
empty-result metadata and returns an error string (koboi/tools/builtin/web.py:234-240) -- these
tests prove the deep_research loop survives that path end-to-end.

The fake client drives one node through web_search -> web_fetch -> answer so the injected
``FailingFetchProvider`` is actually exercised. The fetch provider is injected by monkeypatching
``koboi.websearch.build_fetch_provider`` (the orchestrator imports it fresh each run).
"""

from __future__ import annotations

import json

import pytest

from koboi.events import OrchestrationCompleteEvent
from koboi.orchestration.dag_scheduler import DagScheduler
from koboi.orchestration.orchestrator import Orchestrator
from koboi.orchestration.router import KeywordRouter
from koboi.types import AgentResponse, ToolCall
from koboi.websearch.base import BaseFetchProvider
from koboi.websearch.types import FetchResult


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _StubFetch(BaseFetchProvider):
    """Inner fetch that always succeeds (the recovery path after fail_first)."""

    async def fetch(self, url: str, *, render: str = "auto", timeout: int = 15) -> FetchResult:
        return FetchResult(url=url, content="stub content", status=200)


class FailingFetchProvider(BaseFetchProvider):
    """Wrap an inner provider; fail the first ``fail_first`` calls, then delegate.

    mode="empty" returns a FetchResult with empty content + an error metadata entry (simulates a
    paywalled / JS-rendered page that yields no text). mode="error" raises (simulates a transient
    provider 5xx / network error).
    """

    def __init__(self, inner: BaseFetchProvider, *, mode: str, fail_first: int) -> None:
        self._inner = inner
        self._mode = mode
        self._fail_first = fail_first
        self.calls = 0

    async def fetch(self, url: str, *, render: str = "auto", timeout: int = 15) -> FetchResult:
        self.calls += 1
        if self.calls <= self._fail_first:
            if self._mode == "error":
                raise RuntimeError(f"injected transient failure #{self.calls}")
            return FetchResult(url=url, content="", status=0, metadata={"error": "empty (paywalled)"})
        return await self._inner.fetch(url, render=render, timeout=timeout)


class _FetchDrivingClient:
    """Fake LLM that drives a node: turn 1 web_search, turn 2 web_fetch, turn 3 answer.

    Dispatches on the number of tool-result messages seen so the injected fetch provider is
    actually called once per node.
    """

    model = "fake-model"
    provider = "fake"

    def __init__(self, node_answer: str = "Based on research, the answer is X.") -> None:
        self.node_answer = node_answer

    async def complete(self, messages, tools=None, response_format=None):
        text = " ".join(m.get("content", "") for m in messages)
        if "research planner" in text:
            return AgentResponse(
                content=json.dumps(
                    {
                        "needs_workflow": True,
                        "reason": "research",
                        "steps": [
                            {
                                "id": "research_topic",
                                "instruction": "Investigate the topic",
                                "depends_on": [],
                                "search_queries": ["topic overview"],
                            }
                        ],
                    }
                ),
                tool_calls=[],
            )
        if "evaluating how thoroughly" in text:
            return AgentResponse(
                content=json.dumps({"overall_score": 0.95, "coverage": {"x": 0.95}, "follow_up_queries": []}),
                tool_calls=[],
            )
        if "synthesizing a cited research report" in text:
            return AgentResponse(content=f"## Report\n{self.node_answer} [1].", tool_calls=[])
        # Node turn: count tool-result messages to decide search -> fetch -> answer.
        tool_results = [m for m in messages if m.get("role") == "tool"]
        if len(tool_results) == 0:
            return AgentResponse(
                content="",
                tool_calls=[ToolCall(id="tc_search", name="web_search", arguments=json.dumps({"query": "topic"}))],
            )
        if len(tool_results) == 1:
            return AgentResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc_fetch", name="web_fetch", arguments=json.dumps({"url": "https://example.com/x"}))
                ],
            )
        return AgentResponse(content=self.node_answer, tool_calls=[])


def _orch(client, tmp_path) -> Orchestrator:
    return Orchestrator(
        client=client,
        router=KeywordRouter(),
        research={"max_depth": 1, "coverage_threshold": 0.7, "max_searches": 5, "max_fetches": 8},
        dag_scheduler=DagScheduler(agents_map={}, deps={}, db_path=str(tmp_path / "m.db")),
    )


@pytest.fixture
def inject_fetch(monkeypatch):
    """Patch build_fetch_provider to return a FailingFetchProvider; yields it for assertion."""

    def _install(provider: FailingFetchProvider) -> FailingFetchProvider:
        import koboi.websearch as websearch

        monkeypatch.setattr(websearch, "build_fetch_provider", lambda conf: provider)
        return provider

    return _install


# ---------------------------------------------------------------------------
# M1 + M2: fetch failures must not crash the run
# ---------------------------------------------------------------------------


async def test_m1_empty_fetch_does_not_crash(tmp_path, inject_fetch):
    """M1: a fetch returning empty content (paywall/JS) -> run still completes with a report."""
    failing = inject_fetch(FailingFetchProvider(_StubFetch(), mode="empty", fail_first=3))
    orch = _orch(_FetchDrivingClient(), tmp_path)
    events = [e async for e in orch._run_deep_research("Research the topic.")]
    complete = [e for e in events if isinstance(e, OrchestrationCompleteEvent)]
    assert complete, "run did not complete when fetch returned empty content"
    assert complete[0].final_answer  # a report was still produced
    assert failing.calls >= 1  # the failing fetch path was actually exercised


async def test_m2_raising_fetch_does_not_crash(tmp_path, inject_fetch):
    """M2: a fetch provider that raises -> run still completes (web_fetch catches the exception)."""
    failing = inject_fetch(FailingFetchProvider(_StubFetch(), mode="error", fail_first=3))
    orch = _orch(_FetchDrivingClient(), tmp_path)
    events = [e async for e in orch._run_deep_research("Research the topic.")]
    complete = [e for e in events if isinstance(e, OrchestrationCompleteEvent)]
    assert complete, "run crashed when the fetch provider raised"
    assert complete[0].final_answer
    assert failing.calls >= 1
