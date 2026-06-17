"""Tests for Orchestrator streaming, revision, and uncovered paths."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from koboi.events import (
    AgentDispatchEvent, AgentResultEvent, OrchestrationCompleteEvent,
    RoutingDecisionEvent, TextDeltaEvent, CompleteEvent,
)
from koboi.orchestration.orchestrator import Orchestrator, QualityEvaluator
from koboi.types import AgentResult, AgentResponse, RoutingDecision


def _make_decision(agents=None):
    return RoutingDecision(
        query="test",
        agents=agents or ["agent1"],
        confidence=0.9,
        method="keyword",
        reasoning="test",
    )


class MockRouter:
    def __init__(self, decision=None):
        self._decision = decision or _make_decision()

    async def route(self, query):
        return self._decision


class MockAgent:
    def __init__(self, answer="test answer", fail=False):
        self._answer = answer
        self._fail = fail
        self.memory = MagicMock()
        self.memory.get_messages.return_value = []

    async def run(self, query):
        if self._fail:
            raise RuntimeError("agent failed")
        result = MagicMock()
        result.content = self._answer
        return result


class TestQualityEvaluator:
    async def test_evaluate_success(self):
        client = MagicMock()
        client.complete = AsyncMock(return_value=AgentResponse(
            content='{"score": 0.8, "feedback": "good", "needs_revision": false}',
        ))
        evaluator = QualityEvaluator(client, threshold=0.6)
        score, feedback, needs = await evaluator.evaluate("q", "a")
        assert score == 0.8
        assert feedback == "good"
        assert needs is False

    async def test_evaluate_needs_revision(self):
        client = MagicMock()
        client.complete = AsyncMock(return_value=AgentResponse(
            content='{"score": 0.3, "feedback": "too short", "needs_revision": true}',
        ))
        evaluator = QualityEvaluator(client, threshold=0.6)
        score, feedback, needs = await evaluator.evaluate("q", "a")
        assert score == 0.3
        assert needs is True

    async def test_evaluate_json_error(self):
        client = MagicMock()
        client.complete = AsyncMock(return_value=AgentResponse(content="not json"))
        evaluator = QualityEvaluator(client)
        score, feedback, needs = await evaluator.evaluate("q", "a")
        assert score == 0.5
        assert needs is True

    async def test_evaluate_exception(self):
        client = MagicMock()
        client.complete = AsyncMock(side_effect=Exception("network error"))
        evaluator = QualityEvaluator(client)
        score, feedback, needs = await evaluator.evaluate("q", "a")
        assert score == 0.5


class TestOrchestratorRun:
    async def test_sequential_execution(self):
        router = MockRouter()
        client = MagicMock()
        agent = MockAgent(answer="answer1")
        orch = Orchestrator(
            client=client, router=router,
            agents_map={"agent1": agent},
        )
        result = await orch.run("test query", mode="sequential")
        assert result.final_answer == "answer1"
        assert len(result.agent_results) == 1
        assert result.execution_mode == "sequential"

    async def test_parallel_execution(self):
        router = MockRouter(_make_decision(["a1", "a2"]))
        client = MagicMock()
        agents = {"a1": MockAgent("ans1"), "a2": MockAgent("ans2")}
        orch = Orchestrator(client=client, router=router, agents_map=agents)
        result = await orch.run("q", mode="parallel")
        assert len(result.agent_results) == 2

    async def test_parallel_with_failure(self):
        router = MockRouter(_make_decision(["a1", "a2"]))
        client = MagicMock()
        agents = {"a1": MockAgent("ok"), "a2": MockAgent(fail=True)}
        orch = Orchestrator(client=client, router=router, agents_map=agents)
        result = await orch.run("q", mode="parallel")
        assert len(result.agent_results) == 2

    async def test_single_agent_combine(self):
        router = MockRouter()
        client = MagicMock()
        orch = Orchestrator(client=client, router=router, agents_map={"agent1": MockAgent("ans")})
        result = await orch.run("q")
        assert result.final_answer == "ans"

    async def test_multi_agent_synthesis(self):
        router = MockRouter(_make_decision(["a1", "a2"]))
        client = MagicMock()
        client.complete = AsyncMock(return_value=AgentResponse(content="synthesized"))

        async def _stream(*args, **kwargs):
            yield TextDeltaEvent(content="synthesized")

        client.complete_stream = _stream
        agents = {"a1": MockAgent("ans1"), "a2": MockAgent("ans2")}
        orch = Orchestrator(client=client, router=router, agents_map=agents)
        result = await orch.run("q")
        assert "synthesized" in result.final_answer or "synthesized" == result.final_answer

    async def test_multi_agent_synthesis_failure(self):
        router = MockRouter(_make_decision(["a1", "a2"]))
        client = MagicMock()
        client.complete = AsyncMock(side_effect=Exception("fail"))

        async def _stream(*args, **kwargs):
            raise Exception("fail")
            yield  # make it a generator

        client.complete_stream = _stream
        agents = {"a1": MockAgent("ans1"), "a2": MockAgent("ans2")}
        orch = Orchestrator(client=client, router=router, agents_map=agents)
        result = await orch.run("q")
        assert "Answer from" in result.final_answer

    async def test_with_logger(self):
        router = MockRouter()
        client = MagicMock()
        logger = MagicMock()
        orch = Orchestrator(
            client=client, router=router, logger=logger,
            agents_map={"agent1": MockAgent("ans")},
        )
        await orch.run("q")
        logger.log_routing.assert_called_once()
        logger.log_orchestration_summary.assert_called_once()

    async def test_revision_mode(self):
        router = MockRouter()
        client = MagicMock()
        client.complete = AsyncMock(return_value=AgentResponse(
            content='{"score": 0.9, "feedback": "good", "needs_revision": false}',
        ))
        evaluator = QualityEvaluator(client, threshold=0.6)
        orch = Orchestrator(
            client=client, router=router, evaluator=evaluator,
            use_revision=True, agents_map={"agent1": MockAgent("ans")},
        )
        result = await orch.run("q")
        assert result.execution_mode == "sequential+revision"


class TestOrchestratorStream:
    async def test_stream_sequential(self):
        router = MockRouter()
        client = MagicMock()
        orch = Orchestrator(
            client=client, router=router,
            agents_map={"agent1": MockAgent("ans")},
        )
        events = []
        async for event in orch.run_stream("q", mode="sequential"):
            events.append(event)
        assert any(isinstance(e, RoutingDecisionEvent) for e in events)
        assert any(isinstance(e, AgentDispatchEvent) for e in events)
        assert any(isinstance(e, AgentResultEvent) for e in events)
        assert any(isinstance(e, OrchestrationCompleteEvent) for e in events)

    async def test_stream_parallel(self):
        router = MockRouter(_make_decision(["a1", "a2"]))
        client = MagicMock()
        agents = {"a1": MockAgent("ans1"), "a2": MockAgent("ans2")}
        orch = Orchestrator(client=client, router=router, agents_map=agents)
        events = []
        async for event in orch.run_stream("q", mode="parallel"):
            events.append(event)
        dispatch_events = [e for e in events if isinstance(e, AgentDispatchEvent)]
        assert len(dispatch_events) == 2

    async def test_make_agent_logger(self):
        router = MockRouter()
        client = MagicMock()
        logger = MagicMock()
        logger.session_id = "main"
        logger.log_dir = "/tmp"
        orch = Orchestrator(client=client, router=router, logger=logger)
        result = orch._make_agent_logger("agent1")
        assert result is not None

    async def test_make_agent_logger_no_logger(self):
        router = MockRouter()
        client = MagicMock()
        orch = Orchestrator(client=client, router=router, logger=None)
        assert orch._make_agent_logger("agent1") is None
