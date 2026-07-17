"""tests/test_self_healing_p0.py -- self-healing P0 (behavior-level recovery seeds).

P0-A: Orchestrator._run_single marks node crashes as failed=True (unblocks the
      otherwise-dead dynamic re-plan loop). Tested WITHOUT monkeypatching
      _run_single -- a real raising agent is placed in _agents_map so the
      production path actually runs (existing replan tests force failed=True by
      monkeypatching, which masked the bug).
P0-B: orchestration.execution.max_replans is a validated, opt-in (default 0) knob.
P0-C: a complete-but-empty response is re-asked once with a nudge (default-ON),
      then falls back to the legacy success=True behavior once the budget is spent.
P0-D: tool errors surface a structured errored/error_kind signal on
      ToolPipelineResult (no fragile string-matching), with an actionable,
      "Error:"-prefix-preserving message; ToolRegistry.execute() keeps its -> str
      contract via execute_outcome().
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from koboi.config_models import ExecutionConfig, OrchestrationConfig
from koboi.events import CompleteEvent
from koboi.loop import AgentCore
from koboi.loop_pipeline import ToolExecutionPipeline
from koboi.memory import ConversationMemory
from koboi.orchestration.orchestrator import Orchestrator
from koboi.tools.registry import ToolRegistry
from koboi.types import RunResult, ToolCall
from tests.conftest import MockClient, make_mock_response, make_tool_registry


# --------------------------------------------------------------------------- P0-A


class _RaisingAgent:
    """Stand-in agent whose run() raises -- exercises the REAL _run_single path."""

    def __init__(self, exc: Exception):
        self.exc = exc
        self.memory = ConversationMemory()

    async def run(self, query):
        raise self.exc


class _FakeAgent:
    """Stand-in agent returning a canned RunResult."""

    def __init__(self, result: RunResult):
        self.result = result
        self.memory = ConversationMemory()

    async def run(self, query):
        return self.result


def _orchestrator() -> Orchestrator:
    class _NoRouter:
        async def route(self, q):  # pragma: no cover - not reached in _run_single tests
            raise AssertionError("routing not expected")

    return Orchestrator(client=MockClient([]), router=_NoRouter(), agents_map={}, default_mode="sequential")


class TestRunSingleFailedFlag:
    async def test_crash_marks_failed_true(self):
        orch = _orchestrator()
        orch._agents_map["boom"] = _RaisingAgent(RuntimeError("kaboom"))
        res = await orch._run_single("boom", "do something")
        assert res.failed is True
        assert res.answer.startswith("Error:")

    async def test_success_false_result_marks_failed(self):
        orch = _orchestrator()
        orch._agents_map["meh"] = _FakeAgent(RunResult(content="x", iterations_used=1, success=False))
        res = await orch._run_single("meh", "q")
        assert res.failed is True

    async def test_normal_result_not_failed(self):
        orch = _orchestrator()
        orch._agents_map["ok"] = _FakeAgent(RunResult(content="done", iterations_used=1))
        res = await orch._run_single("ok", "q")
        assert res.failed is False


# --------------------------------------------------------------------------- P0-B


class TestMaxReplansConfig:
    def test_default_is_zero_opt_in(self):
        assert OrchestrationConfig().execution.max_replans == 0

    def test_execution_config_defaults(self):
        ec = ExecutionConfig()
        assert ec.max_replans == 0
        assert ec.max_revisions == 2
        assert ec.use_revision is False
        assert ec.full_graph is False

    def test_validates_max_replans_from_dict(self):
        oc = OrchestrationConfig(execution={"max_replans": 3, "mode": "dynamic"})
        assert oc.execution.max_replans == 3
        assert oc.execution.mode == "dynamic"

    def test_allows_unknown_execution_keys(self):
        # extra="allow": arbitrary execution keys (research caps, custom, ...) pass
        oc = OrchestrationConfig(execution={"mode": "dag", "custom_key": {"nested": 1}})
        assert oc.execution.mode == "dag"

    def test_rejects_non_int_max_replans(self):
        with pytest.raises(ValidationError):
            OrchestrationConfig(execution={"max_replans": "soon"})


# --------------------------------------------------------------------------- P0-C


def _messages_contain(agent: AgentCore, needle: str) -> bool:
    return any(needle in (m.get("content") or "") for m in agent.memory.get_messages())


class TestEmptyResponseReask:
    async def test_reasks_once_then_answers(self):
        client = MockClient([make_mock_response(None), make_mock_response("Hello!")])
        agent = AgentCore(client=client, memory=ConversationMemory(), tools=ToolRegistry(), max_iterations=5)
        result = await agent.run("hi")
        assert result.content == "Hello!"
        assert result.success is True
        assert result.metadata["empty_response_reasked"] == 1
        assert client.call_count == 2  # one empty + one recovered
        assert _messages_contain(agent, "previous response was empty")

    async def test_falls_back_to_success_when_always_empty(self):
        client = MockClient([make_mock_response(None)] * 5)
        agent = AgentCore(client=client, memory=ConversationMemory(), tools=ToolRegistry(), max_iterations=5)
        result = await agent.run("hi")
        # back-compat: still success (legacy behavior) after the single re-ask
        assert result.success is True
        assert result.metadata["empty_response_reasked"] == 1

    async def test_disabled_when_limit_zero(self):
        client = MockClient([make_mock_response(None)])
        agent = AgentCore(
            client=client,
            memory=ConversationMemory(),
            tools=ToolRegistry(),
            max_iterations=5,
            empty_response_reask_limit=0,
        )
        result = await agent.run("hi")
        assert result.success is True
        assert result.metadata["empty_response_reasked"] == 0
        assert client.call_count == 1  # no re-ask

    async def test_stream_reasks_on_empty(self):
        client = MockClient([make_mock_response(None), make_mock_response("Hi")])
        agent = AgentCore(client=client, memory=ConversationMemory(), tools=ToolRegistry(), max_iterations=5)
        events = [e async for e in agent.run_stream("hi")]
        completes = [e for e in events if isinstance(e, CompleteEvent)]
        assert completes and completes[-1].content == "Hi"
        assert agent._empty_response_reasked == 1

    async def test_empty_on_last_iteration_falls_back_to_success(self):
        # Regression (review): an empty response on the FINAL iteration must NOT
        # raise AgentMaxIterationsError -- it falls back to legacy success=True
        # (no room left to re-ask).
        client = MockClient([make_mock_response(None)])
        agent = AgentCore(client=client, memory=ConversationMemory(), tools=ToolRegistry(), max_iterations=1)
        result = await agent.run("hi")
        assert result.success is True
        assert result.metadata["empty_response_reasked"] == 0


# --------------------------------------------------------------------------- P0-D


class TestToolExecOutcome:
    async def test_success_outcome(self):
        reg = make_tool_registry()
        out = await reg.execute_outcome("get_weather", '{"city": "Jakarta"}')
        assert out.errored is False
        assert out.error_kind is None
        assert "Jakarta" in out.content

    async def test_unknown_tool_outcome(self):
        reg = make_tool_registry()
        out = await reg.execute_outcome("nope", "{}")
        assert out.errored is True
        assert out.error_kind == "tool_not_found"
        assert out.content.startswith("Error:")

    async def test_bad_json_outcome(self):
        reg = make_tool_registry()
        out = await reg.execute_outcome("get_weather", "not json")
        assert out.errored is True
        assert out.error_kind == "invalid_args"

    async def test_execution_error_outcome(self):
        reg = ToolRegistry()

        def _raise():
            raise RuntimeError("boom")

        reg.register("boom", "boom tool", {"type": "object", "properties": {}}, _raise)
        out = await reg.execute_outcome("boom", "{}")
        assert out.errored is True
        assert out.error_kind == "execution_error"
        assert "Error executing" in out.content

    async def test_execute_still_returns_str(self):
        reg = make_tool_registry()
        s = await reg.execute("get_weather", '{"city": "X"}')
        out = await reg.execute_outcome("get_weather", '{"city": "X"}')
        assert isinstance(s, str)
        assert s == out.content


class TestPipelineErrorSignal:
    @staticmethod
    def _raising_registry() -> ToolRegistry:
        reg = ToolRegistry()

        def _raise():
            raise ValueError("nope")

        reg.register("boom", "boom tool", {"type": "object", "properties": {}}, _raise, idempotent=False)
        return reg

    async def test_error_sets_structured_signal(self):
        pipe = ToolExecutionPipeline(tools=self._raising_registry(), memory=ConversationMemory())
        res = await pipe.execute_tool_call(ToolCall(id="t1", name="boom", arguments="{}"), iteration=0)
        assert res.errored is True
        assert res.error_kind == "execution_error"
        assert res.skipped is False
        assert res.idempotent is False
        # prefix preserved + actionable hint + side-effect warning
        assert res.result.startswith("Error executing")
        assert "side effects" in res.result

    async def test_success_no_signal(self):
        pipe = ToolExecutionPipeline(tools=make_tool_registry(), memory=ConversationMemory())
        res = await pipe.execute_tool_call(
            ToolCall(id="t1", name="get_weather", arguments='{"city": "X"}'), iteration=0
        )
        assert res.errored is False
        assert res.error_kind is None
        assert res.skipped is False
        assert "X" in res.result
