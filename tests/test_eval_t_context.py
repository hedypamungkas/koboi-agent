"""Tests for koboi.eval.t.context -- TestContext driving and assertions."""

from __future__ import annotations

import pytest

from koboi.eval.t.assertions import Contains, Equals, Severity
from koboi.eval.t.context import TestContext
from koboi.eval.scorers.base import BaseScorer
from koboi.exceptions import AgentMaxIterationsError
from koboi.facade import KoboiAgent
from koboi.loop import AgentCore
from koboi.memory import ConversationMemory
from koboi.tools.registry import ToolRegistry
from koboi.types import EvalScore
from tests.conftest import MockClient, make_mock_response, make_mock_tool_call, make_tool_registry


def _mock_agent(responses, *, tools=None, max_iterations=5):
    client = MockClient(responses)
    core = AgentCore(
        client=client,
        memory=ConversationMemory(),
        tools=tools or make_tool_registry(),
        max_iterations=max_iterations,
    )
    return KoboiAgent(core=core)


class _StubScorer(BaseScorer):
    def __init__(self, value, reason="stub"):
        self._value = value
        self._reason = reason

    async def score(self, case, output, context):
        return EvalScore("stub", self._value, self._reason)


class _AlwaysToolsClient(MockClient):
    """Always requests a tool call so the loop hits max_iterations."""

    async def complete(self, messages, tools=None):
        self.call_count += 1
        return make_mock_response(None, [make_mock_tool_call("get_weather", {"city": "Jakarta"})])


class TestDriving:
    async def test_send_records_turn_and_reply(self):
        agent = _mock_agent([make_mock_response("Hello world")])
        ctx = TestContext(agent)
        result = await ctx.send("hi")
        assert result.content == "Hello world"
        assert ctx.reply == "Hello world"
        assert ctx.output == "Hello world"
        assert len(ctx.turns) == 1

    async def test_send_multiple_turns_accumulates(self):
        agent = _mock_agent([make_mock_response("first"), make_mock_response("second")])
        ctx = TestContext(agent)
        await ctx.send("q1")
        await ctx.send("q2")
        assert ctx.reply == "second"
        assert len(ctx.turns) == 2

    async def test_reply_empty_before_send(self):
        ctx = TestContext(_mock_agent([]))
        assert ctx.reply == ""

    def test_last_before_send_raises(self):
        ctx = TestContext(_mock_agent([]))
        with pytest.raises(RuntimeError):
            _ = ctx.last

    async def test_send_catches_agent_error(self):
        agent = _mock_agent([], tools=ToolRegistry(), max_iterations=2)
        agent.core.client = _AlwaysToolsClient([])  # never completes -> max iterations
        ctx = TestContext(agent)
        result = await ctx.send("loop me")
        assert result.success is False
        assert isinstance(result.error, AgentMaxIterationsError)
        assert ctx.reply == ""
        assert "send:error" in [assertion.name for assertion in ctx.collect()]

    async def test_messages_trace(self):
        agent = _mock_agent([make_mock_response("hi")])
        ctx = TestContext(agent)
        await ctx.send("hello")
        roles = [message["role"] for message in ctx.messages]
        assert "user" in roles and "assistant" in roles


class TestToolAssertions:
    async def test_called_tool_pass_and_fail(self):
        call = make_mock_tool_call("get_weather", {"city": "Jakarta"})
        agent = _mock_agent([make_mock_response(None, [call]), make_mock_response("Sunny")])
        ctx = TestContext(agent)
        await ctx.send("weather?")
        ctx.calledTool("get_weather")
        ctx.calledTool("missing")
        outcomes = {a.name: a.outcome() for a in ctx.collect()}
        assert outcomes["calledTool:get_weather"].passed is True
        assert outcomes["calledTool:missing"].passed is False

    async def test_called_tool_with_subset_match(self):
        call = make_mock_tool_call("get_weather", {"city": "Jakarta", "unit": "C"})
        agent = _mock_agent([make_mock_response(None, [call]), make_mock_response("Sunny")])
        ctx = TestContext(agent)
        await ctx.send("weather?")
        ctx.calledToolWith("get_weather", {"city": "Jakarta"})  # subset -> pass
        ctx.calledToolWith("get_weather", {"city": "Bandung"})  # mismatch -> fail
        outcomes = [assertion.outcome() for assertion in ctx.collect()]
        assert outcomes[0].passed is True
        assert outcomes[1].passed is False

    async def test_used_no_tools_pass(self):
        ctx = TestContext(_mock_agent([make_mock_response("just text")]))
        await ctx.send("hi")
        ctx.usedNoTools()
        assert ctx.collect()[0].outcome().passed is True

    async def test_used_no_tools_fail(self):
        call = make_mock_tool_call("calculate", {"expression": "1+1"})
        agent = _mock_agent([make_mock_response(None, [call]), make_mock_response("2")])
        ctx = TestContext(agent)
        await ctx.send("calc")
        ctx.usedNoTools()
        assert ctx.collect()[0].outcome().passed is False

    async def test_tool_assertions_default_to_gate(self):
        ctx = TestContext(_mock_agent([make_mock_response("ok")]))
        await ctx.send("q")
        ctx.calledTool("x")
        assert ctx.collect()[0].severity is Severity.GATE


class TestCompleted:
    async def test_completed_pass(self):
        ctx = TestContext(_mock_agent([make_mock_response("done")]))
        await ctx.send("hi")
        ctx.completed()
        assert ctx.collect()[0].outcome().passed is True

    def test_completed_no_turns_fails(self):
        ctx = TestContext(_mock_agent([]))
        ctx.completed()
        assert ctx.collect()[0].outcome().passed is False


class TestCheck:
    async def test_check_with_matcher(self):
        ctx = TestContext(_mock_agent([make_mock_response("The answer is 4")]))
        await ctx.send("q")
        ctx.check(ctx.reply, Contains("4"))
        ctx.check(ctx.reply, Equals("nope"))
        outcomes = ctx.collect()
        assert outcomes[0].outcome().passed is True
        assert outcomes[1].outcome().passed is False
        assert outcomes[0].severity is Severity.SOFT  # check defaults to soft

    def test_check_truthy_without_matcher(self):
        ctx = TestContext(_mock_agent([]))
        ctx.check(True)
        ctx.check(0)
        outcomes = ctx.collect()
        assert outcomes[0].outcome().passed is True
        assert outcomes[1].outcome().passed is False

    def test_check_severity_override(self):
        ctx = TestContext(_mock_agent([]))
        ctx.check(False, severity=Severity.GATE)
        outcome = ctx.collect()[0].outcome()
        assert outcome.passed is False
        assert outcome.value == 0.0  # gate failure


class TestJudge:
    async def test_judge_instance_scorer_pass(self):
        ctx = TestContext(_mock_agent([make_mock_response("the answer")]))
        await ctx.send("q")
        await ctx.judge(_StubScorer(0.9, "great"))
        outcome = ctx.collect()[0].outcome()
        assert outcome.passed is True
        assert outcome.value == 0.9

    async def test_judge_below_min_score_fails(self):
        ctx = TestContext(_mock_agent([make_mock_response("the answer")]))
        await ctx.send("q")
        await ctx.judge(_StubScorer(0.3), min_score=0.7)
        outcome = ctx.collect()[0].outcome()
        assert outcome.passed is False
        assert outcome.value == 0.3

    async def test_judge_registry_keyword_presence(self):
        ctx = TestContext(_mock_agent([make_mock_response("contains the keyword yes")]))
        await ctx.send("q")
        await ctx.judge("keyword_presence", expected=["keyword"])
        outcome = ctx.collect()[0].outcome()
        assert outcome.passed is True
        assert outcome.value == 1.0

    async def test_judge_missing_dependency_is_soft_skip(self):
        # llm_judge requires a client kwarg -> construction fails -> fail-soft record
        ctx = TestContext(_mock_agent([make_mock_response("x")]))
        await ctx.send("q")
        await ctx.judge("llm_judge")
        outcome = ctx.collect()[0].outcome()
        assert outcome.passed is False
        assert "unavailable" in outcome.reason

    async def test_judge_unknown_name_is_soft_skip(self):
        ctx = TestContext(_mock_agent([make_mock_response("x")]))
        await ctx.send("q")
        await ctx.judge("does_not_exist")
        assert ctx.collect()[0].outcome().passed is False


class TestTokenUsage:
    async def test_total_token_usage_sums_turns(self):
        ctx = TestContext(_mock_agent([make_mock_response("a"), make_mock_response("b")]))
        await ctx.send("q1")
        await ctx.send("q2")
        usage = ctx.total_token_usage()
        # make_mock_response uses prompt_tokens=10, completion_tokens=20 each
        assert usage.prompt_tokens == 20
        assert usage.completion_tokens == 40
        assert usage.total_tokens == 60
