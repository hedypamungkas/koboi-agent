"""koboi/eval/t/context.py -- branch coverage for TestContext (record-then-evaluate)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from koboi.exceptions import AgentError, AgentGuardrailError
from koboi.eval.t.assertions import Severity
from koboi.eval.t.context import TestContext, _parse_args
from koboi.types import RunResult, TokenUsage, ToolCall


def _eval_all(t: TestContext):
    return [a.evaluate() for a in t.collect()]


def _agent(run_results=None, run_exc=None) -> MagicMock:
    agent = MagicMock()
    if run_exc is not None:
        agent.run = AsyncMock(side_effect=run_exc)
    else:
        agent.run = AsyncMock(side_effect=list(run_results or []))
    return agent


class TestParseArgs:
    def test_none(self):
        assert _parse_args(None) == {}

    def test_empty(self):
        assert _parse_args("") == {}

    def test_invalid_json(self):
        assert _parse_args("{bad") == {}

    def test_non_dict_json(self):
        assert _parse_args("[1,2]") == {}

    def test_valid(self):
        assert _parse_args('{"a": 1}') == {"a": 1}


class TestSend:
    async def test_send_success_records_turn(self):
        agent = _agent(run_results=[RunResult(content="hi", iterations_used=1, success=True)])
        t = TestContext(agent)
        res = await t.send("hello")
        assert res.content == "hi"
        assert t.reply == "hi"
        assert t.turns and len(t.turns) == 1

    async def test_send_list_message(self):
        agent = _agent(run_results=[RunResult(content="x", iterations_used=1, success=True)])
        t = TestContext(agent)
        await t.send(["a", "b"])
        assert "a" in t._sent[0]

    async def test_send_agent_error_records_soft(self):
        agent = _agent(run_exc=AgentError("boom"))
        t = TestContext(agent)
        res = await t.send("x")
        assert res.success is False
        outs = _eval_all(t)
        assert any("send raised" in o.reason for o in outs)

    async def test_send_guardrail_error_stamps_outcomes(self):
        agent = _agent(run_exc=AgentGuardrailError("blocked", "input"))
        t = TestContext(agent)
        res = await t.send("x")
        assert res.metadata["guardrail_outcomes"][0]["direction"] == "input"


class TestIntrospection:
    async def test_reply_empty_before_send(self):
        t = TestContext(_agent())
        assert t.reply == ""
        assert t.output == ""
        assert t.total_token_usage().total_tokens == 0

    def test_last_before_send_raises(self):
        t = TestContext(_agent())
        with pytest.raises(RuntimeError):
            _ = t.last

    async def test_messages_and_tokens(self):
        agent = _agent(
            run_results=[
                RunResult(
                    content="a",
                    iterations_used=1,
                    success=True,
                    token_usage=TokenUsage(prompt_tokens=5, completion_tokens=7),
                ),
            ]
        )
        agent.core.memory.get_messages.return_value = [{"role": "user", "content": "x"}]
        t = TestContext(agent)
        await t.send("hi")
        assert t.messages == [{"role": "user", "content": "x"}]
        assert t.total_token_usage().total_tokens == 12

    async def test_messages_no_core(self):
        agent = MagicMock()
        agent.core = None
        t = TestContext(agent)
        assert t.messages == []


class TestLiveReady:
    async def test_scripted_client_not_ready(self):
        from koboi.eval.t.mock import ScriptedClient

        agent = MagicMock()
        agent.core.client = ScriptedClient([])
        t = TestContext(agent)
        assert t.live_ready() is False

    async def test_extra_missing_not_ready(self):
        agent = MagicMock()
        agent.core.client = object()  # not a ScriptedClient
        t = TestContext(agent)
        # "ragas" absent in CI-faithful -> not ready
        assert t.live_ready(extra="ragas") is False

    async def test_require_live_records_skip(self, monkeypatch):
        agent = MagicMock()
        agent.core.client = object()
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        t = TestContext(agent)
        assert t.require_live() is False
        outs = _eval_all(t)
        assert outs and outs[0].value == 1.0  # soft pass skip


class TestToolAssertions:
    async def test_called_tool_and_with(self):
        tc = ToolCall(id="1", name="get_weather", arguments='{"city": "Jakarta"}')
        res = RunResult(content="x", iterations_used=1, success=True, tool_calls_made=[tc])
        agent = _agent(run_results=[res])
        t = TestContext(agent)
        await t.send("q")
        t.calledTool("get_weather")
        t.calledTool("get_weather", severity=Severity.SOFT)
        t.calledToolWith("get_weather", {"city": "Jakarta"})
        t.calledToolWith("get_weather", {"city": "Bali"})  # no match
        t.usedNoTools()  # will fail (tools were used)
        outs = _eval_all(t)
        reasons = " | ".join(o.reason for o in outs)
        assert "1 call(s)" in reasons
        assert "1 match(es)" in reasons
        assert "0 match(es)" in reasons

    async def test_tool_was_blocked(self):
        res = RunResult(
            content="x",
            iterations_used=1,
            success=True,
            pipeline_outcomes=[{"tool_name": "rm", "skipped": True, "skip_reason": "mode_blocked"}],
        )
        agent = _agent(run_results=[res])
        t = TestContext(agent)
        await t.send("q")
        t.toolWasBlocked("rm")
        outs = _eval_all(t)
        assert any("1 block(s)" in o.reason for o in outs)


class TestRagAssertions:
    async def test_retrieved_chunk(self):
        res = RunResult(
            content="x",
            iterations_used=1,
            success=True,
            metadata={"rag_results": [{"content": "annual leave is 12 days"}]},
        )
        agent = _agent(run_results=[res])
        t = TestContext(agent)
        await t.send("q")
        t.retrievedChunk("12 days")
        t.retrievedChunk("nope")  # miss
        outs = _eval_all(t)
        assert any("1 match(es)" in o.reason for o in outs)

    async def test_ranking_metric_variants(self):
        res = RunResult(
            content="x",
            iterations_used=1,
            success=True,
            metadata={"rag_results": [{"content": "alpha"}, {"content": "beta"}]},
        )
        agent = _agent(run_results=[res])
        t = TestContext(agent)
        await t.send("q")
        t.rankingMetric("alpha", k=10, metric="recall", min_score=1.0)
        t.rankingMetric("missing", k=10, metric="mrr", min_score=0.5)
        outs = _eval_all(t)
        assert any("recall@10" in o.reason for o in outs)
        assert any("no rag" not in o.reason for o in outs)

    async def test_ranking_metric_no_turns(self):
        t = TestContext(_agent())
        t.rankingMetric("x")
        outs = _eval_all(t)
        assert "no turns recorded" in outs[0].reason

    async def test_ranking_metric_no_rag(self):
        res = RunResult(content="x", iterations_used=1, success=True, metadata={})
        agent = _agent(run_results=[res])
        t = TestContext(agent)
        await t.send("q")
        t.rankingMetric("x")
        outs = _eval_all(t)
        assert "no rag_results" in outs[0].reason

    async def test_citation_resolves_with_n(self):
        res = RunResult(
            content="see [1] and [9]",
            iterations_used=1,
            success=True,
            metadata={"rag_results": [{"content": "a"}, {"content": "b"}]},
        )
        agent = _agent(run_results=[res])
        t = TestContext(agent)
        await t.send("q")
        t.citationResolves(1)  # in range
        t.citationResolves(9)  # out of range
        outs = _eval_all(t)
        assert any(o.value == 1.0 for o in outs)
        assert any(o.value == 0.0 for o in outs)

    async def test_citation_resolves_all(self):
        res = RunResult(
            content="see [1]", iterations_used=1, success=True, metadata={"rag_results": [{"content": "a"}]}
        )
        agent = _agent(run_results=[res])
        t = TestContext(agent)
        await t.send("q")
        t.citationResolves()
        outs = _eval_all(t)
        assert outs[0].value == 1.0

    async def test_abstains_empty_rag(self):
        res = RunResult(content="anything", iterations_used=1, success=True, metadata={})
        agent = _agent(run_results=[res])
        t = TestContext(agent)
        await t.send("q")
        t.abstains()
        outs = _eval_all(t)
        assert outs[0].value == 1.0  # empty rag -> abstains

    async def test_abstains_marker(self):
        res = RunResult(
            content="I don't know", iterations_used=1, success=True, metadata={"rag_results": [{"content": "x"}]}
        )
        agent = _agent(run_results=[res])
        t = TestContext(agent)
        await t.send("q")
        t.abstains()
        t.abstains(markers=["custom marker"])
        outs = _eval_all(t)
        assert outs[0].value == 1.0  # refused

    async def test_blocked_and_warned(self):
        res = RunResult(
            content="x",
            iterations_used=1,
            success=True,
            metadata={
                "guardrail_outcomes": [
                    {"action": "block", "direction": "input"},
                    {"action": "warn", "guardrail": "G"},
                ]
            },
        )
        agent = _agent(run_results=[res])
        t = TestContext(agent)
        await t.send("q")
        t.blocked("input")
        t.blocked()
        t.warned("G")
        t.warned()
        outs = _eval_all(t)
        assert any("1 block(s)" in o.reason for o in outs)
        assert any("1 warn(s)" in o.reason for o in outs)


class TestSkillAndCompleted:
    async def test_activated_skill(self):
        agent = _agent(run_results=[RunResult(content="x", iterations_used=1, success=True)])
        snap = MagicMock(skills_activated=["review"])
        tel = MagicMock(snapshot=snap)
        agent.get_telemetry.return_value = tel
        t = TestContext(agent)
        await t.send("q")
        t.activatedSkill("review")
        outs = _eval_all(t)
        assert any("-> 1" in o.reason for o in outs)

    async def test_completed(self):
        agent = _agent(run_results=[RunResult(content="x", iterations_used=1, success=True)])
        t = TestContext(agent)
        await t.send("q")
        t.completed()
        outs = _eval_all(t)
        assert outs[0].value == 1.0

    def test_completed_no_turns(self):
        t = TestContext(_agent())
        t.completed()
        outs = _eval_all(t)
        assert "no turns recorded" in outs[0].reason


class TestCheckAndJudge:
    async def test_check_truthy_and_matcher(self):
        from koboi.eval.t.assertions import Contains

        agent = _agent(run_results=[RunResult(content="hello world", iterations_used=1, success=True)])
        t = TestContext(agent)
        await t.send("q")
        t.check(True)
        t.check("hello world", Contains("hello"))
        t.check("nope", Contains("missing"))
        outs = _eval_all(t)
        assert outs[0].value == 1.0
        assert outs[1].value == 1.0
        # check is SOFT by default -> a failed check scores 0.5 (not 0.0)
        assert outs[2].value == 0.5

    async def test_judge_unresolvable_scorer(self):
        agent = _agent(run_results=[RunResult(content="x", iterations_used=1, success=True)])
        t = TestContext(agent)
        await t.send("q")
        await t.judge(12345)  # not str/type/BaseScorer -> _resolve raises -> fail-soft
        outs = _eval_all(t)
        assert any("judge unavailable" in o.reason for o in outs)

    async def test_judge_with_scorer_instance_error(self):
        from koboi.eval.scorers.base import BaseScorer

        scorer = MagicMock(spec=BaseScorer)  # spec so isinstance(scorer, BaseScorer) is True
        scorer.score = AsyncMock(side_effect=RuntimeError("boom"))
        agent = _agent(run_results=[RunResult(content="x", iterations_used=1, success=True)])
        t = TestContext(agent)
        await t.send("q")
        await t.judge(scorer)
        outs = _eval_all(t)
        assert any("judge error" in o.reason for o in outs)

    async def test_resolve_scorer_type_error(self):
        t = TestContext(_agent())
        with pytest.raises(TypeError):
            t._resolve_scorer(12345)

    async def test_synthetic_case_and_build_context(self):
        res = RunResult(
            content="x",
            iterations_used=1,
            success=True,
            metadata={"rag_results": [{"content": "ctx"}]},
            token_usage=TokenUsage(prompt_tokens=1, completion_tokens=2),
        )
        agent = _agent(run_results=[res])
        snap = MagicMock(skills_activated=["s"])
        tel = MagicMock(snapshot=snap)
        agent.get_telemetry.return_value = tel
        t = TestContext(agent)
        await t.send("q")
        case = t._synthetic_case(expected=["a"], expected_answer="ans")
        assert case.expected_answer == "ans" and case.context_docs == ["ctx"]
        ctx = t._build_context()
        assert ctx["rag_augmented"] is True
        assert ctx["skills_activated"] == ["s"]


class TestRecordGateError:
    def test_record_gate_error(self):
        t = TestContext(_agent())
        t.record_gate_error("boom")
        outs = _eval_all(t)
        assert outs[0].value == 0.0 and "boom" in outs[0].reason
