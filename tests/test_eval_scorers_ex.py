"""Tests for koboi/eval/scorers/ -- Additional scorer coverage."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from koboi.eval.scorers.base import (
    CostScorer,
    HealthScoreScorer,
    IterationEfficiencyScorer,
    KeywordPresenceScorer,
    LLMJudgeScorer,
    OutputLengthScorer,
    ToolUsageScorer,
    RAGNoiseScorer,
    ContextEfficiencyScorer,
    ToolSelectionScorer,
    TokenEfficiencyScorer,
)
from koboi.types import EvalCase, EvalScore, TokenUsage, ToolCall


def _case(**kwargs):
    defaults = dict(name="t", user_message="q", expected_tools=[], expected_keywords=[], max_iterations=10)
    defaults.update(kwargs)
    return EvalCase(**defaults)


class TestToolUsageScorer:
    @pytest.mark.asyncio
    async def test_no_expected_tools(self):
        s = ToolUsageScorer()
        score = await s.score(_case(), "out", {})
        assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_no_telemetry(self):
        s = ToolUsageScorer()
        score = await s.score(_case(expected_tools=["read"]), "out", {})
        assert score.value == 0.0

    @pytest.mark.asyncio
    async def test_all_tools_used(self):
        s = ToolUsageScorer()
        telemetry = MagicMock()
        telemetry.snapshot.unique_tools_used = {"read", "write"}
        score = await s.score(_case(expected_tools=["read", "write"]), "out", {"telemetry": telemetry})
        assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_partial_tools_used(self):
        s = ToolUsageScorer()
        telemetry = MagicMock()
        telemetry.snapshot.unique_tools_used = {"read"}
        score = await s.score(_case(expected_tools=["read", "write"]), "out", {"telemetry": telemetry})
        assert score.value == 0.5


class TestKeywordPresenceScorer:
    @pytest.mark.asyncio
    async def test_no_expected(self):
        s = KeywordPresenceScorer()
        score = await s.score(_case(), "output", {})
        assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_all_found(self):
        s = KeywordPresenceScorer()
        score = await s.score(_case(expected_keywords=["hello", "world"]), "hello world", {})
        assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_partial(self):
        s = KeywordPresenceScorer()
        score = await s.score(_case(expected_keywords=["hello", "missing"]), "hello", {})
        assert score.value == 0.5

    @pytest.mark.asyncio
    async def test_none_found(self):
        s = KeywordPresenceScorer()
        score = await s.score(_case(expected_keywords=["missing"]), "hello", {})
        assert score.value == 0.0


class TestOutputLengthScorer:
    @pytest.mark.asyncio
    async def test_empty(self):
        s = OutputLengthScorer()
        score = await s.score(_case(), "", {})
        assert score.value == 0.0

    @pytest.mark.asyncio
    async def test_too_short(self):
        s = OutputLengthScorer(min_length=100)
        score = await s.score(_case(), "short", {})
        assert score.value == 0.3

    @pytest.mark.asyncio
    async def test_too_long(self):
        s = OutputLengthScorer(max_length=5)
        score = await s.score(_case(), "a" * 100, {})
        assert score.value == 0.7

    @pytest.mark.asyncio
    async def test_just_right(self):
        s = OutputLengthScorer()
        score = await s.score(_case(), "a" * 100, {})
        assert score.value == 1.0


class TestIterationEfficiencyScorer:
    @pytest.mark.asyncio
    async def test_no_telemetry(self):
        s = IterationEfficiencyScorer()
        score = await s.score(_case(), "out", {})
        assert score.value == 0.5

    @pytest.mark.asyncio
    async def test_efficient(self):
        s = IterationEfficiencyScorer()
        telemetry = MagicMock()
        telemetry.snapshot.total_iterations = 2
        score = await s.score(_case(max_iterations=10), "out", {"telemetry": telemetry})
        assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_moderate(self):
        s = IterationEfficiencyScorer()
        telemetry = MagicMock()
        telemetry.snapshot.total_iterations = 7
        score = await s.score(_case(max_iterations=10), "out", {"telemetry": telemetry})
        assert score.value == 0.7

    @pytest.mark.asyncio
    async def test_inefficient(self):
        s = IterationEfficiencyScorer()
        telemetry = MagicMock()
        telemetry.snapshot.total_iterations = 9
        score = await s.score(_case(max_iterations=10), "out", {"telemetry": telemetry})
        assert score.value == 0.4

    @pytest.mark.asyncio
    async def test_single_iteration(self):
        s = IterationEfficiencyScorer()
        telemetry = MagicMock()
        telemetry.snapshot.total_iterations = 1
        score = await s.score(_case(max_iterations=2), "out", {"telemetry": telemetry})
        assert score.value == 1.0  # ratio 0.5 == target_ratio


class TestHealthScoreScorer:
    @pytest.mark.asyncio
    async def test_no_telemetry(self):
        s = HealthScoreScorer()
        score = await s.score(_case(), "out", {})
        assert score.value == 0.5

    @pytest.mark.asyncio
    async def test_with_telemetry(self):
        s = HealthScoreScorer()
        telemetry = MagicMock()
        telemetry.health_score.return_value = 85
        score = await s.score(_case(), "out", {"telemetry": telemetry})
        assert score.value == 0.85


class TestCostScorer:
    @pytest.mark.asyncio
    async def test_no_usage(self):
        s = CostScorer()
        score = await s.score(_case(), "out", {})
        assert score.value == 0.5

    @pytest.mark.asyncio
    async def test_with_usage(self):
        s = CostScorer(max_tokens=10000)
        usage = TokenUsage(prompt_tokens=100, completion_tokens=200)
        score = await s.score(_case(), "out", {"token_usage": usage})
        assert score.value > 0.9  # 300/10000 = 0.03, so score = 0.97

    @pytest.mark.asyncio
    async def test_high_usage(self):
        s = CostScorer(max_tokens=100)
        usage = TokenUsage(prompt_tokens=50, completion_tokens=60)
        score = await s.score(_case(), "out", {"token_usage": usage})
        assert score.value == 0.0  # clamped


class TestLLMJudgeScorer:
    def test_parse_score(self):
        s = LLMJudgeScorer(client=None)
        score = s._parse_judge_response("SCORE: 4\nREASON: Good answer")
        assert score.name == "llm_judge"
        assert score.value == 0.8  # 4/5

    def test_parse_no_score(self):
        s = LLMJudgeScorer(client=None)
        score = s._parse_judge_response("I don't know")
        assert score.value == 0.3

    def test_parse_partial(self):
        s = LLMJudgeScorer(client=None)
        score = s._parse_judge_response("SCORE: 3")
        assert score.value == 0.6  # 3/5


# ---------------------------------------------------------------------------
# System-level scorers (M11-M15)
# ---------------------------------------------------------------------------


class TestRAGNoiseScorer:
    @pytest.mark.asyncio
    async def test_no_telemetry(self):
        s = RAGNoiseScorer()
        score = await s.score(_case(), "out", {})
        assert score.value == 1.0
        assert "No telemetry" in score.reason

    @pytest.mark.asyncio
    async def test_no_rag_used(self):
        s = RAGNoiseScorer()
        telemetry = MagicMock()
        score = await s.score(_case(), "out", {"telemetry": telemetry, "rag_augmented": False})
        assert score.value == 1.0
        assert "not used" in score.reason.lower()

    @pytest.mark.asyncio
    async def test_rag_used_keywords_found(self):
        s = RAGNoiseScorer()
        telemetry = MagicMock()
        score = await s.score(
            _case(expected_keywords=["hello", "world"]),
            "hello world answer",
            {"telemetry": telemetry, "rag_augmented": True},
        )
        assert score.value == 1.0
        assert "useful" in score.reason.lower()

    @pytest.mark.asyncio
    async def test_rag_used_no_keywords_found(self):
        s = RAGNoiseScorer()
        telemetry = MagicMock()
        score = await s.score(
            _case(expected_keywords=["missing", "absent"]),
            "completely unrelated answer",
            {"telemetry": telemetry, "rag_augmented": True},
        )
        assert score.value == 0.3
        assert "noise" in score.reason.lower()

    @pytest.mark.asyncio
    async def test_rag_used_partial_keywords(self):
        s = RAGNoiseScorer()
        telemetry = MagicMock()
        # 1/2 = 0.5 which is >= 0.5 threshold -> "useful"
        score = await s.score(
            _case(expected_keywords=["hello", "missing"]),
            "hello there",
            {"telemetry": telemetry, "rag_augmented": True},
        )
        assert score.value == 1.0
        assert "useful" in score.reason.lower()

    @pytest.mark.asyncio
    async def test_rag_used_low_keywords(self):
        s = RAGNoiseScorer()
        telemetry = MagicMock()
        # 1/3 = 0.33 which is < 0.5 -> "partial noise"
        score = await s.score(
            _case(expected_keywords=["hello", "missing", "absent"]),
            "hello there",
            {"telemetry": telemetry, "rag_augmented": True},
        )
        assert score.value == 0.6
        assert "partial" in score.reason.lower()

    @pytest.mark.asyncio
    async def test_rag_used_no_expected_keywords(self):
        s = RAGNoiseScorer()
        telemetry = MagicMock()
        score = await s.score(
            _case(expected_keywords=[]),
            "some answer",
            {"telemetry": telemetry, "rag_augmented": True},
        )
        assert score.value == 0.8


class TestContextEfficiencyScorer:
    @pytest.mark.asyncio
    async def test_no_telemetry(self):
        s = ContextEfficiencyScorer()
        score = await s.score(_case(), "out", {})
        assert score.value == 0.5

    @pytest.mark.asyncio
    async def test_high_efficiency(self):
        s = ContextEfficiencyScorer()
        telemetry = MagicMock()
        telemetry.context_efficiency.return_value = 0.95
        score = await s.score(_case(), "out", {"telemetry": telemetry})
        assert score.value == 0.95

    @pytest.mark.asyncio
    async def test_low_efficiency(self):
        s = ContextEfficiencyScorer()
        telemetry = MagicMock()
        telemetry.context_efficiency.return_value = 0.2
        score = await s.score(_case(), "out", {"telemetry": telemetry})
        assert score.value == 0.2


class TestToolSelectionScorer:
    @pytest.mark.asyncio
    async def test_no_expected_tools(self):
        s = ToolSelectionScorer()
        score = await s.score(_case(), "out", {})
        assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_no_tool_calls(self):
        s = ToolSelectionScorer()
        score = await s.score(_case(expected_tools=["calculator"]), "out", {"tool_calls": []})
        assert score.value == 0.0

    @pytest.mark.asyncio
    async def test_exact_match(self):
        s = ToolSelectionScorer()
        calls = [ToolCall(id="1", name="calculator", arguments="{}")]
        score = await s.score(_case(expected_tools=["calculator"]), "out", {"tool_calls": calls})
        assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_subset(self):
        s = ToolSelectionScorer()
        calls = [ToolCall(id="1", name="calculator", arguments="{}")]
        score = await s.score(
            _case(expected_tools=["calculator", "web_search"]),
            "out",
            {"tool_calls": calls},
        )
        assert score.value == 0.7

    @pytest.mark.asyncio
    async def test_superset(self):
        s = ToolSelectionScorer()
        calls = [
            ToolCall(id="1", name="calculator", arguments="{}"),
            ToolCall(id="2", name="web_search", arguments="{}"),
        ]
        score = await s.score(_case(expected_tools=["calculator"]), "out", {"tool_calls": calls})
        assert score.value == 0.5

    @pytest.mark.asyncio
    async def test_no_overlap(self):
        s = ToolSelectionScorer()
        calls = [ToolCall(id="1", name="shell", arguments="{}")]
        score = await s.score(_case(expected_tools=["calculator"]), "out", {"tool_calls": calls})
        assert score.value == 0.3


class TestTokenEfficiencyScorer:
    @pytest.mark.asyncio
    async def test_no_usage(self):
        s = TokenEfficiencyScorer()
        score = await s.score(_case(), "out", {})
        assert score.value == 0.5

    @pytest.mark.asyncio
    async def test_low_usage(self):
        s = TokenEfficiencyScorer(max_tokens=5000)
        usage = TokenUsage(prompt_tokens=100, completion_tokens=200)
        score = await s.score(_case(), "out", {"token_usage": usage})
        assert score.value > 0.9  # 300/5000 = 0.06, score = 0.94

    @pytest.mark.asyncio
    async def test_high_usage(self):
        s = TokenEfficiencyScorer(max_tokens=100)
        usage = TokenUsage(prompt_tokens=50, completion_tokens=60)
        score = await s.score(_case(), "out", {"token_usage": usage})
        assert score.value == 0.0  # clamped

    @pytest.mark.asyncio
    async def test_custom_max(self):
        s = TokenEfficiencyScorer(max_tokens=1000)
        usage = TokenUsage(prompt_tokens=500, completion_tokens=500)
        score = await s.score(_case(), "out", {"token_usage": usage})
        assert score.value == 0.0  # 1000/1000 = 1.0, score = 0.0
