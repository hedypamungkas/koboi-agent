"""Tests for koboi.eval module."""

from __future__ import annotations

from koboi.eval.scorers import (
    ToolUsageScorer,
    KeywordPresenceScorer,
    OutputLengthScorer,
    IterationEfficiencyScorer,
    HealthScoreScorer,
    LLMJudgeScorer,
)
from koboi.types import EvalScore
from tests.conftest import MockClient, make_mock_response


def _mock_case(**kwargs):
    """Create a mock EvalCase-like object."""
    defaults = {"expected_tools": [], "expected_keywords": [], "max_iterations": 5, "user_message": "test"}
    defaults.update(kwargs)
    return type("Case", (), defaults)()


def _mock_telemetry(unique_tools=None, total_iterations=1, health=80):
    """Create a mock telemetry object."""
    tools = unique_tools or []
    snap = type(
        "S",
        (),
        {
            "unique_tools_used": tools,
            "total_iterations": total_iterations,
        },
    )()
    tel = type(
        "T",
        (),
        {
            "snapshot": snap,
            "health_score": lambda self=None: health,
        },
    )()
    return tel


class TestToolUsageScorer:
    async def test_all_tools_used(self):
        scorer = ToolUsageScorer()
        case = _mock_case(expected_tools=["web_search", "calculator"])
        result = await scorer.score(case, "Result", {"telemetry": _mock_telemetry(["web_search", "calculator"])})
        assert result.value == 1.0

    async def test_no_tools_expected(self):
        scorer = ToolUsageScorer()
        case = _mock_case(expected_tools=[])
        result = await scorer.score(case, "Result", {})
        assert result.value == 1.0

    async def test_missing_tools(self):
        scorer = ToolUsageScorer()
        case = _mock_case(expected_tools=["web_search", "calculator"])
        result = await scorer.score(case, "Result", {"telemetry": _mock_telemetry(["web_search"])})
        assert 0 < result.value < 1.0


class TestKeywordPresenceScorer:
    async def test_all_keywords_present(self):
        scorer = KeywordPresenceScorer()
        case = _mock_case(expected_keywords=["sunny", "28"])
        result = await scorer.score(case, "The weather is sunny and 28 degrees", {})
        assert result.value == 1.0

    async def test_some_keywords_missing(self):
        scorer = KeywordPresenceScorer()
        case = _mock_case(expected_keywords=["sunny", "rainy", "cloudy"])
        result = await scorer.score(case, "The weather is sunny today", {})
        assert 0 < result.value < 1.0

    async def test_case_insensitive(self):
        scorer = KeywordPresenceScorer()
        case = _mock_case(expected_keywords=["SUNNY"])
        result = await scorer.score(case, "The weather is sunny", {})
        assert result.value == 1.0


class TestOutputLengthScorer:
    async def test_valid_length(self):
        scorer = OutputLengthScorer(min_length=10, max_length=1000)
        result = await scorer.score(None, "A" * 100, {})
        assert result.value == 1.0

    async def test_too_short(self):
        scorer = OutputLengthScorer(min_length=100, max_length=1000)
        result = await scorer.score(None, "Short", {})
        assert result.value < 1.0

    async def test_too_long(self):
        scorer = OutputLengthScorer(min_length=10, max_length=20)
        result = await scorer.score(None, "A" * 100, {})
        assert result.value < 1.0


class TestIterationEfficiencyScorer:
    async def test_efficient_iterations(self):
        scorer = IterationEfficiencyScorer(target_ratio=0.5)
        case = _mock_case(max_iterations=10)
        result = await scorer.score(case, "Done", {"telemetry": _mock_telemetry(total_iterations=3)})
        assert result.value == 1.0

    async def test_high_iteration_count(self):
        scorer = IterationEfficiencyScorer(target_ratio=0.5)
        case = _mock_case(max_iterations=10)
        result = await scorer.score(case, "Done", {"telemetry": _mock_telemetry(total_iterations=9)})
        assert result.value < 0.8

    async def test_no_telemetry(self):
        scorer = IterationEfficiencyScorer()
        case = _mock_case()
        result = await scorer.score(case, "Done", {})
        assert result.value == 0.5

    async def test_zero_max_iterations(self):
        scorer = IterationEfficiencyScorer()
        case = _mock_case(max_iterations=0)
        result = await scorer.score(case, "Done", {"telemetry": _mock_telemetry(total_iterations=0)})
        assert result.value == 1.0


class TestHealthScoreScorer:
    async def test_good_health(self):
        scorer = HealthScoreScorer()
        case = _mock_case()
        result = await scorer.score(case, "Done", {"telemetry": _mock_telemetry(health=90)})
        assert result.value == 0.9

    async def test_bad_health(self):
        scorer = HealthScoreScorer()
        case = _mock_case()
        result = await scorer.score(case, "Done", {"telemetry": _mock_telemetry(health=20)})
        assert result.value == 0.2

    async def test_no_telemetry(self):
        scorer = HealthScoreScorer()
        case = _mock_case()
        result = await scorer.score(case, "Done", {})
        assert result.value == 0.5


class TestLLMJudgeScorer:
    async def test_parses_judge_response(self):
        resp = make_mock_response("SCORE: 4\nREASON: Good answer")
        client = MockClient([resp])
        scorer = LLMJudgeScorer(client=client)
        case = _mock_case(user_message="What is 2+2?")
        result = await scorer.score(case, "The answer is 4", {})
        assert result.value == 0.8
        assert "Good answer" in result.reason

    async def test_handles_unparseable_response(self):
        resp = make_mock_response("I don't know how to score this")
        client = MockClient([resp])
        scorer = LLMJudgeScorer(client=client)
        case = _mock_case(user_message="test")
        result = await scorer.score(case, "some output", {})
        assert result.value < 1.0

    async def test_handles_client_error(self):
        client = MockClient([])
        scorer = LLMJudgeScorer(client=client)
        case = _mock_case(user_message="test")
        result = await scorer.score(case, "output", {})
        assert isinstance(result, EvalScore)
