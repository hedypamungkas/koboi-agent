"""Tests for koboi/eval/runner.py -- Evaluation runner."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from koboi.eval.runner import EvalRunner, _default_scorers
from koboi.types import EvalCase, EvalResult, EvalScore, RunResult, TokenUsage


def _make_case(name="test_case", msg="hello", tools=None, keywords=None):
    return EvalCase(
        name=name,
        user_message=msg,
        expected_tools=tools or [],
        expected_keywords=keywords or [],
        max_iterations=5,
    )


def _make_harness_mock(response="response text"):
    harness = MagicMock()
    run_result = MagicMock(spec=RunResult)
    run_result.content = response
    run_result.token_usage = TokenUsage(prompt_tokens=10, completion_tokens=20)
    run_result.tool_calls_made = []
    harness.run = AsyncMock(return_value=run_result)
    harness.close = AsyncMock()
    harness.hook_chain = MagicMock()
    harness.hook_chain.find_hook = MagicMock(return_value=None)
    harness.hook_chain.add = MagicMock()
    return harness


class TestEvalRunner:
    @pytest.mark.asyncio
    async def test_run_case_basic(self):
        harness = _make_harness_mock("test output")
        factory = MagicMock(return_value=harness)
        runner = EvalRunner(harness_factory=factory, scorers=[], threshold=0.5)
        case = _make_case()
        result = await runner.run_case(case)
        assert isinstance(result, EvalResult)
        assert result.case_name == "test_case"
        assert result.output == "test output"

    @pytest.mark.asyncio
    async def test_run_case_with_scorers(self):
        harness = _make_harness_mock("result")
        factory = MagicMock(return_value=harness)

        class FakeScorer:
            async def score(self, case, output, context):
                return EvalScore("fake", 0.9, "good")

        runner = EvalRunner(harness_factory=factory, scorers=[FakeScorer()])
        result = await runner.run_case(_make_case())
        assert result.overall_score == 0.9
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_run_case_scorer_error(self):
        harness = _make_harness_mock()
        factory = MagicMock(return_value=harness)

        class ErrorScorer:
            async def score(self, case, output, context):
                raise RuntimeError("scorer broke")

        runner = EvalRunner(harness_factory=factory, scorers=[ErrorScorer()])
        result = await runner.run_case(_make_case())
        assert result.scores[0].value == 0.0
        assert "Error" in result.scores[0].reason

    @pytest.mark.asyncio
    async def test_run_case_with_tool_definitions(self):
        harness = _make_harness_mock()
        harness.inject_tool_definitions = MagicMock()
        factory = MagicMock(return_value=harness)
        runner = EvalRunner(harness_factory=factory, scorers=[])
        case = _make_case()
        case.tool_definitions = [
            {"function": {"name": "test_fn", "description": "test", "parameters": {"type": "object"}}}
        ]
        await runner.run_case(case)
        harness.inject_tool_definitions.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_case_with_file_attachments(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("file content here")
        harness = _make_harness_mock()
        factory = MagicMock(return_value=harness)
        runner = EvalRunner(harness_factory=factory, scorers=[])
        case = _make_case()
        case.file_attachments = [str(f)]
        result = await runner.run_case(case)
        harness.run.assert_called_once()
        call_msg = harness.run.call_args[0][0]
        assert "file content here" in call_msg

    @pytest.mark.asyncio
    async def test_run_suite_sequential(self):
        harness = _make_harness_mock("ok")
        factory = MagicMock(return_value=harness)
        runner = EvalRunner(harness_factory=factory, scorers=[])
        cases = [_make_case("c1"), _make_case("c2")]
        results = await runner.run_suite(cases, parallel=False)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_run_suite_parallel(self):
        harness = _make_harness_mock("ok")
        factory = MagicMock(return_value=harness)
        runner = EvalRunner(harness_factory=factory, scorers=[])
        cases = [_make_case("c1"), _make_case("c2")]
        results = await runner.run_suite(cases, parallel=True, max_concurrency=2)
        assert len(results) == 2

    def test_format_results_plain(self):
        results = [
            EvalResult(
                case_name="c1",
                output="out",
                scores=[EvalScore("s", 0.8, "ok")],
                overall_score=0.8,
                passed=True,
                duration_seconds=1.5,
            )
        ]
        text = EvalRunner.format_results_plain(results, threshold=0.6)
        assert "PASS" in text
        assert "c1" in text

    def test_format_results_rich(self):
        results = [
            EvalResult(
                case_name="c1",
                output="out",
                scores=[EvalScore("s", 0.8, "ok")],
                overall_score=0.8,
                passed=True,
                duration_seconds=1.5,
            )
        ]
        text = EvalRunner.format_results(results, threshold=0.6)
        assert "c1" in text

    def test_default_scorers(self):
        scorers = _default_scorers()
        assert len(scorers) == 5

    def test_load_attachments_missing(self):
        result = EvalRunner._load_attachments(["/nonexistent/file.txt"])
        assert result == ""

    def test_load_attachments_valid(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("content")
        result = EvalRunner._load_attachments([str(f)])
        assert "content" in result

    def test_inject_tool_definitions_no_tools(self):
        harness = MagicMock()
        harness.tools = None
        EvalRunner._inject_tool_definitions(harness, [{"function": {"name": "t"}}])
        # should not raise since tools is None

    def test_inject_tool_definitions_with_registry(self):
        harness = MagicMock()
        harness.inject_tool_definitions = MagicMock()
        EvalRunner._inject_tool_definitions(harness, [{"function": {"name": "t", "parameters": {}}}])
        harness.inject_tool_definitions.assert_called_once()

    def test_extract_telemetry_none(self):
        harness = MagicMock(spec=[])
        assert EvalRunner._extract_telemetry(harness) is None
