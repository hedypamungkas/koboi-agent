"""Tests for koboi.eval.t.runner -- TestRunner folding, binding, and batch runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from koboi.eval.t import run_tests
from koboi.eval.t.assertions import Contains, Severity
from koboi.eval.t.loader import LoadedTest
from koboi.eval.t.mock import scripted_response, scripted_tool_call
from koboi.eval.t.runner import TestRunner


def _test(file_name, func, **kwargs):
    return LoadedTest(file=Path(file_name), func_name=func.__name__, func=func, **kwargs)


async def _passing(t):
    """A minimal passing test body shared across batch tests."""
    await t.send("q")


class TestRunTestFolding:
    async def test_passing_test(self):
        async def fn(t):
            await t.send("q")
            t.calledTool("calc")
            t.check(t.reply, Contains("4"))

        responses = [scripted_response(None, [scripted_tool_call("calc")]), scripted_response("4")]
        result = await TestRunner(threshold=0.6).run_test(_test("a.eval.py", fn), mock=True, mock_responses=responses)
        assert result.passed is True
        assert result.metadata["framework"] == "eval-test"
        assert result.metadata["gate_failed"] is False

    async def test_gate_failure_fails(self):
        async def fn(t):
            await t.send("q")
            t.calledTool("missing")

        result = await TestRunner().run_test(
            _test("a.eval.py", fn), mock=True, mock_responses=[scripted_response("ok")]
        )
        assert result.passed is False
        assert result.metadata["gate_failed"] is True

    async def test_soft_failure_can_lower_below_threshold(self):
        async def fn(t):
            await t.send("q")
            t.check(t.reply, "nope")  # single soft fail -> 0.5 overall

        result = await TestRunner(threshold=0.6).run_test(
            _test("a.eval.py", fn), mock=True, mock_responses=[scripted_response("ok")]
        )
        assert result.metadata["gate_failed"] is False
        assert result.passed is False  # 0.5 < 0.6 threshold

    async def test_soft_failure_does_not_gate_when_overall_ok(self):
        async def fn(t):
            await t.send("q")
            t.check(t.reply, Contains("ok"))  # soft pass -> 1.0
            t.check(t.reply, "nope")  # soft fail -> 0.5

        result = await TestRunner(threshold=0.6).run_test(
            _test("a.eval.py", fn), mock=True, mock_responses=[scripted_response("ok")]
        )
        assert result.metadata["gate_failed"] is False
        assert result.passed is True
        assert result.overall_score == 0.75

    async def test_uncaught_exception_is_gate_failure(self):
        async def fn(t):
            raise ValueError("boom")

        result = await TestRunner().run_test(
            _test("a.eval.py", fn), mock=True, mock_responses=[scripted_response("ok")]
        )
        assert result.passed is False
        assert result.metadata["gate_failed"] is True
        assert "boom" in (result.metadata["error"] or "")

    async def test_default_severity_soft_disables_gate(self):
        async def fn(t):
            await t.send("q")
            t.calledTool("missing")  # would gate by default

        result = await TestRunner(default_severity=Severity.SOFT, threshold=0.0).run_test(
            _test("a.eval.py", fn), mock=True, mock_responses=[scripted_response("ok")]
        )
        assert result.metadata["gate_failed"] is False

    async def test_live_mode_without_config_raises(self):
        async def fn(t):
            await t.send("q")

        with pytest.raises(ValueError):
            await TestRunner().run_test(_test("a.eval.py", fn))  # no config, no mock


class TestRunTestsBatch:
    async def test_sequential_and_parallel_counts_match(self):
        responses = [scripted_response("ok")]
        tests = [_test("a.eval.py", _passing), _test("b.eval.py", _passing)]
        runner = TestRunner()
        sequential = await runner.run_tests(tests, mock=True, mock_responses=responses)
        parallel = await runner.run_tests(tests, parallel=True, max_concurrency=2, mock=True, mock_responses=responses)
        assert len(sequential) == 2
        assert len(parallel) == 2

    async def test_tags_filter(self):
        responses = [scripted_response("ok")]
        tests = [
            _test("a.eval.py", _passing, tags=["smoke"]),
            _test("b.eval.py", _passing, tags=["integration"]),
        ]
        result = await TestRunner().run_tests(tests, mock=True, mock_responses=responses, tags=["smoke"])
        assert len(result) == 1
        assert result[0].metadata["file"].endswith("a.eval.py")

    async def test_empty_when_no_tags_match(self):
        tests = [_test("a.eval.py", _passing, tags=["smoke"], mock_responses=[scripted_response("ok")])]
        assert await TestRunner().run_tests(tests, mock=True, tags=["nope"]) == []

    async def test_batch_survives_misconfigured_test(self):
        """A single misconfigured test must fold into a failed result, not abort the suite."""

        async def good(t):
            await t.send("q")

        async def bad(t):
            await t.send("q")  # body is fine; the test just lacks config/mock

        tests = [
            _test("good.eval.py", good, mock_responses=[scripted_response("ok")]),
            _test("bad.eval.py", bad),  # no config, no mock -> build error
        ]
        results = await TestRunner().run_tests(tests)
        assert len(results) == 2  # batch was not aborted
        good_result = next(r for r in results if "good" in r.case_name)
        bad_result = next(r for r in results if "bad" in r.case_name)
        assert good_result.passed is True
        assert bad_result.passed is False
        assert bad_result.metadata["error"]


class TestInterop:
    async def test_public_run_tests_with_format_and_regression(self, tmp_path):
        (tmp_path / "calc.eval.py").write_text(
            "from koboi.eval.t import scripted_response, scripted_tool_call\n"
            "MOCK_RESPONSES=[scripted_response(None, [scripted_tool_call('calc')]), scripted_response('4')]\n"
            "async def test_calc(t):\n"
            "    await t.send('q')\n"
            "    t.calledTool('calc')\n"
        )
        results = await run_tests(tmp_path, threshold=0.6)
        assert len(results) == 1
        assert results[0].passed is True

        # Real EvalResults flow through the shared formatter unchanged.
        from koboi.eval.runner import EvalRunner

        formatted = EvalRunner.format_results(results, 0.6)
        assert "calc" in formatted

        # Regression tracking works unchanged.
        from koboi.eval.regression import RegressionTracker

        tracker = RegressionTracker(baseline_dir=str(tmp_path / "baselines"))
        tracker.save_baseline("suite", results)
        baseline = tracker.load_baseline("suite")
        assert baseline is not None
        assert baseline[0]["case_name"].endswith("test_calc")
        report = tracker.compare(results, baseline)
        assert report.has_regression is False

    async def test_config_mock_mode_swaps_client(self, tmp_path):
        """Mock mode with a CONFIG builds a real agent and swaps only the LLM transport."""
        config_path = tmp_path / "agent.yaml"
        config_path.write_text(
            "agent:\n  name: t\n  max_iterations: 3\n  system_prompt: helpful\n"
            "llm:\n  model: gpt-4o-mini\n  api_key: test-key\n  base_url: http://localhost:8080/v1\n"
        )
        responses = [scripted_response("hello from script")]
        from koboi.eval.t.mock import ScriptedClient

        runner = TestRunner()
        agent = await runner._build_mock_agent(
            _test("a.eval.py", _passing),
            responses=responses,
            config=str(config_path),
        )
        assert isinstance(agent.core.client, ScriptedClient)
        result = await agent.run("q")
        assert result.content == "hello from script"
        await agent.close()
