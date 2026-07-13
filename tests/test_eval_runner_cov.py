"""koboi/eval/runner.py -- branch coverage for EvalRunner (mocked harness/scorers)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


from koboi.types import EvalCase, EvalScore
from koboi.eval.runner import EvalRunner


def _case(**kw) -> EvalCase:
    base = dict(name="c1", user_message="hello", tags=["t"], metadata={})
    base.update(kw)
    return EvalCase(**base)


def _score(name: str, val: float) -> AsyncMock:
    m = MagicMock()
    m.__class__.__name__ = name
    m.score = AsyncMock(return_value=EvalScore(name, val, "ok"))
    return m


def _harness(result_content="out", token_usage=None, tool_calls=None) -> MagicMock:
    h = MagicMock()
    result = MagicMock()
    result.content = result_content
    result.token_usage = token_usage
    result.tool_calls_made = tool_calls if tool_calls is not None else []
    h.run = AsyncMock(return_value=result)
    h.close = AsyncMock()
    h.hook_chain = None
    h.core = None
    return h


class TestRunCase:
    async def test_basic_with_telemetry_and_toolcalls(self):
        tel = MagicMock()
        harness = _harness(token_usage={"p": 1}, tool_calls=[{"name": "t"}])
        harness.get_telemetry.return_value = tel
        runner = EvalRunner(harness_factory=lambda: harness, scorers=[_score("A", 1.0), _score("B", 0.5)])
        res = await runner.run_case(_case())
        assert res.case_name == "c1"
        assert res.passed is True  # avg 0.75 >= 0.6
        assert res.token_usage == {"p": 1}
        assert res.tool_calls_made == [{"name": "t"}]
        assert res.telemetry_report == tel.report.return_value

    async def test_scorer_error_branch(self):
        bad = MagicMock()
        bad.__class__.__name__ = "Bad"
        bad.score = AsyncMock(side_effect=RuntimeError("x"))
        harness = _harness()
        runner = EvalRunner(harness_factory=lambda: harness, scorers=[bad])
        res = await runner.run_case(_case())
        assert res.scores[0].value == 0.0
        assert "Error:" in res.scores[0].reason

    async def test_tool_definitions_injected(self):
        harness = _harness()
        runner = EvalRunner(harness_factory=lambda: harness, scorers=[_score("A", 1.0)])
        await runner.run_case(_case(tool_definitions=[{"name": "f"}]))
        harness.inject_tool_definitions.assert_called_once()

    async def test_file_attachments_appended(self, tmp_path):
        f = tmp_path / "note.txt"
        f.write_text("hello world")
        harness = _harness()
        runner = EvalRunner(harness_factory=lambda: harness, scorers=[_score("A", 1.0)])
        await runner.run_case(_case(file_attachments=[str(f)]))
        sent = harness.run.call_args.args[0]
        assert "hello world" in sent and "Attached files" in sent

    async def test_no_scorers(self):
        harness = _harness()
        runner = EvalRunner(harness_factory=lambda: harness)
        runner.scorers = []  # bypass the __init__ default-scorers fallback
        res = await runner.run_case(_case())
        assert res.overall_score == 0.0


class TestLoadAttachments:
    def test_success_and_missing(self, tmp_path, caplog):
        f = tmp_path / "a.txt"
        f.write_text("AAA")
        out = EvalRunner._load_attachments([str(f), "/no/such/file"])
        assert "AAA" in out
        assert "end a.txt" in out

    def test_read_error(self, tmp_path, monkeypatch):
        f = tmp_path / "b.txt"
        f.write_text("x")

        def boom(self, *a, **k):
            raise OSError("denied")

        monkeypatch.setattr("pathlib.Path.read_text", boom)
        out = EvalRunner._load_attachments([str(f)])
        assert out == ""


class TestSuiteRunners:
    async def test_sequential(self, monkeypatch):
        async def _sleep(*_a):
            return None

        monkeypatch.setattr("asyncio.sleep", _sleep)
        harness = _harness()
        runner = EvalRunner(harness_factory=lambda: harness, scorers=[_score("A", 1.0)])
        results = await runner.run_suite([_case(name="a"), _case(name="b")], parallel=False)
        assert len(results) == 2
        assert harness.run.await_count == 2

    async def test_parallel(self):
        harness = _harness()
        runner = EvalRunner(harness_factory=lambda: harness, scorers=[_score("A", 1.0)])
        results = await runner.run_suite([_case(name="a"), _case(name="b")], parallel=True, max_concurrency=2)
        assert len(results) == 2

    async def test_threshold_override(self, monkeypatch):
        async def _sleep(*_a):
            return None

        monkeypatch.setattr("asyncio.sleep", _sleep)
        harness = _harness()
        runner = EvalRunner(harness_factory=lambda: harness, scorers=[_score("A", 0.5)])
        results = await runner.run_suite([_case()], parallel=False, threshold=0.9)
        assert results[0].passed is False  # 0.5 < 0.9


class TestPrinting:
    def test_progress_and_result_with_console(self, capsys):
        console = MagicMock()
        runner = EvalRunner(harness_factory=lambda: None, scorers=[], console=console)
        runner._print_progress(1, 2, "case")
        runner._print_result(MagicMock(overall_score=0.9, duration_seconds=1.0), 0.6)
        assert console.print.call_count == 2

    def test_progress_and_result_no_console(self, capsys):
        runner = EvalRunner(harness_factory=lambda: None, scorers=[])
        runner._print_progress(1, 2, "case")
        runner._print_result(MagicMock(overall_score=0.2, duration_seconds=1.0), 0.6)
        out = capsys.readouterr().out
        assert "Running" in out and "FAIL" in out


class TestLangfusePush:
    def test_push_with_trace_id(self):
        harness = MagicMock()
        harness.push_langfuse_scores = MagicMock()
        runner = EvalRunner(harness_factory=lambda: harness, scorers=[])
        runner._push_scores_to_langfuse(harness, "trace-1", [EvalScore("A", 1.0, "r")])
        harness.push_langfuse_scores.assert_called_once()

    def test_push_no_trace_id(self):
        harness = MagicMock()
        runner = EvalRunner(harness_factory=lambda: harness, scorers=[])
        runner._push_scores_to_langfuse(harness, None, [])
        harness.push_langfuse_scores.assert_not_called()


class TestFormatters:
    def test_format_results_plain(self):
        from koboi.types import EvalResult

        r = EvalResult(
            case_name="c1",
            output="out",
            scores=[EvalScore("A", 0.8, "ok")],
            overall_score=0.8,
            telemetry_report={},
            trace_id=None,
            duration_seconds=1.0,
            token_usage=None,
            tool_calls_made=[],
            passed=True,
            metadata={},
        )
        text = EvalRunner.format_results_plain([r], 0.6)
        assert "PASS" in text and "c1" in text and "Summary" in text

    def test_format_results_rich(self):
        from koboi.types import EvalResult

        r = EvalResult(
            case_name="c1",
            output="out",
            scores=[EvalScore("A", 0.4, "low")],
            overall_score=0.4,
            telemetry_report={},
            trace_id=None,
            duration_seconds=1.0,
            token_usage=None,
            tool_calls_made=[],
            passed=False,
            metadata={},
        )
        text = EvalRunner.format_results([r], 0.6)
        assert "c1" in text


class TestStaticHelpers:
    def test_extract_telemetry_none(self):
        h = MagicMock(spec=[])  # no get_telemetry attr
        assert EvalRunner._extract_telemetry(h) is None

    def test_ensure_telemetry_hook(self):
        h = MagicMock()
        EvalRunner._ensure_telemetry_hook(h)
        h.ensure_telemetry_hook.assert_called_once()

    def test_inject_no_method(self):
        h = MagicMock(spec=[])
        EvalRunner._inject_tool_definitions(h, [{"x": 1}])  # no-op, no raise
