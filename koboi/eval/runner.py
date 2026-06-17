"""koboi/eval/runner.py -- Evaluation runner for agent test cases.

Runs test cases through a harness, scores outputs with heuristic and LLM-as-judge
scorers, and optionally sends results to Langfuse.

Adapted from agent/eval.py.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from koboi.types import EvalCase, EvalScore, EvalResult
from koboi.eval.scorers.base import BaseScorer

_logger = logging.getLogger(__name__)

import importlib.util

_LANGFUSE_AVAILABLE = importlib.util.find_spec("langfuse") is not None

if TYPE_CHECKING:
    from rich.console import Console
    from koboi.hooks.langfuse_hook import LangfuseTracingHook


class EvalRunner:
    """Runs evaluation cases through a harness and scores results."""

    def __init__(
        self,
        harness_factory: Callable,
        scorers: list[BaseScorer] | None = None,
        langfuse_hook: LangfuseTracingHook | None = None,
        session_id: str | None = None,
        console: Console | None = None,
        threshold: float = 0.6,
    ):
        self.harness_factory = harness_factory
        self._external_hook = langfuse_hook
        self._session_id = session_id
        self._console = console
        self.scorers = scorers or _default_scorers()
        self.threshold = threshold

    def _create_hook(self) -> LangfuseTracingHook | None:
        if not self._external_hook and not _LANGFUSE_AVAILABLE:
            return None
        from koboi.hooks.langfuse_hook import LangfuseTracingHook

        hook = LangfuseTracingHook(
            session_id=self._session_id,
        )
        return hook if hook.available else None

    async def run_case(self, case: EvalCase) -> EvalResult:

        hook = self._create_hook()
        harness = self.harness_factory()

        if hook and hasattr(harness, "hook_chain") and harness.hook_chain:
            harness.hook_chain.add(hook)

        # Auto-attach telemetry hook if not present
        self._ensure_telemetry_hook(harness)

        # Inject BFCL tool definitions into harness
        if case.tool_definitions:
            self._inject_tool_definitions(harness, case.tool_definitions)

        # Handle file attachments
        user_message = case.user_message
        if case.file_attachments:
            attachment_content = self._load_attachments(case.file_attachments)
            if attachment_content:
                user_message = f"{user_message}\n\nAttached files:\n{attachment_content}"

        start = time.time()
        try:
            result = await harness.run(user_message)
        finally:
            # Close HTTP client to avoid connection leaks between cases
            await harness.close()
        output = result.content if hasattr(result, "content") else str(result)
        duration = time.time() - start

        telemetry = self._extract_telemetry(harness)

        context: dict = {}
        if telemetry:
            context["telemetry"] = telemetry

        # Propagate token usage from RunResult
        token_usage = getattr(result, "token_usage", None)
        if token_usage:
            context["token_usage"] = token_usage

        # Propagate tool calls from RunResult
        tool_calls = getattr(result, "tool_calls_made", [])
        if tool_calls:
            context["tool_calls"] = tool_calls

        scores = []
        for scorer in self.scorers:
            try:
                s = await scorer.score(case, output, context)
                scores.append(s)
            except Exception as e:
                scores.append(EvalScore(scorer.__class__.__name__, 0.0, f"Error: {e}"))

        overall = sum(s.value for s in scores) / len(scores) if scores else 0.0
        passed = overall >= self.threshold

        trace_id = None
        if hook:
            trace_id = hook.trace_id
            self._push_scores_to_langfuse(harness, trace_id, scores)

        return EvalResult(
            case_name=case.name,
            output=output,
            scores=scores,
            overall_score=round(overall, 3),
            telemetry_report=telemetry.report() if telemetry else {},
            trace_id=trace_id,
            duration_seconds=round(duration, 2),
            token_usage=token_usage,
            tool_calls_made=tool_calls,
            passed=passed,
            metadata={"tags": case.tags, "framework": case.metadata.get("framework")},
        )

    @staticmethod
    def _load_attachments(file_paths: list[str]) -> str:
        """Load file attachments and return concatenated content."""
        parts: list[str] = []
        for fp in file_paths:
            path = Path(fp)
            if path.exists() and path.is_file():
                try:
                    content = path.read_text(errors="replace")
                    parts.append(f"--- {path.name} ---\n{content}\n--- end {path.name} ---")
                except Exception as e:
                    _logger.warning("Failed to read attachment %s: %s", fp, e)
            else:
                _logger.warning("Attachment not found: %s", fp)
        return "\n\n".join(parts)

    @staticmethod
    def _inject_tool_definitions(harness, tool_definitions: list[dict]) -> None:
        """Inject tool definitions from eval case into harness tool registry."""
        if hasattr(harness, "inject_tool_definitions"):
            harness.inject_tool_definitions(tool_definitions)

    @staticmethod
    def _extract_telemetry(harness) -> object | None:
        """Extract telemetry from harness."""
        if hasattr(harness, "get_telemetry"):
            return harness.get_telemetry()
        return None

    @staticmethod
    def _ensure_telemetry_hook(harness) -> None:
        """Attach a TelemetryHook if not already present."""
        if hasattr(harness, "ensure_telemetry_hook"):
            harness.ensure_telemetry_hook()

    async def run_suite(
        self,
        cases: list[EvalCase],
        parallel: bool = False,
        max_concurrency: int = 5,
        threshold: float | None = None,
    ) -> list[EvalResult]:
        """Run all cases and return results.

        Args:
            cases: Eval cases to run.
            parallel: If True, run cases concurrently with bounded concurrency.
            max_concurrency: Max parallel cases when parallel=True.
            threshold: Override pass/fail threshold. Defaults to self.threshold.
        """
        effective_threshold = threshold if threshold is not None else self.threshold

        if parallel:
            return await self._run_suite_parallel(cases, max_concurrency, effective_threshold)
        return await self._run_suite_sequential(cases, effective_threshold)

    async def _run_suite_sequential(
        self,
        cases: list[EvalCase],
        threshold: float,
    ) -> list[EvalResult]:
        results = []
        for i, case in enumerate(cases, 1):
            self._print_progress(i, len(cases), case.name)
            result = await self.run_case(case)
            result.passed = result.overall_score >= threshold
            results.append(result)
            self._print_result(result, threshold)
            # Brief delay between cases to avoid rate limiting
            if i < len(cases):
                await asyncio.sleep(1.0)
        return results

    async def _run_suite_parallel(
        self,
        cases: list[EvalCase],
        max_concurrency: int,
        threshold: float,
    ) -> list[EvalResult]:
        semaphore = asyncio.Semaphore(max_concurrency)
        total = len(cases)

        async def _run_with_limit(idx: int, case: EvalCase) -> EvalResult:
            async with semaphore:
                self._print_progress(idx + 1, total, case.name)
                result = await self.run_case(case)
                result.passed = result.overall_score >= threshold
                self._print_result(result, threshold)
                return result

        tasks = [_run_with_limit(i, case) for i, case in enumerate(cases)]
        return await asyncio.gather(*tasks)

    def _print_progress(self, current: int, total: int, name: str) -> None:
        msg = f"  [{current}/{total}] Running: {name}..."
        if self._console:
            self._console.print(msg)
        else:
            print(msg)

    def _print_result(self, result: EvalResult, threshold: float) -> None:
        passed = "PASS" if result.overall_score >= threshold else "FAIL"
        status_msg = f"    {passed} -- Score: {result.overall_score:.1%} ({result.duration_seconds}s)"
        if self._console:
            style = "green" if passed == "PASS" else "red"
            self._console.print(f"[{style}]{status_msg}[/{style}]")
        else:
            print(status_msg)

    def _push_scores_to_langfuse(self, harness, trace_id: str | None, scores: list[EvalScore]) -> None:
        if not trace_id:
            return
        if hasattr(harness, "push_langfuse_scores"):
            harness.push_langfuse_scores(trace_id, scores)

    @staticmethod
    def format_results(results: list[EvalResult], threshold: float = 0.6):
        """Format eval results as a Rich Table (or plain text as fallback)."""
        try:
            from rich.table import Table
        except ImportError:
            return EvalRunner.format_results_plain(results, threshold)

        table = Table(title="Evaluation Results", show_lines=True)
        table.add_column("Status", width=6)
        table.add_column("Test Case", style="bold")
        table.add_column("Score", justify="right")
        table.add_column("Time", justify="right")
        table.add_column("Details", max_width=60)

        for r in results:
            status = "PASS" if r.overall_score >= threshold else "FAIL"
            status_style = "green" if status == "PASS" else "red"
            score_bar = "+" * int(r.overall_score * 10) + "-" * (10 - int(r.overall_score * 10))
            details_lines = []
            for s in r.scores:
                details_lines.append(f"{s.name}: {s.value:.2f} {s.reason}")
            details = "\n".join(details_lines)

            table.add_row(
                f"[{status_style}]{status}[/{status_style}]",
                r.case_name,
                f"{r.overall_score:.1%} [{score_bar}]",
                f"{r.duration_seconds}s",
                details,
            )

        total = len(results)
        passed = sum(1 for r in results if r.overall_score >= threshold)
        avg = sum(r.overall_score for r in results) / total if total else 0
        table.caption = f"Summary: {passed}/{total} passed — Average: {avg:.1%}"

        # Render to string so print() works without a Rich Console
        from rich.console import Console
        from io import StringIO

        buf = StringIO()
        Console(file=buf, force_terminal=True, width=100).print(table)
        return buf.getvalue()

    @staticmethod
    def format_results_plain(results: list[EvalResult], threshold: float = 0.6) -> str:
        """Format results as plain ASCII text (backward-compatible fallback)."""
        lines = [
            "",
            "=" * 70,
            "  EVALUATION RESULTS",
            "=" * 70,
            "",
        ]

        for r in results:
            status = "PASS" if r.overall_score >= threshold else "FAIL"
            lines.append(f"  [{status}] {r.case_name} -- {r.overall_score:.1%} ({r.duration_seconds}s)")
            for s in r.scores:
                bar = "+" * int(s.value * 10) + "-" * (10 - int(s.value * 10))
                lines.append(f"    {s.name:25s} [{bar}] {s.value:.2f}  {s.reason}")
            lines.append("")

        total = len(results)
        passed = sum(1 for r in results if r.overall_score >= threshold)
        avg = sum(r.overall_score for r in results) / total if total else 0
        lines.append(f"  Summary: {passed}/{total} passed -- Average: {avg:.1%}")
        lines.append("=" * 70)
        return "\n".join(lines)


def _default_scorers() -> list[BaseScorer]:
    from koboi.eval.scorers.base import (
        ToolUsageScorer,
        KeywordPresenceScorer,
        OutputLengthScorer,
        IterationEfficiencyScorer,
        HealthScoreScorer,
    )

    return [
        ToolUsageScorer(),
        KeywordPresenceScorer(),
        OutputLengthScorer(),
        IterationEfficiencyScorer(),
        HealthScoreScorer(),
    ]
