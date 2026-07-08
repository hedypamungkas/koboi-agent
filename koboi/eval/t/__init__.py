"""koboi/eval/t -- eve-style `t` authoring surface for evals.

Write ``evals/**/*.eval.py`` files exporting ``async def test_*(t)`` functions.
The ``t`` object drives the agent (``await t.send(...)``) and records assertions
(``t.calledTool``, ``t.check``, ``t.judge``, ...) which are folded into real
``EvalResult`` objects with gate/soft severity.

Run via ``koboi eval-test <path> [--strict]`` or :func:`run_tests`.

Example::

    # evals/calc.eval.py
    from koboi.eval.t import scripted_response, scripted_tool_call

    MOCK_RESPONSES = [
        scripted_response(None, [scripted_tool_call("calculator", {"expression": "2+2"})]),
        scripted_response("The answer is 4"),
    ]

    async def test_adds(t):
        await t.send("What is 2+2?")
        t.calledTool("calculator")
        t.check(t.reply, Contains("4"))
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from koboi.eval.t.assertions import (
    AssertionOutcome,
    Contains,
    Equals,
    Matcher,
    Matches,
    RecordedAssertion,
    Regex,
    Severity,
    Truth,
    coerce_matcher,
)
from koboi.eval.t.context import TestContext
from koboi.eval.t.loader import LoadedTest, PythonTestLoader, discover
from koboi.eval.t.mock import ScriptedClient, scripted_response, scripted_tool_call
from koboi.eval.t.runner import TestRunner

if TYPE_CHECKING:
    from koboi.types import EvalResult

__all__ = [
    "AssertionOutcome",
    "Contains",
    "Equals",
    "LoadedTest",
    "Matcher",
    "Matches",
    "PythonTestLoader",
    "RecordedAssertion",
    "Regex",
    "ScriptedClient",
    "Severity",
    "TestContext",
    "TestRunner",
    "Truth",
    "coerce_matcher",
    "discover",
    "run_tests",
    "run_tests_sync",
    "scripted_response",
    "scripted_tool_call",
]


async def run_tests(
    path: str | Path,
    *,
    glob: str = "**/*.eval.py",
    threshold: float = 0.6,
    default_severity: Severity = Severity.GATE,
    parallel: bool = False,
    max_concurrency: int = 5,
    tags: list[str] | None = None,
    config: str | dict | None = None,
    mock: bool | None = None,
    mock_responses: list | None = None,
    console=None,
) -> list[EvalResult]:
    """Discover and run all ``async def test_*`` in ``*.eval.py`` files under ``path``.

    Returns real :class:`~koboi.types.EvalResult` objects so
    :meth:`EvalRunner.format_results` and :class:`RegressionTracker` work unchanged.
    """
    tests = PythonTestLoader(glob=glob).discover(path)
    runner = TestRunner(threshold=threshold, default_severity=default_severity, console=console)
    return await runner.run_tests(
        tests,
        parallel=parallel,
        max_concurrency=max_concurrency,
        tags=tags,
        config=config,
        mock=mock,
        mock_responses=mock_responses,
    )


def run_tests_sync(*args, **kwargs) -> list[EvalResult]:
    """Blocking wrapper around :func:`run_tests` for CLI / sync callers."""
    return asyncio.run(run_tests(*args, **kwargs))
