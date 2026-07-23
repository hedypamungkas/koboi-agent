"""E2E proof of the Wave 1 coding harness: workspace -> agent edit -> real tests gate.

Uses a mock harness whose ``run`` actually mutates the workspace (standing in
for the agent's edit_file), so the pipeline exercised is exactly production's:
EvalRunner materializes a clone, the "agent" edits files inside it, and
TestSuiteScorer runs the project's genuine unittest suite -- pass/fail comes
from the real exit code, not text similarity.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from koboi.eval.runner import EvalRunner
from koboi.eval.scorers.test_suite import TestSuiteScorer
from koboi.types import EvalCase, RunResult, TokenUsage

_TEST_CMD = f"{sys.executable} -m unittest discover -q"


def _git(args: list[str], cwd: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def buggy_repo(tmp_path):
    repo = tmp_path / "buggy_repo"
    repo.mkdir()
    _git(["init"], str(repo))
    _git(["config", "user.email", "eval@test.local"], str(repo))
    _git(["config", "user.name", "Eval Test"], str(repo))
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (repo / "test_calc.py").write_text(
        "import unittest\n"
        "from calc import add\n\n\n"
        "class T(unittest.TestCase):\n"
        "    def test_add(self):\n"
        "        self.assertEqual(add(2, 3), 5)\n"
    )
    _git(["add", "."], str(repo))
    _git(["commit", "-m", "buggy"], str(repo))
    return repo


def _harness(fix: bool):
    """Mock agent whose run() edits calc.py in its workspace when fix=True."""
    harness = MagicMock()
    harness.workspace: str | None = None

    async def run(_message: str) -> RunResult:
        if fix and harness.workspace:
            calc = Path(harness.workspace) / "calc.py"
            calc.write_text(calc.read_text().replace("return a - b", "return a + b"))
        return RunResult(content="done", token_usage=TokenUsage())

    harness.run = AsyncMock(side_effect=run)
    harness.close = AsyncMock()
    harness.hook_chain = MagicMock()
    harness.hook_chain.add = MagicMock()
    return harness


class TestCodingEvalEndToEnd:
    async def test_fixing_agent_passes(self, buggy_repo, tmp_path):
        harness = _harness(fix=True)

        def factory(workspace: str):
            harness.workspace = workspace
            return harness

        runner = EvalRunner(
            harness_factory=factory,
            scorers=[TestSuiteScorer()],
            workspace_root=str(tmp_path / "ws"),
        )
        (tmp_path / "ws").mkdir()
        case = EvalCase(
            name="fix-add",
            user_message="add() subtracts; fix it",
            repo=str(buggy_repo),
            test_command=_TEST_CMD,
        )
        result = await runner.run_case(case)
        assert result.passed is True
        assert result.scores[0].value == 1.0
        # Source repo must be untouched -- the agent edited only its clone.
        assert (buggy_repo / "calc.py").read_text() == "def add(a, b):\n    return a - b\n"

    async def test_non_fixing_agent_fails(self, buggy_repo, tmp_path):
        harness = _harness(fix=False)

        def factory(workspace: str):
            harness.workspace = workspace
            return harness

        runner = EvalRunner(
            harness_factory=factory,
            scorers=[TestSuiteScorer()],
            workspace_root=str(tmp_path / "ws"),
        )
        (tmp_path / "ws").mkdir()
        case = EvalCase(
            name="no-fix",
            user_message="add() subtracts; fix it",
            repo=str(buggy_repo),
            test_command=_TEST_CMD,
        )
        result = await runner.run_case(case)
        assert result.passed is False
        assert result.scores[0].value == 0.0
