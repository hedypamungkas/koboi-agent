"""Tests for koboi.eval.scorers.test_suite.TestSuiteScorer (Wave 1)."""

from __future__ import annotations

import sys

from koboi.eval.registry import ScorerRegistry
from koboi.eval.scorers.test_suite import TestSuiteScorer
from koboi.types import EvalCase

# Interpreter-module form: the restricted sandbox scrubs env/PATH, so bare
# `pytest`/`python` lookups are unreliable -- sys.executable is the contract.
_PASS_CMD = f"{sys.executable} -m unittest discover -q"


def _case(**kw) -> EvalCase:
    base = {"name": "c", "user_message": "m"}
    base.update(kw)
    return EvalCase(**base)


def _project(tmp_path, passing: bool):
    body = "return a + b" if passing else "return a - b"
    (tmp_path / "calc.py").write_text(f"def add(a, b):\n    {body}\n")
    (tmp_path / "test_calc.py").write_text(
        "import unittest\n"
        "from calc import add\n\n\n"
        "class T(unittest.TestCase):\n"
        "    def test_add(self):\n"
        "        self.assertEqual(add(2, 3), 5)\n"
    )
    return tmp_path


class TestTestSuiteScorer:
    async def test_passing_suite_scores_one(self, tmp_path):
        ws = _project(tmp_path, passing=True)
        scorer = TestSuiteScorer()
        s = await scorer.score(_case(test_command=_PASS_CMD), "", {"workspace": str(ws)})
        assert s.value == 1.0
        assert "exit=0" in s.reason

    async def test_failing_suite_scores_zero_with_exit_code(self, tmp_path):
        ws = _project(tmp_path, passing=False)
        scorer = TestSuiteScorer()
        s = await scorer.score(_case(test_command=_PASS_CMD), "", {"workspace": str(ws)})
        assert s.value == 0.0
        assert "exit=" in s.reason and "exit=0" not in s.reason

    async def test_not_applicable_convention(self):
        s = await TestSuiteScorer().score(_case(), "", {})
        assert s.value == 1.0
        assert "not applicable" in s.reason

    async def test_workspace_without_command_fails(self, tmp_path):
        s = await TestSuiteScorer().score(_case(), "", {"workspace": str(tmp_path)})
        assert s.value == 0.0
        assert "no test_command" in s.reason

    async def test_command_with_missing_workspace_fails(self, tmp_path):
        s = await TestSuiteScorer().score(_case(test_command=_PASS_CMD), "", {"workspace": str(tmp_path / "gone")})
        assert s.value == 0.0
        assert "workspace missing" in s.reason

    async def test_timeout_scores_zero(self, tmp_path):
        ws = _project(tmp_path, passing=True)
        scorer = TestSuiteScorer(timeout=0.2)
        s = await scorer.score(
            _case(test_command=f"{sys.executable} -c 'import time; time.sleep(5)'"),
            "",
            {"workspace": str(ws)},
        )
        assert s.value == 0.0
        assert "timed out" in s.reason

    async def test_metadata_fallbacks(self, tmp_path):
        ws = _project(tmp_path, passing=True)
        case = _case(metadata={"test_command": _PASS_CMD, "workspace": str(ws)})
        s = await TestSuiteScorer().score(case, "", {})
        assert s.value == 1.0

    async def test_constructor_kwargs_the_t_judge_path(self, tmp_path):
        ws = _project(tmp_path, passing=True)
        scorer = ScorerRegistry.create("test_suite", test_command=_PASS_CMD, workspace=str(ws))
        assert isinstance(scorer, TestSuiteScorer)
        s = await scorer.score(_case(), "", {})
        assert s.value == 1.0

    async def test_case_field_wins_over_constructor(self, tmp_path):
        ws = _project(tmp_path, passing=True)
        scorer = TestSuiteScorer(test_command=f"{sys.executable} -c 'raise SystemExit(9)'")
        s = await scorer.score(_case(test_command=_PASS_CMD), "", {"workspace": str(ws)})
        assert s.value == 1.0  # case.test_command took precedence

    async def test_exit_126_command_not_attributed_to_sandbox(self, tmp_path):
        # P1: a real ``exit 126`` from the test command (a normal POSIX code) must
        # NOT be mislabeled "cwd escaped sandbox workdir" -- only a SANDBOX-emitted
        # 126 (empty stdout + sandbox message) gets that framing.
        ws = _project(tmp_path, passing=True)
        scorer = TestSuiteScorer()
        s = await scorer.score(_case(test_command="exit 126"), "", {"workspace": str(ws)})
        assert s.value == 0.0
        assert "exit=126" in s.reason
        assert "sandbox workdir" not in s.reason

    async def test_signal_exit_codes_score_zero(self, tmp_path):
        ws = _project(tmp_path, passing=True)
        scorer = TestSuiteScorer()
        for code in (137, 139):  # 128+SIGKILL / 128+SIGSEGV
            s = await scorer.score(_case(test_command=f"exit {code}"), "", {"workspace": str(ws)})
            assert s.value == 0.0
            assert f"exit={code}" in s.reason
