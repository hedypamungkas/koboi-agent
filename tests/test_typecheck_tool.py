"""Tests for koboi.tools.builtin.typecheck (run_typecheck tool)."""

from __future__ import annotations

from koboi.sandbox.base import SandboxResult
from koboi.tools.builtin.typecheck import run_typecheck


class _FakeSandbox:
    """Minimal sandbox double: returns a canned SandboxResult, no containment."""

    def __init__(self, result: SandboxResult, reject: str | None = None) -> None:
        self._result = result
        self._reject = reject  # if set, validate_path raises for this path prefix

    def validate_path(self, path: str) -> str:
        if self._reject and path.startswith(self._reject):
            raise PermissionError(f"Path '{path}' is outside the sandbox directory")
        return path

    def build_env(self, cfg: dict | None = None) -> dict:
        return {}

    def run(self, command, *, cwd=None, env=None, timeout=None, shell=False, input=None):
        return self._result


class TestRunTypecheck:
    def test_clean_run_says_no_issues(self, tmp_path):
        f = tmp_path / "ok.py"
        f.write_text("x = 1\n")
        result = run_typecheck(
            path=str(f),
            _deps={"sandbox": _FakeSandbox(SandboxResult(0, "", ""))},
        )
        assert result == "No issues found."

    def test_errors_prefixed_with_exit_code(self, tmp_path):
        f = tmp_path / "bad.py"
        f.write_text("x = 1\n")
        ruff_out = "bad.py:1:1: F841 Local variable 'x' is assigned to but never used\n"
        result = run_typecheck(
            path=str(f),
            _deps={"sandbox": _FakeSandbox(SandboxResult(1, ruff_out, ""))},
        )
        assert result.startswith("[exit code: 1]")
        assert "F841" in result

    def test_unknown_checker_rejected(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("x = 1\n")
        result = run_typecheck(
            path=str(f),
            checker="bandit",
            _deps={"sandbox": _FakeSandbox(SandboxResult(0, "", ""))},
        )
        assert "Error" in result
        assert "unknown checker" in result

    def test_checker_config_override(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("x = 1\n")
        seen: dict = {}

        class _Capturing(_FakeSandbox):
            def run(self, command, **kw):
                seen["cmd"] = command
                return SandboxResult(0, "", "")

        run_typecheck(
            path=str(f),
            _tool_config={"checker": "mypy"},
            _deps={"sandbox": _Capturing(SandboxResult(0, "", ""))},
        )
        assert seen["cmd"].startswith("mypy ")

    def test_path_outside_sandbox_blocked(self, tmp_path):
        # _validate_path delegates to sandbox.validate_path; the fake rejects it.
        result = run_typecheck(
            path="/etc/passwd",
            _deps={"sandbox": _FakeSandbox(SandboxResult(0, "", ""), reject="/etc")},
        )
        assert "Error" in result
        assert "no access" in result

    def test_missing_binary_is_graceful(self, tmp_path, monkeypatch):
        f = tmp_path / "x.py"
        f.write_text("x = 1\n")

        class _NoBinary(_FakeSandbox):
            def run(self, command, **kw):
                raise FileNotFoundError(2, "No such file or directory", "ruff")

        result = run_typecheck(
            path=str(f),
            _deps={"sandbox": _NoBinary(SandboxResult(0, "", ""))},
        )
        assert "Error" in result
        assert "not installed" in result
