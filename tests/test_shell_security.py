"""Tests for the shell tool in koboi.tools.builtin.shell."""

from __future__ import annotations

import os
import subprocess
from unittest.mock import patch, MagicMock

from koboi.tools.builtin.shell import _build_env, _get_npm_root, run_shell, MAX_OUTPUT, TIMEOUT


class TestRunShellBasic:
    def test_simple_command(self):
        result = run_shell("echo hello")
        assert "hello" in result

    def test_command_with_pipe(self):
        result = run_shell("echo hello world | tr ' ' '_'")
        assert "hello_world" in result

    def test_command_with_and(self):
        result = run_shell("echo a && echo b")
        assert "a" in result
        assert "b" in result

    def test_command_not_found(self):
        result = run_shell("nonexistent_command_xyz_12345")
        assert "Error" in result or "not found" in result.lower()

    def test_command_exit_code(self):
        result = run_shell("exit 42")
        assert "exit code: 42" in result

    def test_stderr_captured(self):
        result = run_shell("echo error >&2")
        assert "error" in result

    def test_empty_output(self):
        result = run_shell("true")
        assert "no output" in result


class TestPythonShim:
    def test_python3_shim(self):
        """The 'python' command should be shimmed to 'python3'."""
        result = run_shell("python3 --version")
        assert "Python" in result

    def test_python_replaced(self):
        """'python' (without '3') should be rewritten to 'python3'."""
        result = run_shell("python --version")
        assert "Python" in result

    def test_python3_not_replaced(self):
        """'python3' should remain unchanged."""
        result = run_shell("python3 --version")
        assert "Python" in result


class TestBuildEnv:
    def test_returns_dict(self):
        env = _build_env()
        assert isinstance(env, dict)

    def test_inherits_current_env(self):
        env = _build_env()
        # PATH should always exist
        assert "PATH" in env

    def test_node_path_set_if_npm_root_exists(self):
        """If npm root returns a valid directory, NODE_PATH should be set."""
        _get_npm_root.cache_clear()
        npm_root = _get_npm_root()
        env = _build_env()
        if npm_root and os.path.isdir(npm_root):
            assert "NODE_PATH" in env
            assert npm_root in env["NODE_PATH"]

    def test_node_path_appended_not_replaced(self):
        """If NODE_PATH already exists, npm root should be appended."""
        with patch.dict(os.environ, {"NODE_PATH": "/existing/path"}, clear=False):
            _get_npm_root.cache_clear()
            with patch("koboi.tools.builtin.shell._get_npm_root", return_value="/mock/npm/root"):
                with patch("os.path.isdir", return_value=True):
                    env = _build_env()
                    assert "/mock/npm/root" in env["NODE_PATH"]
                    assert "/existing/path" in env["NODE_PATH"]


class TestNpmRootCaching:
    def test_npm_root_cached(self):
        """_get_npm_root uses lru_cache so repeated calls return the same value."""
        _get_npm_root.cache_clear()
        r1 = _get_npm_root()
        r2 = _get_npm_root()
        assert r1 == r2

    def test_npm_root_returns_string(self):
        _get_npm_root.cache_clear()
        result = _get_npm_root()
        assert isinstance(result, str)

    def test_npm_root_handles_missing_npm(self):
        """If npm is not installed, _get_npm_root should return empty string."""
        _get_npm_root.cache_clear()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = _get_npm_root()
            assert result == ""


class TestWorkingDirectory:
    def test_cwd_option(self, tmp_path):
        """run_shell should use the provided cwd."""
        result = run_shell("pwd", cwd=str(tmp_path))
        assert str(tmp_path) in result

    def test_invalid_cwd(self):
        """Invalid cwd should produce an error."""
        result = run_shell("echo x", cwd="/nonexistent_dir_xyz_12345")
        assert "Error" in result

    def test_empty_cwd_uses_default(self):
        """Empty cwd string should use the current working directory."""
        result = run_shell("pwd")
        assert result.strip()  # Should contain some path


class TestOutputTruncation:
    def test_output_truncated_at_max(self):
        """Output longer than MAX_OUTPUT should be truncated."""
        # Generate output larger than MAX_OUTPUT
        large_cmd = f"python3 -c \"print('x' * {MAX_OUTPUT + 1000})\""
        result = run_shell(large_cmd)
        assert len(result) <= MAX_OUTPUT + 200  # allow room for truncation message
        assert "truncated" in result


class TestTimeout:
    def test_timeout_value(self):
        """Module-level TIMEOUT should be 30 seconds."""
        assert TIMEOUT == 30

    def test_max_output_value(self):
        """Module-level MAX_OUTPUT should be 10000 chars."""
        assert MAX_OUTPUT == 10000

    def test_timeout_returns_error_message(self):
        """A timed-out command should return an error message."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="sleep 999", timeout=30)):
            result = run_shell("sleep 999")
            assert "timed out" in result
