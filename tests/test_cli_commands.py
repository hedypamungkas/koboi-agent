"""Tests for koboi.cli_commands -- the core (no-extra) command handlers.

Covers validate / run / chat-print / eval scenarios that previously lived under
the click group in :mod:`koboi.tui.app`. Output is plain (print-based), so we
assert on captured stdout.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import yaml

from koboi.cli_commands import cmd_eval, cmd_run, cmd_validate


def _invoke(fn, *args, **kwargs) -> tuple[int, str, str]:
    """Call a cmd_* handler capturing stdout/stderr; return (exit_code, out, err)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = fn(*args, **kwargs)
    return code, out.getvalue(), err.getvalue()


def _make_temp_config() -> str:
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        yaml.dump(
            {
                "agent": {"name": "test-agent", "max_iterations": 5},
                "llm": {"model": "gpt-4o-mini", "provider": "openai"},
            },
            f,
        )
        return f.name


def _make_mock_agent():
    mock_agent = MagicMock()
    mock_agent.config.agent_name = "test-agent"
    mock_agent.run = AsyncMock(return_value="Hello from agent!")
    return mock_agent


class TestValidate:
    def test_validate_valid_config(self):
        cfg = _make_temp_config()
        code, out, _ = _invoke(cmd_validate, cfg)
        assert code == 0
        assert "valid" in out.lower()

    def test_validate_missing_config_file(self):
        code, _, _ = _invoke(cmd_validate, "nonexistent.yaml")
        assert code != 0


class TestRun:
    def test_run_missing_config_file(self):
        # from_config raises -> exit 1
        code, _, err = _invoke(cmd_run, "nonexistent_config.yaml", "hi", False, False, None)
        assert code == 1

    @patch("koboi.facade.KoboiAgent.from_config")
    def test_run_single_shot_success(self, mock_from_config):
        mock_from_config.return_value = _make_mock_agent()
        cfg = _make_temp_config()
        code, out, _ = _invoke(cmd_run, cfg, "Hi", False, False, None)
        assert code == 0
        assert "Hello from agent!" in out

    @patch("koboi.facade.KoboiAgent.from_config")
    def test_run_from_config_error(self, mock_from_config):
        mock_from_config.side_effect = Exception("API key not configured")
        cfg = _make_temp_config()
        code, _, err = _invoke(cmd_run, cfg, "Hi", False, False, None)
        assert code == 1
        assert "API key not configured" in err

    @patch("koboi.facade.KoboiAgent.from_config")
    def test_run_agent_error(self, mock_from_config):
        agent = _make_mock_agent()
        agent.run.side_effect = RuntimeError("LLM timeout")
        mock_from_config.return_value = agent
        cfg = _make_temp_config()
        code, _, err = _invoke(cmd_run, cfg, "Hi", False, False, None)
        assert code == 1
        assert "LLM timeout" in err

    @patch("koboi.facade.KoboiAgent.from_config")
    def test_run_print_mode_emits_json(self, mock_from_config):
        agent = _make_mock_agent()

        async def fake_stream(msg):
            yield MagicMock()  # event_to_dict tolerates any object with __dict__

        agent.run_stream = fake_stream
        mock_from_config.return_value = agent
        cfg = _make_temp_config()
        # print mode reads message from the explicit arg, not stdin
        code, _, _ = _invoke(cmd_run, cfg, "Hi", False, True, None)
        assert code == 0


class TestEval:
    def test_eval_no_cases_file(self):
        cfg = _make_temp_config()
        code, out, _ = _invoke(cmd_eval, cfg, None)
        # No cases -> prints notice, returns 0
        assert "No eval cases" in out


class TestChatPrint:
    @patch("koboi.facade.KoboiAgent.from_config")
    def test_chat_print_from_config_error_emits_json_error(self, mock_from_config):
        mock_from_config.side_effect = Exception("API key missing")
        cfg = _make_temp_config()
        from koboi.cli_commands import cmd_chat_print

        code, out, _ = _invoke(cmd_chat_print, cfg, False)
        assert code == 1
        assert '"type": "error"' in out or '"type":"error"' in out
        assert "API key missing" in out
