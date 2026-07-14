"""Branch coverage for ``koboi/cli_commands.py`` and ``koboi/cli.py``.

Targets the gaps left by tests/test_cli_commands.py, test_cli.py, test_cli_graph.py
and test_cli_sessions.py: the streaming helpers, validate failure branches, run
resume/no-message paths, sessions delete, eval runner path, the whole eval-test /
diagnostics / init-zsh bodies, and the cli.py serve/keys/mcp-serve/chat/graph
dispatch + fallback.

Command bodies are plain functions returning an int exit code; we call them
directly and assert on captured output. Optional extras (fastapi/textual/mcp) are
installed in this venv, so lazy imports succeed and we patch the leaf callables.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from koboi.cli_commands import (
    _chat_print_mode,
    _run_print_mode,
    cmd_chat_print,
    cmd_diagnostics,
    cmd_eval,
    cmd_eval_test,
    cmd_graph,
    cmd_init_zsh,
    cmd_run,
    cmd_sessions,
    cmd_validate,
)
from koboi.types import EvalResult

EVALS = Path(__file__).resolve().parent.parent / "evals"


@pytest.fixture(autouse=True)
def _isolate_env():
    """Snapshot/restore os.environ (cli.main/load_dotenv and env-key reads)."""
    saved = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(saved)


def _invoke(fn, *args, **kwargs):
    """Call a cmd_* handler capturing stdout/stderr; return (exit, out, err)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = fn(*args, **kwargs)
    return code, out.getvalue(), err.getvalue()


def _write_cfg(tmp_path, **overrides) -> str:
    """A valid-enough config for from_config-mocked commands."""
    data = {
        "agent": {"name": "t", "max_iterations": 5},
        "llm": {"model": "gpt-4o-mini", "provider": "openai", "api_key": "sk-test-key-1234"},
    }
    data.update(overrides)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.dump(data))
    return str(p)


def _write_cfg_yaml(tmp_path, mapping) -> str:
    p = tmp_path / "c.yaml"
    p.write_text(yaml.dump(mapping))
    return str(p)


def _invoke_cli(argv):
    """Run cli.main(argv) capturing stdout/stderr; return (exit, out, err)."""
    from koboi import cli

    old = sys.argv
    sys.argv = ["koboi"] + argv
    out, err = io.StringIO(), io.StringIO()
    code = 0
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            cli.main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    finally:
        sys.argv = old
    return code, out.getvalue(), err.getvalue()


# --------------------------------------------------------------------------- #
# streaming helpers (_run_print_mode / _chat_print_mode)
# --------------------------------------------------------------------------- #
class TestStreamingHelpers:
    def test_run_print_mode_emits_jsonl(self, capsys):
        agent = MagicMock()

        async def fake_stream(msg):
            yield {"type": "text", "content": "hi"}

        agent.run_stream = fake_stream
        asyncio.run(_run_print_mode(agent, "hello"))
        out = capsys.readouterr().out.strip()
        # event_to_dict wraps unknown objects as {"type": "unknown", "data": ...}
        payload = json.loads(out)
        assert "type" in payload

    def test_chat_print_mode_eof_breaks(self, capsys, monkeypatch):
        agent = MagicMock()
        agent.config.agent_name = "t"
        agent.config.provider = "openai"
        agent.config.model = "m"

        def _eof():
            raise EOFError

        monkeypatch.setattr("builtins.input", _eof)
        asyncio.run(_chat_print_mode(agent))
        out = capsys.readouterr().out
        assert "session_start" in out
        assert "session_end" in out

    def test_chat_print_mode_message_then_quit(self, capsys, monkeypatch):
        agent = MagicMock()
        agent.config.agent_name = "t"
        agent.config.provider = "openai"
        agent.config.model = "m"

        async def fake_stream(msg):
            yield {"type": "text", "content": "reply"}

        agent.run_stream = fake_stream
        answers = iter(["hello", "quit"])
        monkeypatch.setattr("builtins.input", lambda: next(answers))
        asyncio.run(_chat_print_mode(agent))
        out = capsys.readouterr().out
        assert "session_start" in out
        assert "session_end" in out

    def test_chat_print_mode_blank_then_exit(self, capsys, monkeypatch):
        agent = MagicMock()
        agent.config.agent_name = "t"
        agent.config.provider = "openai"
        agent.config.model = "m"
        answers = iter(["   ", "exit"])
        monkeypatch.setattr("builtins.input", lambda: next(answers))
        asyncio.run(_chat_print_mode(agent))
        assert "session_end" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# cmd_graph -- config parse error
# --------------------------------------------------------------------------- #
class TestGraph:
    def test_graph_bad_config(self, capsys):
        rc = cmd_graph("/no/such/config.yaml", "json")
        assert rc == 1
        assert "Config parse error" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# cmd_validate -- failure branches
# --------------------------------------------------------------------------- #
class TestValidateFailures:
    def test_unknown_provider(self, tmp_path, capsys):
        cfg = _write_cfg_yaml(
            tmp_path, {"agent": {"name": "a"}, "llm": {"model": "m", "provider": "weird", "api_key": "sk-x"}}
        )
        assert cmd_validate(cfg) == 1
        assert "Unknown provider" in capsys.readouterr().err

    def test_placeholder_api_key_no_env(self, tmp_path, capsys, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = _write_cfg_yaml(
            tmp_path,
            {"agent": {"name": "a"}, "llm": {"model": "m", "provider": "openai", "api_key": "your-api-key-here"}},
        )
        assert cmd_validate(cfg) == 1
        assert "API key not set" in capsys.readouterr().err

    def test_empty_api_key_no_env(self, tmp_path, capsys, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = _write_cfg_yaml(
            tmp_path, {"agent": {"name": "a"}, "llm": {"model": "m", "provider": "openai", "api_key": ""}}
        )
        assert cmd_validate(cfg) == 1
        assert "API key not set" in capsys.readouterr().err

    def test_defensive_branches_via_mocked_config(self, capsys, monkeypatch):
        # agent.name/llm.model are Pydantic-validated at load (can't be empty via
        # from_yaml), so the cmd-level defensive checks (129/131) need a bypass.
        # A mocked Config exercises every issue branch in one shot.
        from types import SimpleNamespace

        fake = SimpleNamespace(
            agent_name="",
            model="",
            provider="weird",
            api_key="sk-xxx",
            rag_enabled=False,
            max_iterations=5,
        )
        monkeypatch.delenv("WEIRD_API_KEY", raising=False)
        with patch("koboi.config.Config.from_yaml", return_value=fake):
            rc = cmd_validate("whatever.yaml")
        assert rc == 1
        err = capsys.readouterr().err
        assert "agent.name is missing" in err
        assert "llm.model is missing" in err
        assert "Unknown provider" in err
        assert "API key not set" in err


# --------------------------------------------------------------------------- #
# cmd_run -- resume + no-message branches
# --------------------------------------------------------------------------- #
class TestRunResume:
    @patch("koboi.facade.KoboiAgent.from_config")
    def test_run_resume_success(self, mock_from_config, tmp_path, capsys):
        agent = MagicMock()
        agent.resume = AsyncMock(return_value="RESUMED OUTPUT")
        mock_from_config.return_value = agent
        rc, out, _ = _invoke(cmd_run, _write_cfg(tmp_path), None, False, False, "abcdef1234567890")
        assert rc == 0
        assert "Resumed" in out
        assert "RESUMED OUTPUT" in out

    @patch("koboi.facade.KoboiAgent.from_config")
    def test_run_resume_error(self, mock_from_config, tmp_path, capsys):
        agent = MagicMock()
        agent.resume = AsyncMock(side_effect=RuntimeError("nope"))
        mock_from_config.return_value = agent
        rc, _, err = _invoke(cmd_run, _write_cfg(tmp_path), None, False, False, "abcdef1234567890")
        assert rc == 1
        assert "Resume error" in err


class TestRunNoMessage:
    @patch("koboi.facade.KoboiAgent.from_config")
    def test_print_mode_empty_stdin(self, mock_from_config, tmp_path):
        mock_from_config.return_value = MagicMock()
        with patch("sys.stdin", io.StringIO("   \n")):
            rc, out, _ = _invoke(cmd_run, _write_cfg(tmp_path), None, False, True, None)
        assert rc == 1
        assert '"type": "error"' in out

    @patch("koboi.facade.KoboiAgent.from_config")
    def test_prompted_empty_message(self, mock_from_config, tmp_path, monkeypatch):
        mock_from_config.return_value = MagicMock()
        monkeypatch.setattr("builtins.input", lambda prompt="": "")
        rc, _, err = _invoke(cmd_run, _write_cfg(tmp_path), None, False, False, None)
        assert rc == 1
        assert "No message provided" in err

    @patch("koboi.facade.KoboiAgent.from_config")
    def test_prompted_eof(self, mock_from_config, tmp_path, monkeypatch):
        mock_from_config.return_value = MagicMock()

        def _eof(prompt=""):
            raise EOFError

        monkeypatch.setattr("builtins.input", _eof)
        rc, _, _ = _invoke(cmd_run, _write_cfg(tmp_path), None, False, False, None)
        assert rc == 1


# --------------------------------------------------------------------------- #
# cmd_chat_print -- success path (227-228)
# --------------------------------------------------------------------------- #
class TestChatPrintSuccess:
    @patch("koboi.facade.KoboiAgent.from_config")
    def test_chat_print_runs_then_eof(self, mock_from_config, tmp_path, monkeypatch):
        agent = MagicMock()
        agent.config.agent_name = "t"
        agent.config.provider = "openai"
        agent.config.model = "m"
        mock_from_config.return_value = agent

        def _eof():
            raise EOFError

        monkeypatch.setattr("builtins.input", _eof)
        rc, out, _ = _invoke(cmd_chat_print, _write_cfg(tmp_path), False)
        assert rc == 0
        assert "session_end" in out


# --------------------------------------------------------------------------- #
# cmd_sessions -- config error + delete branches
# --------------------------------------------------------------------------- #
class TestSessionsMore:
    def test_sessions_bad_config(self, capsys):
        rc = cmd_sessions("/no/such/cfg.yaml", 50)
        assert rc == 1
        assert "Config error" in capsys.readouterr().err

    def test_sessions_delete_found(self, tmp_path, capsys):
        from koboi.memory_sqlite import SQLiteMemory

        db = str(tmp_path / "m.db")
        cfg = _write_cfg_yaml(
            tmp_path,
            {
                "agent": {"name": "s"},
                "llm": {"provider": "openai", "model": "m", "api_key": "t"},
                "memory": {"backend": "sqlite", "db_path": db},
            },
        )
        mem = SQLiteMemory(db_path=db, session_id="sess1234567890")
        mem.add_user_message("hi")
        mem.ensure_session_record(agent_name="s", model="m")
        mem.close()

        rc, out, _ = _invoke(cmd_sessions, cfg, 50, delete="sess1234567890")
        assert rc == 0
        assert "Deleted session" in out

    def test_sessions_delete_not_found(self, tmp_path, capsys):
        from koboi.memory_sqlite import SQLiteMemory

        db = str(tmp_path / "m.db")
        cfg = _write_cfg_yaml(
            tmp_path,
            {
                "agent": {"name": "s"},
                "llm": {"provider": "openai", "model": "m", "api_key": "t"},
                "memory": {"backend": "sqlite", "db_path": db},
            },
        )
        SQLiteMemory(db_path=db, session_id="seed").close()

        rc, out, _ = _invoke(cmd_sessions, cfg, 50, delete="nonexistent")
        assert rc == 0
        assert "not found" in out


# --------------------------------------------------------------------------- #
# cmd_eval -- cases loading + runner
# --------------------------------------------------------------------------- #
class TestEvalRun:
    def test_eval_loads_cases_and_runs(self, tmp_path, capsys):
        cases_file = tmp_path / "cases.yaml"
        cases_file.write_text(yaml.dump({"cases": [{"name": "c1", "user_message": "hi"}]}))
        cfg = _write_cfg(tmp_path)
        result = EvalResult(case_name="c1", output="ok")
        with patch("koboi.eval.runner.EvalRunner") as MockRunner:
            MockRunner.return_value.run_suite = AsyncMock(return_value=[result])
            MockRunner.return_value.format_results.return_value = "RESULTS-TEXT"
            rc = cmd_eval(cfg, str(cases_file))
        assert rc == 0
        assert "RESULTS-TEXT" in capsys.readouterr().out

    def test_eval_runner_error(self, tmp_path, capsys):
        cases_file = tmp_path / "cases.yaml"
        cases_file.write_text(yaml.dump({"cases": [{"name": "c1", "user_message": "hi"}]}))
        cfg = _write_cfg(tmp_path)
        with patch("koboi.eval.runner.EvalRunner") as MockRunner:
            MockRunner.return_value.run_suite = AsyncMock(side_effect=RuntimeError("boom"))
            rc = cmd_eval(cfg, str(cases_file))
        assert rc == 1
        assert "Eval runner error" in capsys.readouterr().err

    def test_eval_cases_file_missing_falls_back(self, tmp_path, capsys):
        cfg = _write_cfg(tmp_path)
        rc = cmd_eval(cfg, str(tmp_path / "nope.yaml"))
        assert rc == 0
        assert "No eval cases" in capsys.readouterr().out

    def test_eval_factory_systemexit_propagates(self, tmp_path, capsys):
        # Real EvalRunner drives the harness_factory; a failing from_config makes
        # factory() print + raise SystemExit, which cmd_eval re-raises (314-318, 325).
        cases_file = tmp_path / "cases.yaml"
        cases_file.write_text(yaml.dump({"cases": [{"name": "c1", "user_message": "hi"}]}))
        cfg = _write_cfg(tmp_path)
        with patch("koboi.facade.KoboiAgent.from_config", side_effect=RuntimeError("no key")):
            with pytest.raises(SystemExit):
                cmd_eval(cfg, str(cases_file))
        assert "Error creating agent for eval" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# cmd_eval_test (350-378)
# --------------------------------------------------------------------------- #
class TestEvalTest:
    def test_real_mock_eval_passes(self):
        rc = cmd_eval_test(str(EVALS / "no_tools.eval.py"), None, True, False, 0.6, False, 5, None)
        assert rc == 0

    def test_no_tests_found(self, capsys):
        with patch("koboi.eval.t.run_tests_sync", return_value=[]):
            rc = cmd_eval_test("evals/", None, True, False, 0.6, False, 5, None)
        assert rc == 2
        assert "No tests found" in capsys.readouterr().out

    def test_discovery_error(self, capsys):
        with patch("koboi.eval.t.run_tests_sync", side_effect=RuntimeError("bad module")):
            rc = cmd_eval_test("evals/", None, True, False, 0.6, False, 5, None)
        assert rc == 2
        assert "eval-test error" in capsys.readouterr().err

    def test_strict_failure(self, capsys):
        failing = EvalResult(case_name="x", output="y", passed=False)
        with patch("koboi.eval.t.run_tests_sync", return_value=[failing]):
            rc = cmd_eval_test("evals/", None, True, True, 0.6, False, 5, None)
        assert rc == 1
        assert "failed" in capsys.readouterr().err

    def test_tags_parsed_and_passed(self):
        passing = EvalResult(case_name="x", output="y", passed=True)
        with patch("koboi.eval.t.run_tests_sync", return_value=[passing]) as m:
            rc = cmd_eval_test("evals/", None, True, False, 0.6, False, 5, "smoke, fast")
        assert rc == 0
        assert m.call_args.kwargs["tags"] == ["smoke", "fast"]


# --------------------------------------------------------------------------- #
# cmd_diagnostics (386-407)
# --------------------------------------------------------------------------- #
class TestDiagnostics:
    @patch("koboi.facade.KoboiAgent.from_config")
    def test_diagnostics_success(self, mock_from_config, tmp_path, capsys, monkeypatch):
        agent = MagicMock()
        agent.close = AsyncMock()
        mock_from_config.return_value = agent
        monkeypatch.chdir(tmp_path)
        with patch("koboi.diagnostics.collect_diagnostics", return_value=b"PK\x03\x04data"):
            rc = cmd_diagnostics(_write_cfg(tmp_path), None)
        assert rc == 0
        assert "Diagnostics exported" in capsys.readouterr().out
        assert list(tmp_path.glob("diagnostics_*.zip"))

    @patch("koboi.facade.KoboiAgent.from_config")
    def test_diagnostics_custom_output(self, mock_from_config, tmp_path, capsys):
        agent = MagicMock()
        agent.close = AsyncMock()
        mock_from_config.return_value = agent
        out_file = tmp_path / "custom.zip"
        with patch("koboi.diagnostics.collect_diagnostics", return_value=b"data"):
            rc = cmd_diagnostics(_write_cfg(tmp_path), str(out_file))
        assert rc == 0
        assert out_file.exists()

    @patch("koboi.facade.KoboiAgent.from_config")
    def test_diagnostics_from_config_error(self, mock_from_config, tmp_path, capsys):
        mock_from_config.side_effect = RuntimeError("no key")
        rc = cmd_diagnostics(_write_cfg(tmp_path), None)
        assert rc == 1
        assert "Error loading agent" in capsys.readouterr().err

    @patch("koboi.facade.KoboiAgent.from_config")
    def test_diagnostics_collect_error(self, mock_from_config, tmp_path, capsys):
        agent = MagicMock()
        agent.close = AsyncMock()
        mock_from_config.return_value = agent
        with patch("koboi.diagnostics.collect_diagnostics", side_effect=RuntimeError("zip fail")):
            rc = cmd_diagnostics(_write_cfg(tmp_path), None)
        assert rc == 1
        assert "Error generating diagnostics" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# cmd_init_zsh (415-446)
# --------------------------------------------------------------------------- #
class TestInitZsh:
    def test_install_to_target(self, tmp_path, capsys):
        target = tmp_path / "plugin"
        rc = cmd_init_zsh(str(target))
        assert rc == 0
        assert (target / "koboi.plugin.zsh").exists()
        assert "source" in capsys.readouterr().out  # non-oh-my-zsh activation hint

    def test_install_oh_my_zsh_path(self, tmp_path, capsys):
        target = tmp_path / "oh-my-zsh" / "custom" / "plugins" / "koboi"
        rc = cmd_init_zsh(str(target))
        assert rc == 0
        assert "plugins=(... koboi)" in capsys.readouterr().out

    def test_install_with_zsh_custom(self, tmp_path, monkeypatch):
        zsh_custom = tmp_path / "custom"
        zsh_custom.mkdir()
        monkeypatch.setenv("ZSH_CUSTOM", str(zsh_custom))
        rc = cmd_init_zsh(None)
        assert rc == 0
        assert (zsh_custom / "plugins" / "koboi" / "koboi.plugin.zsh").exists()

    def test_install_default_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ZSH_CUSTOM", "")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        rc = cmd_init_zsh(None)
        assert rc == 0
        assert (tmp_path / ".zsh" / "koboi" / "koboi.plugin.zsh").exists()

    def test_plugin_source_missing(self, monkeypatch, capsys):
        real_exists = Path.exists

        def fake_exists(self):
            if "koboi.plugin.zsh" in str(self):
                return False
            return real_exists(self)

        monkeypatch.setattr(Path, "exists", fake_exists)
        rc = cmd_init_zsh(None)
        assert rc == 1
        assert "Plugin source not found" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# cli.py -- serve / mcp-serve / keys / chat-interactive / graph dispatch + fallback
# --------------------------------------------------------------------------- #
class TestCliServeDispatch:
    def test_serve_calls_serve_app(self):
        with patch("koboi.server.app.serve_app") as mock_serve:
            code, _, _ = _invoke_cli(["serve", "c.yaml"])
        mock_serve.assert_called_once()
        assert code == 0

    def test_serve_with_host_port(self):
        with patch("koboi.server.app.serve_app") as mock_serve:
            _invoke_cli(["serve", "c.yaml", "--host", "0.0.0.0", "--port", "9000"])
        _, kwargs = mock_serve.call_args
        assert kwargs["host"] == "0.0.0.0"
        assert kwargs["port"] == 9000


class TestCliMcpServe:
    def test_mcp_serve_dispatch(self):
        with patch("koboi.mcp.tool_server.serve_koboi_tools") as mock_serve:
            code, _, _ = _invoke_cli(["mcp-serve", "c.yaml", "--allow", "fs", "--allow-all"])
        mock_serve.assert_called_once_with("c.yaml", allow=["fs"], allow_all=True)
        assert code == 0

    def test_mcp_serve_default_flags(self):
        with patch("koboi.mcp.tool_server.serve_koboi_tools") as mock_serve:
            _invoke_cli(["mcp-serve", "c.yaml"])
        mock_serve.assert_called_once_with("c.yaml", allow=[], allow_all=False)


class TestCliKeys:
    def test_keys_create_list_rotate_revoke(self, tmp_path):
        keys_file = str(tmp_path / "k.json")
        # create
        code, out, _ = _invoke_cli(["keys", "--file", keys_file, "create", "--label", "dev"])
        assert code == 0
        assert "Created key" in out
        # list (one active key)
        code, out, _ = _invoke_cli(["keys", "--file", keys_file, "list"])
        assert code == 0
        assert "key_0001" in out
        assert "active" in out
        # rotate key_0001 -> key_0002
        code, out, _ = _invoke_cli(["keys", "--file", keys_file, "rotate", "key_0001", "--label", "new"])
        assert code == 0
        assert "Rotated" in out
        # revoke key_0002
        code, out, _ = _invoke_cli(["keys", "--file", keys_file, "revoke", "key_0002"])
        assert code == 0
        assert "Revoked" in out
        # list shows revoked
        _, out, _ = _invoke_cli(["keys", "--file", keys_file, "list"])
        assert "REVOKED" in out

    def test_keys_list_empty(self, tmp_path):
        keys_file = str(tmp_path / "k.json")
        code, out, _ = _invoke_cli(["keys", "--file", keys_file, "list"])
        assert code == 0
        assert "No keys found" in out

    def test_keys_subcommand_file_flag(self, tmp_path):
        keys_file = str(tmp_path / "sub.json")
        # file flag on the subcommand (not the parent) also works
        code, out, _ = _invoke_cli(["keys", "create", "--file", keys_file])
        assert code == 0
        assert "Created key" in out
        assert (tmp_path / "sub.json").exists()

    def test_keys_revoke_missing_exits_1(self, tmp_path):
        keys_file = str(tmp_path / "k.json")
        code, _, err = _invoke_cli(["keys", "--file", keys_file, "revoke", "nope"])
        assert code == 1
        assert "Key not found" in err

    def test_keys_rotate_missing_exits_1(self, tmp_path):
        keys_file = str(tmp_path / "k.json")
        code, _, err = _invoke_cli(["keys", "--file", keys_file, "rotate", "nope"])
        assert code == 1
        assert "Key not found" in err


class TestCliChatInteractive:
    def test_chat_interactive_dispatch(self):
        with patch("koboi.tui.app.run_chat_interactive", return_value=0) as mock_run:
            code, _, _ = _invoke_cli(["chat", "c.yaml"])
        mock_run.assert_called_once()
        assert code == 0

    def test_chat_interactive_passes_flags(self):
        with patch("koboi.tui.app.run_chat_interactive", return_value=0) as mock_run:
            _invoke_cli(["chat", "c.yaml", "--no-tui", "--no-stream", "-v"])
        _, kwargs = mock_run.call_args
        assert kwargs["no_tui"] is True
        assert kwargs["no_stream"] is True
        assert kwargs["verbose"] is True


class TestCliGraphDispatch:
    def test_graph_via_main(self):
        with patch("koboi.cli_commands.cmd_graph", return_value=0) as m:
            code, _, _ = _invoke_cli(["graph", "c.yaml", "--format", "json"])
        m.assert_called_once_with("c.yaml", "json")
        assert code == 0


class TestCliFallback:
    def test_unknown_command_fallback_help(self, monkeypatch):
        from koboi import cli

        fake_parser = MagicMock()
        fake_ns = MagicMock()
        fake_ns.command = "totally_unknown"
        fake_parser.parse_args.return_value = fake_ns
        monkeypatch.setattr(cli, "_build_parser", lambda: fake_parser)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli.main()
        fake_parser.print_help.assert_called_once()
