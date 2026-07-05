"""Tests for koboi/cli -- the argparse console-script dispatcher.

Covers: importability without extras, ``--help``, per-command routing to
:mod:`koboi.cli_commands`, graceful refusal when ``[api]``/``[tui]`` extras are
missing, and a bare-install simulation (all extras poisoned -> every no-TUI
command still routes to its core handler without an ImportError).
"""

from __future__ import annotations

import contextlib
import io
from unittest.mock import patch

import pytest


def _invoke(argv: list[str]) -> tuple[int, str, str]:
    """Run cli.main(argv) capturing stdout/stderr; return (exit_code, out, err)."""
    import sys

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


class TestCliMain:
    """Top-level dispatch behaviour."""

    def test_main_importable(self):
        """cli.py is importable without any optional extras."""
        import koboi.cli

        assert koboi.cli is not None

    def test_help_exits_zero(self):
        """``koboi --help`` lists every subcommand and exits 0 (argparse, no extras)."""
        code, out, _ = _invoke(["--help"])
        assert code == 0
        for cmd in (
            "serve",
            "keys",
            "validate",
            "run",
            "chat",
            "sessions",
            "eval",
            "eval-test",
            "diagnostics",
            "init-zsh",
        ):
            assert cmd in out

    def test_no_args_prints_help(self):
        """``koboi`` with no subcommand prints help and exits 0."""
        code, out, _ = _invoke([])
        assert code == 0
        assert "usage:" in out

    def test_validate_routes_to_handler(self):
        """``validate`` dispatches to cli_commands.cmd_validate (not the TUI group)."""
        with patch("koboi.cli_commands.cmd_validate", return_value=0) as m:
            code, _, _ = _invoke(["validate", "some.yaml"])
        assert code == 0
        m.assert_called_once_with("some.yaml")

    def test_run_print_routes_to_handler(self):
        with patch("koboi.cli_commands.cmd_run", return_value=0) as m:
            code, _, _ = _invoke(["run", "c.yaml", "--print", "-m", "hi"])
        assert code == 0
        m.assert_called_once()
        assert m.call_args.args[0] == "c.yaml"

    def test_chat_print_routes_to_handler(self):
        with patch("koboi.cli_commands.cmd_chat_print", return_value=0) as m:
            code, _, _ = _invoke(["chat", "c.yaml", "--print"])
        assert code == 0
        m.assert_called_once()

    def test_sessions_eval_evaltest_initzsh_route_to_handlers(self):
        for argv, attr in [
            (["sessions", "c.yaml"], "cmd_sessions"),
            (["eval", "c.yaml"], "cmd_eval"),
            (["eval-test", "evals/", "--mock"], "cmd_eval_test"),
            (["diagnostics", "c.yaml"], "cmd_diagnostics"),
            (["init-zsh"], "cmd_init_zsh"),
        ]:
            with patch(f"koboi.cli_commands.{attr}", return_value=0) as m:
                code, _, _ = _invoke(argv)
            assert code == 0, f"{argv} did not route"
            m.assert_called_once(), f"{argv} did not call {attr}"


class TestGracefulExtraRefusal:
    """Commands needing extras fail with a clear install hint, not a traceback."""

    def test_chat_interactive_refuses_without_tui(self):
        """Interactive ``chat`` (no --print) exits 1 with a [tui] hint when extras missing."""
        with patch.dict("sys.modules", {"koboi.tui.app": None}):
            code, _, err = _invoke(["chat", "c.yaml"])
        assert code == 1
        assert "pip install koboi-agent[tui]" in err

    def test_serve_refuses_without_api(self):
        """``serve`` exits 1 with an [api] hint when fastapi/uvicorn missing."""
        with patch.dict("sys.modules", {"koboi.server.app": None}):
            code, _, err = _invoke(["serve", "c.yaml"])
        assert code == 1
        assert "pip install koboi-agent[api]" in err

    def test_chat_print_does_not_require_tui(self):
        """``chat --print`` must NOT touch the TUI module (it's core-only)."""
        with patch.dict("sys.modules", {"koboi.tui.app": None}):
            with patch("koboi.cli_commands.cmd_chat_print", return_value=0) as m:
                code, _, _ = _invoke(["chat", "c.yaml", "--print"])
        assert code == 0
        m.assert_called_once()


class TestBareInstallSimulation:
    """Poison ALL optional extras and prove the no-TUI commands still route."""

    POISONED = {
        "click": None,
        "rich": None,
        "rich.console": None,
        "rich.panel": None,
        "rich.table": None,
        "textual": None,
        "fastapi": None,
        "uvicorn": None,
    }

    @pytest.mark.parametrize(
        "argv, attr",
        [
            (["validate", "c.yaml"], "cmd_validate"),
            (["run", "c.yaml", "--print", "-m", "hi"], "cmd_run"),
            (["chat", "c.yaml", "--print"], "cmd_chat_print"),
            (["sessions", "c.yaml"], "cmd_sessions"),
            (["eval-test", "evals/", "--mock"], "cmd_eval_test"),
            (["init-zsh"], "cmd_init_zsh"),
        ],
    )
    def test_no_tui_command_routes_with_extras_missing(self, argv, attr):
        """With every extra poisoned, the command still reaches its core handler."""
        with patch.dict("sys.modules", self.POISONED):
            with patch(f"koboi.cli_commands.{attr}", return_value=0) as m:
                code, _, _ = _invoke(argv)
        assert code == 0, f"{argv} failed with extras missing"
        m.assert_called_once(), f"{argv} did not route to {attr}"

    def test_help_works_with_extras_missing(self):
        with patch.dict("sys.modules", self.POISONED):
            code, out, _ = _invoke(["--help"])
        assert code == 0
        assert "validate" in out
