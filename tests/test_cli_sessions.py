"""Tests for the `koboi sessions` CLI subcommand (P2-A).

Invoked via :func:`koboi.cli.main` (the argparse dispatcher) rather than the old
click group, so it exercises the real console-script path.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys

import pytest

from koboi.memory_sqlite import SQLiteMemory


@pytest.fixture(autouse=True)
def _isolate_env():
    """Snapshot/restore os.environ around CLI invocations.

    ``cli.main`` calls ``load_dotenv()``, which would otherwise load the project
    ``.env`` into the process env and pollute env-sensitive tests later.
    """
    saved = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(saved)


def _invoke_sessions(cfg: str) -> tuple[int, str]:
    from koboi import cli

    old = sys.argv
    sys.argv = ["koboi", "sessions", cfg]
    out = io.StringIO()
    code = 0
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            cli.main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    finally:
        sys.argv = old
    return code, out.getvalue()


def _write_config(path, db_path):
    path.write_text(
        f"""
agent:
  name: sessions-agent
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: test
  base_url: http://localhost:8080/v1
memory:
  backend: sqlite
  db_path: {db_path}
"""
    )


class TestSessionsCommand:
    def test_sessions_lists_existing_sessions(self, tmp_path):
        db = str(tmp_path / "mem.db")
        cfg = tmp_path / "cfg.yaml"
        _write_config(cfg, db)

        mem = SQLiteMemory(db_path=db, session_id="abc123def456")
        mem.add_user_message("first question here")
        mem.ensure_session_record(agent_name="sessions-agent", model="gpt-4o-mini")
        mem.close()

        code, output = _invoke_sessions(str(cfg))
        assert code == 0
        assert "abc123def" in output  # session id truncated to 12
        assert "first question here" in output

    def test_sessions_reports_empty(self, tmp_path):
        db = str(tmp_path / "empty.db")
        cfg = tmp_path / "cfg.yaml"
        _write_config(cfg, db)
        SQLiteMemory(db_path=db, session_id="x").close()

        code, output = _invoke_sessions(str(cfg))
        assert code == 0
        assert "No sessions" in output

    def test_sessions_non_sqlite_backend_warns(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            """
agent: {name: x}
llm: {provider: openai, model: gpt-4o-mini, api_key: t, base_url: http://localhost:8080/v1}
memory: {backend: memory}
"""
        )
        code, output = _invoke_sessions(str(cfg))
        assert code == 0
        assert "not sqlite" in output.lower()
