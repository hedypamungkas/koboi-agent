"""Tests for the `koboi sessions` CLI subcommand (P2-A)."""

from __future__ import annotations

import os

import pytest
from click.testing import CliRunner

from koboi.memory_sqlite import SQLiteMemory
from koboi.tui.app import main


@pytest.fixture(autouse=True)
def _isolate_env():
    """Snapshot/restore os.environ around CLI invocations.

    The click ``main`` group calls ``load_dotenv()``, which would otherwise
    load the project ``.env`` into the process env and pollute env-sensitive
    tests (client key validation, tracing) that run later in the suite.
    """
    saved = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(saved)


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
        # A session row only exists once recorded (mirrors facade.build_memory).
        mem.ensure_session_record(agent_name="sessions-agent", model="gpt-4o-mini")
        mem.close()

        runner = CliRunner()
        result = runner.invoke(main, ["sessions", str(cfg)])
        assert result.exit_code == 0
        assert "abc123def" in result.output  # session id truncated to 12
        assert "first question here" in result.output

    def test_sessions_reports_empty(self, tmp_path):
        db = str(tmp_path / "empty.db")
        cfg = tmp_path / "cfg.yaml"
        _write_config(cfg, db)
        # Touch the DB so it exists but has no sessions.
        SQLiteMemory(db_path=db, session_id="x").close()

        runner = CliRunner()
        result = runner.invoke(main, ["sessions", str(cfg)])
        assert result.exit_code == 0
        assert "No sessions" in result.output

    def test_sessions_non_sqlite_backend_warns(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            """
agent: {name: x}
llm: {provider: openai, model: gpt-4o-mini, api_key: t, base_url: http://localhost:8080/v1}
memory: {backend: memory}
"""
        )
        runner = CliRunner()
        result = runner.invoke(main, ["sessions", str(cfg)])
        assert result.exit_code == 0
        assert "not sqlite" in result.output.lower()
