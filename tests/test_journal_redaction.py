"""Tests for secret redaction in the step journal (issue #8a) + redact module."""

from __future__ import annotations

import json

from koboi.journal import StepJournal
from koboi.memory_sqlite import SQLiteMemory
from koboi.redact import redact_tool_arguments, redact_value
from koboi.types import ToolCall


def _stored_args(db_path, session_id):
    conn = sqlite3_conn(db_path)
    row = conn.execute("SELECT tool_calls_json FROM steps WHERE session_id=?", (session_id,)).fetchone()
    conn.close()
    return row[0] if row else None


def sqlite3_conn(db_path):
    import sqlite3

    return sqlite3.connect(db_path)


class TestRedactModule:
    def test_redacts_sensitive_keys(self):
        out = redact_tool_arguments(json.dumps({"password": "hunter2", "api_key": "sk-live-123", "ok": "keep"}))
        parsed = json.loads(out)
        assert parsed["password"] == "***REDACTED***"
        assert parsed["api_key"] == "***REDACTED***"
        assert parsed["ok"] == "keep"

    def test_redacts_value_shapes_on_non_sensitive_keys(self):
        out = redact_tool_arguments(json.dumps({"note": "token=abc123 and bearer xyz and sk-" + "a" * 24}))
        assert "abc123" not in out
        assert "bearer" not in out.lower() or "***REDACTED***" in out

    def test_falls_back_to_value_redaction_on_non_json(self):
        out = redact_tool_arguments("raw token=secret123 here")
        assert "secret123" not in out

    def test_redact_value_handles_sk_and_akia(self):
        assert "sk-" + "a" * 24 not in redact_value("key: sk-" + "a" * 24)
        assert "AKIA" + "B" * 16 not in redact_value("AKIA" + "B" * 16)

    def test_nested_and_globs(self):
        out = redact_tool_arguments(
            json.dumps(
                {
                    "user": {"USER_PASSWORD": "p", "db": {"AWS_SECRET_ACCESS_KEY": "k"}},
                    "CREDIT_CARD": "4242",
                }
            )
        )
        parsed = json.loads(out)
        assert parsed["user"]["USER_PASSWORD"] == "***REDACTED***"
        assert parsed["user"]["db"]["AWS_SECRET_ACCESS_KEY"] == "***REDACTED***"
        assert parsed["CREDIT_CARD"] == "***REDACTED***"

    def test_empty_input_passthrough(self):
        assert redact_tool_arguments("") == ""


class TestJournalRedaction:
    def test_step_journal_redacts_tool_args(self, tmp_path):
        mem = SQLiteMemory(db_path=str(tmp_path / "t.db"), session_id="S1")
        jr = StepJournal(mem._ensure_conn(), mem.session_id, record_tool_calls=True)
        secret_args = json.dumps(
            {
                "password": "hunter2",
                "api_token": "sk-live-SECRET-1234567890",
                "credit_card": "4242-4242-4242-4242",
                "normal": "visible",
            }
        )
        tc = ToolCall(id="c1", name="charge_card", arguments=secret_args)
        jr.record_step(turn_index=1, step_index=0, status="tool_calls", tool_calls=[tc])

        stored = _stored_args(str(tmp_path / "t.db"), "S1")
        assert stored is not None
        # secrets gone, normal value preserved
        for leak in ("hunter2", "sk-live-SECRET-1234567890", "4242-4242-4242-4242"):
            assert leak not in stored
        assert "visible" in stored
        assert "***REDACTED***" in stored
        # stored JSON still round-trips
        parsed = json.loads(stored)
        assert parsed[0]["name"] == "charge_card"


class TestRedactionSafety:
    def test_deeply_nested_arguments_do_not_crash(self):
        # C2: a pathologically/hallucinatorily nested JSON argument must not
        # RecursionError in the durability-critical journal write path.
        nested: dict = {"password": "x"}
        for _ in range(100):
            nested = {"a": nested}
        out = redact_tool_arguments(json.dumps(nested))  # must not raise
        assert isinstance(out, str)

    def test_redaction_failure_does_not_break_record(self, tmp_path):
        # C2: redaction is fail-safe -- if redact_tool_arguments raises, the step
        # row is still written (arguments masked wholesale), never an exception.
        import koboi.journal as jmod

        mem = SQLiteMemory(db_path=str(tmp_path / "t.db"), session_id="S")
        jr = StepJournal(mem._ensure_conn(), mem.session_id, record_tool_calls=True)

        def boom(_args: str) -> str:
            raise RuntimeError("simulated redaction failure")

        original = jmod.redact_tool_arguments
        jmod.redact_tool_arguments = boom
        try:
            tc = ToolCall(id="c1", name="x", arguments='{"a": 1}')
            jr.record_step(  # must NOT raise
                turn_index=1, step_index=0, status="tool_calls", tool_calls=[tc]
            )
        finally:
            jmod.redact_tool_arguments = original

        row = mem._ensure_conn().execute("SELECT tool_calls_json FROM steps WHERE session_id='S'").fetchone()
        assert row[0] is not None
        assert "***redaction-failed***" in row[0]  # masked, row preserved
