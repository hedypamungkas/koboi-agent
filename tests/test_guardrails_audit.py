"""Tests for koboi/guardrails/audit.py -- Audit trail and approval."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from koboi.guardrails.audit import AuditTrail
from koboi.guardrails.approval import (
    ApprovalHandler,
    CallbackApprovalHandler,
    CLIApprovalHandler,
    _risk_color,
)
from koboi.types import AuditEntry, RiskLevel


def _entry(**kwargs):
    defaults = dict(timestamp=time.time(), event_type="test", details="")
    defaults.update(kwargs)
    return AuditEntry(**defaults)


class TestAuditTrail:
    def test_record_and_get(self):
        trail = AuditTrail()
        entry = _entry(
            event_type="tool_call",
            tool_name="shell",
            risk_level="moderate",
            details="executed ls",
            result="ok",
        )
        trail.record(entry)
        entries = trail.get_entries()
        assert len(entries) == 1
        assert entries[0].tool_name == "shell"

    def test_filter_by_event_type(self):
        trail = AuditTrail()
        trail.record(_entry(event_type="input_check", details="passed"))
        trail.record(_entry(event_type="tool_call", details="ran"))
        assert len(trail.get_entries(event_type="input_check")) == 1
        assert len(trail.get_entries(event_type="tool_call")) == 1

    def test_filter_by_tool_name(self):
        trail = AuditTrail()
        trail.record(_entry(event_type="tool_call", tool_name="git", details="d"))
        trail.record(_entry(event_type="tool_call", tool_name="shell", details="d"))
        assert len(trail.get_entries(tool_name="git")) == 1

    def test_summary(self):
        trail = AuditTrail()
        trail.record(_entry(event_type="input_check", details="passed"))
        trail.record(_entry(event_type="tool_call", tool_name="shell", details="denied", result="blocked"))
        s = trail.summary()
        assert s["total_events"] == 2
        assert s["blocked"] >= 1
        assert "input_check" in s["by_type"]

    def test_logger_called(self):
        logger = MagicMock()
        trail = AuditTrail(logger=logger)
        trail.record(_entry(event_type="test", details="info"))
        logger.log.assert_called_once()

    def test_empty_summary(self):
        trail = AuditTrail()
        s = trail.summary()
        assert s["total_events"] == 0
        assert s["blocked"] == 0


class TestRiskColor:
    def test_safe(self):
        assert _risk_color(RiskLevel.SAFE) == "green"

    def test_moderate(self):
        assert _risk_color(RiskLevel.MODERATE) == "yellow"

    def test_destructive(self):
        assert _risk_color(RiskLevel.DESTRUCTIVE) == "red"


class TestApprovalHandler:
    def test_allows_safe(self):
        handler = ApprovalHandler()
        assert handler.should_approve("read", "{}", RiskLevel.SAFE) is True

    def test_allows_moderate(self):
        handler = ApprovalHandler()
        assert handler.should_approve("write", "{}", RiskLevel.MODERATE) is True

    def test_denies_destructive(self):
        handler = ApprovalHandler()
        assert handler.should_approve("rm", "{}", RiskLevel.DESTRUCTIVE) is False


class TestCallbackApprovalHandler:
    def test_callback_approve(self):
        callback = MagicMock(return_value=True)
        handler = CallbackApprovalHandler(callback)
        assert handler.should_approve("tool", "{}", RiskLevel.DESTRUCTIVE) is True
        callback.assert_called_once()

    def test_callback_deny(self):
        callback = MagicMock(return_value=False)
        handler = CallbackApprovalHandler(callback)
        assert handler.should_approve("tool", "{}", RiskLevel.SAFE) is False

    def test_with_audit_trail(self):
        callback = MagicMock(return_value=True)
        trail = MagicMock()
        handler = CallbackApprovalHandler(callback, audit_trail=trail)
        handler.should_approve("tool", "{}", RiskLevel.SAFE)
        trail.record.assert_called_once()


class TestCLIApprovalHandler:
    def test_auto_approve_non_destructive(self):
        handler = CLIApprovalHandler()
        assert handler.should_approve("read", "{}", RiskLevel.SAFE) is True

    def test_auto_approve_moderate(self):
        handler = CLIApprovalHandler(require_for={"destructive"})
        assert handler.should_approve("write", "{}", RiskLevel.MODERATE) is True

    @patch("koboi.guardrails.approval.Confirm.ask", return_value=True)
    def test_prompt_approve(self, mock_ask):
        handler = CLIApprovalHandler()
        result = handler.should_approve("rm", "{}", RiskLevel.DESTRUCTIVE)
        assert result is True
        mock_ask.assert_called_once()

    @patch("koboi.guardrails.approval.Confirm.ask", return_value=False)
    def test_prompt_deny(self, mock_ask):
        handler = CLIApprovalHandler()
        result = handler.should_approve("rm", "{}", RiskLevel.DESTRUCTIVE)
        assert result is False

    @patch("koboi.guardrails.approval.Confirm.ask", side_effect=EOFError)
    def test_eof_denies(self, mock_ask):
        handler = CLIApprovalHandler()
        result = handler.should_approve("rm", "{}", RiskLevel.DESTRUCTIVE)
        assert result is False

    @patch("koboi.guardrails.approval.Confirm.ask", return_value=True)
    def test_with_audit_trail(self, mock_ask):
        trail = MagicMock()
        handler = CLIApprovalHandler(audit_trail=trail)
        handler.should_approve("rm", "{}", RiskLevel.DESTRUCTIVE)
        trail.record.assert_called_once()


class TestSQLiteAuditTrail:
    def test_record_persists(self, tmp_path):
        from koboi.guardrails.audit import SQLiteAuditTrail

        db_path = str(tmp_path / "test_audit.db")
        trail = SQLiteAuditTrail(db_path=db_path)
        trail.record(_entry(event_type="tool_call", tool_name="shell", details="ran"))
        trail.close()
        import sqlite3

        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT event_type, tool_name FROM audit_entries").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0] == ("tool_call", "shell")

    def test_query_db_filters(self, tmp_path):
        from koboi.guardrails.audit import SQLiteAuditTrail

        db_path = str(tmp_path / "test_audit.db")
        trail = SQLiteAuditTrail(db_path=db_path)
        trail.record(_entry(event_type="input_check", details="passed"))
        trail.record(_entry(event_type="tool_call", tool_name="git", details="ran"))
        trail.record(_entry(event_type="tool_call", tool_name="shell", details="ran"))
        result = trail.query_db(event_type="tool_call")
        assert len(result) == 2
        result = trail.query_db(tool_name="git")
        assert len(result) == 1
        trail.close()

    def test_in_memory_list_still_populated(self, tmp_path):
        from koboi.guardrails.audit import SQLiteAuditTrail

        db_path = str(tmp_path / "test_audit.db")
        trail = SQLiteAuditTrail(db_path=db_path)
        trail.record(_entry(event_type="test", details="info"))
        entries = trail.get_entries()
        assert len(entries) == 1
        trail.close()

    def test_schema_indexes(self, tmp_path):
        from koboi.guardrails.audit import SQLiteAuditTrail

        db_path = str(tmp_path / "test_audit.db")
        trail = SQLiteAuditTrail(db_path=db_path)
        trail.close()
        import sqlite3

        conn = sqlite3.connect(db_path)
        indexes = conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        conn.close()
        names = {r[0] for r in indexes}
        assert "idx_audit_event" in names
        assert "idx_audit_tool" in names
        assert "idx_audit_ts" in names

    def test_close_releases_connection(self, tmp_path):
        from koboi.guardrails.audit import SQLiteAuditTrail

        db_path = str(tmp_path / "test_audit.db")
        trail = SQLiteAuditTrail(db_path=db_path)
        trail.close()
        assert trail._conn is None

    def test_memory_list_capped(self, tmp_path):
        from koboi.guardrails.audit import SQLiteAuditTrail

        db_path = str(tmp_path / "test_audit.db")
        trail = SQLiteAuditTrail(db_path=db_path)
        cap = trail._memory_cap
        for i in range(cap + 50):
            trail.record(_entry(event_type="test", details=f"entry {i}"))
        assert len(trail._entries) <= cap
        # Verify all entries are still in SQLite
        db_entries = trail.query_db(event_type="test", limit=cap + 50)
        assert len(db_entries) == cap + 50
        trail.close()

    def test_risk_level_serialized(self, tmp_path):
        from koboi.guardrails.audit import SQLiteAuditTrail

        db_path = str(tmp_path / "test_audit.db")
        trail = SQLiteAuditTrail(db_path=db_path)
        trail.record(_entry(event_type="tool_call", risk_level=RiskLevel.DESTRUCTIVE))
        result = trail.query_db()
        assert result[0]["risk_level"] == "destructive"
        trail.close()
