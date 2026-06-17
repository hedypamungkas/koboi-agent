"""Tests for koboi/harness/policy_audit.py -- JSONL audit log."""
from __future__ import annotations

import json

from koboi.harness.policy_audit import PolicyAuditLog, PolicyAuditEntry


class TestPolicyAuditLog:
    def test_log_and_flush(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log = PolicyAuditLog(str(path), buffer_size=2)
        log.log("shell", '{"cmd":"ls"}', "allow", "default", "moderate")
        assert log.pending_count == 1
        log.log("git", '{"cmd":"status"}', "deny", "block_git", "safe")
        assert log.pending_count == 0  # auto-flushed at buffer_size=2
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        data = json.loads(lines[0])
        assert data["tool"] == "shell"
        assert data["decision"] == "allow"
        assert "args_hash" in data

    def test_manual_flush(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log = PolicyAuditLog(str(path), buffer_size=100)
        log.log("tool", "args", "allow", "rule")
        assert log.pending_count == 1
        log.flush()
        assert log.pending_count == 0
        assert path.exists()

    def test_close_flushes(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log = PolicyAuditLog(str(path), buffer_size=100)
        log.log("tool", "args", "allow", "rule")
        log.close()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1

    def test_flush_empty_noop(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log = PolicyAuditLog(str(path))
        log.flush()
        assert not path.exists()

    def test_arguments_hashed(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log = PolicyAuditLog(str(path), buffer_size=1)
        log.log("tool", "sensitive_data_here", "allow", "rule")
        content = path.read_text()
        assert "sensitive_data_here" not in content
        data = json.loads(content.strip())
        assert len(data["args_hash"]) == 16

    def test_multiple_flushes_append(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log = PolicyAuditLog(str(path), buffer_size=1)
        log.log("a", "x", "allow", "r1")
        log.log("b", "y", "deny", "r2")
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
