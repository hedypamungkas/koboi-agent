"""Tests for koboi/trust.py -- SQLite-backed trust database."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from koboi.trust import TrustDatabase, TrustDecision, TrustRule
from koboi.types import RiskLevel


@pytest.fixture
def trust_db(tmp_path):
    db_path = str(tmp_path / "test_trust.db")
    db = TrustDatabase(db_path)
    yield db
    db.close()


class TestTrustDatabase:
    def test_schema_creation(self, trust_db):
        """Table should exist after init."""
        rows = trust_db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trust_rules'"
        ).fetchall()
        assert len(rows) == 1

    def test_should_auto_approve_no_rules(self, trust_db):
        result = trust_db.should_auto_approve("shell.execute", RiskLevel.MODERATE)
        assert result.auto_approve is False
        assert "No matching" in result.reason

    def test_record_and_approve(self, trust_db):
        trust_db.record_decision("shell.execute", RiskLevel.MODERATE, "allow", always=True)
        result = trust_db.should_auto_approve("shell.execute", RiskLevel.MODERATE)
        assert result.auto_approve is True
        assert result.matched_rule == "shell.execute"

    def test_record_one_shot_no_rule(self, trust_db):
        trust_db.record_decision("shell.execute", RiskLevel.MODERATE, "allow", always=False)
        result = trust_db.should_auto_approve("shell.execute", RiskLevel.MODERATE)
        assert result.auto_approve is False

    def test_deny_rule(self, trust_db):
        trust_db.record_decision("rm.*", RiskLevel.DESTRUCTIVE, "deny", always=True)
        result = trust_db.should_auto_approve("rm.file", RiskLevel.DESTRUCTIVE)
        assert result.auto_approve is False
        assert "denied" in result.reason.lower() or "Auto-denied" in result.reason

    def test_glob_pattern_matching(self, trust_db):
        trust_db.record_decision("git.*", RiskLevel.SAFE, "allow", always=True)
        result = trust_db.should_auto_approve("git.status", RiskLevel.SAFE)
        assert result.auto_approve is True

    def test_risk_level_hierarchy(self, trust_db):
        """Allow rule for moderate should also cover safe."""
        trust_db.record_decision("tool.*", RiskLevel.MODERATE, "allow", always=True)
        result = trust_db.should_auto_approve("tool.x", RiskLevel.SAFE)
        assert result.auto_approve is True

    def test_risk_level_not_higher(self, trust_db):
        """Allow rule for safe should NOT cover destructive."""
        trust_db.record_decision("tool.*", RiskLevel.SAFE, "allow", always=True)
        result = trust_db.should_auto_approve("tool.x", RiskLevel.DESTRUCTIVE)
        assert result.auto_approve is False

    def test_expired_rule_ignored(self, trust_db):
        trust_db.record_decision("tool.*", RiskLevel.SAFE, "allow", always=True, ttl_seconds=-1)
        result = trust_db.should_auto_approve("tool.x", RiskLevel.SAFE)
        assert result.auto_approve is False

    def test_ttl_rule_not_expired(self, trust_db):
        trust_db.record_decision("tool.*", RiskLevel.SAFE, "allow", always=True, ttl_seconds=3600)
        result = trust_db.should_auto_approve("tool.x", RiskLevel.SAFE)
        assert result.auto_approve is True

    def test_clear_rules_all(self, trust_db):
        trust_db.record_decision("a", RiskLevel.SAFE, "allow", always=True)
        trust_db.record_decision("b", RiskLevel.SAFE, "allow", always=True)
        count = trust_db.clear_rules()
        assert count == 2
        assert len(trust_db.get_rules()) == 0

    def test_clear_rules_by_pattern(self, trust_db):
        trust_db.record_decision("git.*", RiskLevel.SAFE, "allow", always=True)
        trust_db.record_decision("shell.*", RiskLevel.SAFE, "allow", always=True)
        count = trust_db.clear_rules("git.*")
        assert count == 1
        assert len(trust_db.get_rules()) == 1

    def test_get_rules(self, trust_db):
        trust_db.record_decision("tool.a", RiskLevel.SAFE, "allow", always=True)
        trust_db.record_decision("tool.b", RiskLevel.MODERATE, "deny", always=True)
        rules = trust_db.get_rules()
        assert len(rules) == 2
        assert all(isinstance(r, TrustRule) for r in rules)

    def test_get_rules_excludes_expired(self, trust_db):
        trust_db.record_decision("tool.a", RiskLevel.SAFE, "allow", always=True)
        trust_db.record_decision("tool.b", RiskLevel.SAFE, "allow", always=True, ttl_seconds=-1)
        rules = trust_db.get_rules()
        assert len(rules) == 1

    def test_risk_leq_static(self):
        assert TrustDatabase._risk_leq("safe", "moderate") is True
        assert TrustDatabase._risk_leq("safe", "destructive") is True
        assert TrustDatabase._risk_leq("moderate", "safe") is False
        assert TrustDatabase._risk_leq("destructive", "destructive") is True
