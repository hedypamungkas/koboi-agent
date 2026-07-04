"""Tests for koboi/trust.py -- SQLite-backed trust database."""

from __future__ import annotations


import pytest

from koboi.trust import TrustDatabase, TrustRule
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

    def test_default_ttl_applied(self, trust_db):
        # H5: recording without an explicit ttl gets the default (no permanent rules).
        from koboi.trust import DEFAULT_TRUST_TTL_SECONDS

        trust_db.record_decision("tool.*", RiskLevel.SAFE, "allow", always=True)
        rows = list(trust_db._conn.execute("SELECT expires_at, created_at FROM trust_rules"))
        assert rows and rows[0][0] is not None
        assert abs((rows[0][0] - rows[0][1]) - DEFAULT_TRUST_TTL_SECONDS) < 5

    def test_args_scoped_allow_does_not_match_other_args(self, trust_db):
        # H5: an arg-scoped allow rule matches only its exact arguments.
        trust_db.record_decision(
            "write_file", RiskLevel.DESTRUCTIVE, "allow", always=True, arguments='{"path":"/tmp/x"}'
        )
        assert (
            trust_db.should_auto_approve("write_file", RiskLevel.DESTRUCTIVE, '{"path":"/tmp/x"}').auto_approve is True
        )
        assert (
            trust_db.should_auto_approve("write_file", RiskLevel.DESTRUCTIVE, '{"path":"/etc/passwd"}').auto_approve
            is False
        )

    def test_args_wildcard_backcompat(self, trust_db):
        # H5: a rule recorded without args (NULL args_hash) matches any arguments.
        trust_db.record_decision("write_file", RiskLevel.DESTRUCTIVE, "allow", always=True)
        assert (
            trust_db.should_auto_approve("write_file", RiskLevel.DESTRUCTIVE, '{"path":"/etc/passwd"}').auto_approve
            is True
        )

    def test_args_hash_exact_match(self, trust_db):
        trust_db.record_decision("t", RiskLevel.SAFE, "allow", always=True, arguments="abc")
        assert trust_db.should_auto_approve("t", RiskLevel.SAFE, "abc").auto_approve is True


class TestTrustStoreProtocol:
    """TrustStore Protocol (M0) -- structural surface the pipeline consumes."""

    def test_trust_database_satisfies_protocol(self, trust_db):
        # TrustDatabase exposes the two methods TrustStore declares.
        assert callable(getattr(trust_db, "should_auto_approve", None))
        assert callable(getattr(trust_db, "record_decision", None))

    def test_fake_store_is_usable(self):
        # A duck-typed store (no SQLite) satisfies TrustStore structurally;
        # this is the seam a future multi-tenant/Redis store will plug into.
        from koboi.trust import TrustDecision
        from koboi.types import RiskLevel

        class _FakeStore:
            def should_auto_approve(self, tool_name, risk_level):
                return TrustDecision(auto_approve=False)

            def record_decision(self, tool_name, risk_level, decision, always=False, ttl_seconds=None):
                pass

        fake = _FakeStore()
        assert fake.should_auto_approve("x", RiskLevel.SAFE).auto_approve is False
