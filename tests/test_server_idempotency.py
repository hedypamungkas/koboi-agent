"""Unit tests for koboi/server/idempotency.py (no FastAPI)."""

from __future__ import annotations

from koboi.server.idempotency import IdempotencyRegistry


class TestIdempotencyRegistry:
    def test_new_key_recorded_returns_true(self):
        reg = IdempotencyRegistry(ttl_seconds=60)
        assert reg.check_and_record("dev:s1:k") is True
        assert len(reg) == 1

    def test_duplicate_key_returns_false(self):
        reg = IdempotencyRegistry(ttl_seconds=60)
        assert reg.check_and_record("dev:s1:k") is True
        assert reg.check_and_record("dev:s1:k") is False

    def test_isolation_by_owner_and_session(self):
        reg = IdempotencyRegistry(ttl_seconds=60)
        assert reg.check_and_record("alice:s1:k") is True
        assert reg.check_and_record("bob:s1:k") is True  # different owner
        assert reg.check_and_record("alice:s2:k") is True  # different session
        assert reg.check_and_record("alice:s1:k") is False  # exact repeat

    def test_ttl_expiry_re_allows_key(self):
        t = [0.0]
        reg = IdempotencyRegistry(ttl_seconds=10.0, clock=lambda: t[0])
        assert reg.check_and_record("dev:s1:k") is True
        assert reg.check_and_record("dev:s1:k") is False  # within window
        t[0] = 20.0  # advance past the TTL
        assert reg.check_and_record("dev:s1:k") is True  # expired → re-allowed
