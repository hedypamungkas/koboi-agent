"""Unit tests for koboi/server/health.py (no FastAPI)."""

from __future__ import annotations

from koboi.server.health import CheckResult, HealthRegistry, make_db_check, make_pool_alive_check


class _FakeDbStore:
    """Stand-in for a store exposing ``.ping()`` (cf. ``OwnershipStore.ping``)."""

    def __init__(self, alive: bool) -> None:
        self._alive = alive

    def ping(self) -> bool:
        return self._alive


def _ok(name: str):
    async def _check() -> CheckResult:
        return CheckResult(name=name, ok=True, detail="fine")

    return _check


class TestHealthRegistry:
    async def test_register_and_run_all_in_order(self):
        reg = HealthRegistry()
        reg.register("a", _ok("a"))
        reg.register("b", _ok("b"))
        results = await reg.run_all()
        assert [r.name for r in results] == ["a", "b"]
        assert all(r.ok for r in results)

    async def test_failing_check_does_not_crash_run_all(self):
        reg = HealthRegistry()

        async def boom() -> CheckResult:
            raise RuntimeError("nope")

        reg.register("boom", boom)
        reg.register("ok", _ok("ok"))
        results = await reg.run_all()
        assert results[0].ok is False
        assert "nope" in results[0].detail
        assert results[1].ok is True

    async def test_pool_alive_check_reports_count(self):
        class _FakePool:
            _closed = False

            def __len__(self) -> int:
                return 3

        result = await make_pool_alive_check(_FakePool())()
        assert result.ok is True
        assert "3" in result.detail

    async def test_pool_alive_false_after_close(self):
        class _FakePool:
            _closed = True

            def __len__(self) -> int:
                return 0

        result = await make_pool_alive_check(_FakePool())()
        assert result.ok is False

    async def test_db_check_ok_when_reachable(self):
        result = await make_db_check(_FakeDbStore(alive=True), backend="sqlite")()
        assert result.ok is True
        assert "sqlite" in result.detail
        assert "reachable" in result.detail

    async def test_db_check_false_when_unreachable(self):
        result = await make_db_check(_FakeDbStore(alive=False), backend="sqlite")()
        assert result.ok is False
        assert "unreachable" in result.detail
