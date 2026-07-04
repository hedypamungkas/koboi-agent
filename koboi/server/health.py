"""koboi/server/health -- HealthCheck registry + built-in checks (/readyz).

Reuses the name->callable registry idiom used elsewhere in the repo
(cf. ``koboi/sandbox/registry.py``). A failing check does not crash ``run_all``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


class HealthRegistry:
    """Ordered registry of async health checks."""

    def __init__(self) -> None:
        self._checks: list[tuple[str, Callable[[], Awaitable[CheckResult]]]] = []

    def register(self, name: str, fn: Callable[[], Awaitable[CheckResult]]) -> None:
        self._checks.append((name, fn))

    def names(self) -> list[str]:
        return [name for name, _ in self._checks]

    async def run_all(self) -> list[CheckResult]:
        results: list[CheckResult] = []
        for name, fn in self._checks:
            try:
                results.append(await fn())
            except Exception as exc:  # a single failing check must not abort readiness
                results.append(CheckResult(name=name, ok=False, detail=f"check error: {exc}"))
        return results


def make_pool_alive_check(pool: Any) -> Callable[[], Awaitable[CheckResult]]:
    async def _check() -> CheckResult:
        alive = not getattr(pool, "_closed", False)
        return CheckResult("pool", ok=alive, detail=f"{len(pool)} active session(s)")

    return _check


def make_db_check(store: Any, *, backend: str = "sqlite") -> Callable[[], Awaitable[CheckResult]]:
    """Probe the sidecar DB connection (``SELECT 1``) instead of just reporting the backend.

    A closed/wedged connection makes readiness fail (503). Honest limitation: this
    probes the ownership sidecar connection (same file as the memory DB in the sqlite
    case), not the memory backend's own connection — a sufficient liveness signal.
    """

    async def _check() -> CheckResult:
        ok = bool(store.ping())
        state = "reachable" if ok else "unreachable"
        return CheckResult("db", ok=ok, detail=f"backend={backend} ({state})")

    return _check
