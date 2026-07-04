"""G4: lock-in tests for three verified-working §19 behaviors that had no test.

The deep-dive found these behaviors are CORRECT (not bugs); these tests guard
them against regression:

* workdir content isolation -- a path in session B's workdir is rejected by
  session A's sandbox (tenant safety).
* TTL-persist-after-evict -- evicting a session keeps its workdir on disk so
  resume/artifacts still work within the TTL window.
* disconnect -> lock release -- cancelling a holder of the per-session lock
  releases it (no orphan lock after a dropped interactive stream).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("fastapi")
from koboi.config import Config  # noqa: E402
from koboi.sandbox.restricted import RestrictedProcessBackend  # noqa: E402
from koboi.server.pool import AgentPool  # noqa: E402


def _pool(tmp_path) -> AgentPool:
    cfg = Config.from_dict(
        {"agent": {"name": "t", "system_prompt": "h", "max_iterations": 3}}, validate=False
    )
    return AgentPool(cfg, workspace_root=str(tmp_path))


class TestWorkdirContentIsolation:
    """§19: two jobs writing report.md must not cross-read."""

    def test_cross_workdir_path_rejected(self, tmp_path):
        a = tmp_path / "A"
        b = tmp_path / "B"
        a.mkdir()
        b.mkdir()
        (b / "secret.txt").write_text("B's private data")
        be_a = RestrictedProcessBackend(workdir=str(a))

        with pytest.raises(PermissionError):
            be_a.validate_path(str(b / "secret.txt"))  # session A can't see session B

    def test_own_workdir_path_accepted(self, tmp_path):
        a = tmp_path / "A"
        a.mkdir()
        be_a = RestrictedProcessBackend(workdir=str(a))
        # A relative path is anchored inside A; an absolute path inside A passes.
        assert be_a.validate_path("report.md").startswith(str(a))
        assert be_a.validate_path(str(a / "report.md")).startswith(str(a))


class TestWorkdirSurvivesEviction:
    """§19: resume after eviction within TTL must still find the workdir."""

    async def test_evict_keeps_workdir_on_disk(self, tmp_path):
        pool = _pool(tmp_path)
        sid = "sess1"
        workdir = pool.workdir_for(sid)
        os.makedirs(workdir, exist_ok=True)  # mimic _build_agent's eager makedirs
        pool._agents[sid] = MagicMock(close=AsyncMock())
        pool._locks[sid] = asyncio.Lock()
        pool._last_used[sid] = time.monotonic()

        await pool.evict(sid)

        assert sid not in pool._agents  # agent evicted
        assert os.path.isdir(workdir)  # but the workdir is NOT deleted (TTL GC owns it)


class TestSessionLockReleasedOnCancel:
    """§19: a dropped interactive stream must release the per-session lock."""

    async def test_lock_released_when_holder_cancelled(self, tmp_path):
        pool = _pool(tmp_path)
        sid = "sess1"
        # Inject so session_lock's get_or_create returns early (no real build).
        pool._agents[sid] = MagicMock()
        pool._locks[sid] = asyncio.Lock()
        pool._last_used[sid] = time.monotonic()

        acquired = asyncio.Event()

        async def hold():
            async with pool.session_lock(sid):
                acquired.set()
                await asyncio.sleep(60)

        task = asyncio.create_task(hold())
        await acquired.wait()
        assert pool._locks[sid].locked()

        task.cancel()  # simulate client disconnect -> _run_agent task cancelled
        with contextlib.suppress(asyncio.CancelledError):
            await task

        assert not pool._locks[sid].locked()  # lock released via async-with __aexit__
