"""G3: graceful drain -- cancel in-flight interactive streams + flush Langfuse.

Before G3, ``_shutdown`` did not deterministically cancel interactive stream
producers (relied on uvicorn) and never flushed Langfuse (``agent.close`` doesn't
-- the hook flushes on SESSION_END from the loop, which never fires on shutdown).
These tests cover the two extracted capabilities; the ``_shutdown`` wiring
(cancel -> flush -> close) is verified by reading the lifespan drain path.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")
from koboi.config import Config  # noqa: E402
from koboi.server.app import _cancel_tasks  # noqa: E402
from koboi.server.pool import AgentPool  # noqa: E402


def _pool(tmp_path) -> AgentPool:
    cfg = Config.from_dict(
        {"agent": {"name": "t", "system_prompt": "h", "max_iterations": 3}}, validate=False
    )
    return AgentPool(cfg, workspace_root=str(tmp_path))


def _mock_agent_with_langfuse(flush_sink: list) -> MagicMock:
    """A mock pooled agent whose langfuse hook's flush() appends to ``flush_sink``."""
    lf = MagicMock()
    lf.flush = MagicMock(side_effect=lambda: flush_sink.append(1))
    agent = MagicMock()
    agent._core.hooks.find_hook.return_value = lf
    return agent


class TestCancelTasks:
    async def test_cancels_and_clears(self):
        started = asyncio.Event()

        async def blocker():
            started.set()
            await asyncio.sleep(60)

        captured = [asyncio.create_task(blocker()) for _ in range(3)]
        tasks: set[asyncio.Task] = set(captured)
        await started.wait()

        await _cancel_tasks(tasks)

        assert not tasks  # the set is cleared
        assert all(t.cancelled() for t in captured)  # every task was cancelled

    async def test_swallows_task_exception(self):
        # A task that raises (not CancelledError) must not abort the drain.

        async def raiser():
            await asyncio.sleep(0)
            raise RuntimeError("boom")

        tasks: set[asyncio.Task] = {asyncio.create_task(raiser())}
        await asyncio.sleep(0.01)  # let it raise
        await _cancel_tasks(tasks)  # must not re-raise
        assert not tasks


class TestPoolFlushLangfuse:
    async def test_flush_calls_every_agents_hook(self, tmp_path):
        pool = _pool(tmp_path)
        sink: list = []
        pool._agents["s1"] = _mock_agent_with_langfuse(sink)
        pool._agents["s2"] = _mock_agent_with_langfuse(sink)

        await pool.flush_langfuse()

        assert len(sink) == 2  # one flush per pooled agent
        for agent in pool._agents.values():
            agent._core.hooks.find_hook.assert_called()

    async def test_flush_skips_agents_without_langfuse(self, tmp_path):
        pool = _pool(tmp_path)
        agent = MagicMock()
        agent._core.hooks.find_hook.return_value = None  # no langfuse hook
        pool._agents["s1"] = agent

        await pool.flush_langfuse()  # must not raise

    async def test_flush_noop_on_empty_pool(self, tmp_path):
        pool = _pool(tmp_path)
        await pool.flush_langfuse()  # no agents -> no error
