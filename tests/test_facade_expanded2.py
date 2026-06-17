"""Tests for facade.py uncovered paths -- on(), add_hook()."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from koboi.facade import KoboiAgent
from koboi.hooks.chain import HookEvent


class TestFacadeOnEvent:
    def test_on_valid_event(self):
        agent = KoboiAgent(core=MagicMock())
        callback = MagicMock()
        agent.on("pre_tool_use", callback)

    def test_on_list_of_events(self):
        agent = KoboiAgent(core=MagicMock())
        callback = MagicMock()
        agent.on(["pre_tool_use", "post_tool_use"], callback)

    def test_on_invalid_event_raises(self):
        agent = KoboiAgent(core=MagicMock())
        with pytest.raises(ValueError, match="Unknown event"):
            agent.on("nonexistent_event", MagicMock())

    def test_on_with_hook_event_enum(self):
        agent = KoboiAgent(core=MagicMock())
        agent.on(HookEvent.PRE_TOOL_USE, MagicMock())


class TestFacadeAddHook:
    def test_add_hook(self):
        agent = KoboiAgent(core=MagicMock())
        hook = MagicMock()
        agent.add_hook(hook)


class TestFacadeRunSync:
    def test_run_sync_basic(self):
        core = MagicMock()
        core.run = AsyncMock(return_value=MagicMock(content="answer"))
        core.hooks = MagicMock()
        core.memory = MagicMock()
        agent = KoboiAgent(core=core)
        result = agent.run_sync("hello")
        assert result.content == "answer"


class TestFacadeContextManager:
    async def test_async_context_manager(self):
        core = MagicMock()
        core.hooks = MagicMock()
        core.hooks.emit = AsyncMock()
        core.memory = MagicMock()
        core.client = MagicMock()
        core.client.close = AsyncMock()
        agent = KoboiAgent(core=core)
        async with agent as a:
            assert a is agent


class TestFacadeRunStream:
    async def test_run_stream(self):
        core = MagicMock()
        core.hooks = MagicMock()
        core.hooks.emit = AsyncMock()
        core.memory = MagicMock()
        core.client = MagicMock()
        core.client.close = AsyncMock()

        async def mock_stream(msg):
            yield MagicMock()
            yield MagicMock()

        core.run_stream = mock_stream
        agent = KoboiAgent(core=core)
        events = []
        async for event in agent.run_stream("hi"):
            events.append(event)
        assert len(events) == 2

    async def test_run_stream_with_orchestrator(self):
        core = MagicMock()
        core.hooks = MagicMock()
        core.hooks.emit = AsyncMock()
        core.memory = MagicMock()
        core.client = MagicMock()
        core.client.close = AsyncMock()

        orchestrator = MagicMock()

        async def mock_orch_stream(msg):
            yield MagicMock()

        orchestrator.run_stream = mock_orch_stream
        agent = KoboiAgent(core=core, orchestrator=orchestrator)
        events = []
        async for event in agent.run_stream("hi"):
            events.append(event)
        assert len(events) == 1


class TestFacadeChat:
    async def test_chat(self):
        core = MagicMock()
        core.chat = AsyncMock(return_value=MagicMock(content="hello"))
        core.hooks = MagicMock()
        core.hooks.emit = AsyncMock()
        core.memory = MagicMock()
        core.client = MagicMock()
        core.client.close = AsyncMock()
        agent = KoboiAgent(core=core)
        result = await agent.chat("hi")
        assert result.content == "hello"

    async def test_chat_with_orchestrator(self):
        core = MagicMock()
        core.hooks = MagicMock()
        core.hooks.emit = AsyncMock()
        core.memory = MagicMock()
        core.client = MagicMock()
        core.client.close = AsyncMock()

        orchestrator = MagicMock()
        orch_result = MagicMock()
        orch_result.final_answer = "orchestrated"
        orch_result.agent_results = [MagicMock(tokens_used=100)]
        orch_result.routing = MagicMock(method="keyword", confidence=0.9)
        orch_result.execution_mode = "sequential"
        orchestrator.run = AsyncMock(return_value=orch_result)

        agent = KoboiAgent(core=core, orchestrator=orchestrator)
        result = await agent.chat("hi")
        assert result.content == "orchestrated"


class TestFacadeClose:
    async def test_close_mcp_clients(self):
        core = MagicMock()
        core.hooks = MagicMock()
        core.hooks.emit = AsyncMock()
        core.memory = MagicMock()
        core.client = MagicMock()
        core.client.close = AsyncMock()

        mcp_client = MagicMock()
        agent = KoboiAgent(core=core, mcp_clients=[mcp_client])
        await agent.close()
        mcp_client.close.assert_called_once()

    async def test_close_with_orchestrator(self):
        core = MagicMock()
        core.hooks = MagicMock()
        core.hooks.emit = AsyncMock()
        core.memory = MagicMock()
        core.client = MagicMock()
        core.client.close = AsyncMock()

        orchestrator = MagicMock()
        orchestrator._agents_map = {}
        orchestrator.client = MagicMock()
        orchestrator.client.close = AsyncMock()

        agent = KoboiAgent(core=core, orchestrator=orchestrator)
        await agent.close()
        orchestrator.client.close.assert_called_once()

    async def test_close_cleans_bg_loop(self):
        core = MagicMock()
        core.hooks = MagicMock()
        core.hooks.emit = AsyncMock()
        core.memory = MagicMock()
        core.client = MagicMock()
        core.client.close = AsyncMock()

        agent = KoboiAgent(core=core)
        # Simulate background loop state
        bg_loop = MagicMock()
        bg_thread = MagicMock()
        bg_thread.is_alive.return_value = True
        agent._bg_loop = bg_loop
        agent._bg_thread = bg_thread
        await agent.close()
        bg_loop.call_soon_threadsafe.assert_called_once_with(bg_loop.stop)
        bg_thread.join.assert_called_once()

    async def test_close_closes_logger(self):
        core = MagicMock()
        core.client = MagicMock()
        core.client.close = AsyncMock()
        core.memory = MagicMock()

        logger = MagicMock()
        agent = KoboiAgent(core=core, logger=logger)
        await agent.close()
        logger.close.assert_called_once()


class TestFacadeDel:
    def test_del_closes_mcp_clients(self):
        core = MagicMock()
        mcp_client = MagicMock()
        agent = KoboiAgent(core=core, mcp_clients=[mcp_client])
        agent.__del__()
        mcp_client.close.assert_called_once()

    def test_del_closes_logger(self):
        core = MagicMock()
        logger = MagicMock()
        agent = KoboiAgent(core=core, logger=logger)
        agent.__del__()
        logger.close.assert_called_once()

    def test_del_stops_bg_loop(self):
        core = MagicMock()
        agent = KoboiAgent(core=core)
        bg_loop = MagicMock()
        bg_thread = MagicMock()
        agent._bg_loop = bg_loop
        agent._bg_thread = bg_thread
        agent.__del__()
        bg_loop.call_soon_threadsafe.assert_called_once_with(bg_loop.stop)
        bg_thread.join.assert_called_once()
        bg_loop.close.assert_called_once()

    def test_del_no_raise_without_close(self):
        """__del__ should not raise even if resources are already gone."""
        agent = KoboiAgent(core=None)
        agent.__del__()  # should not raise


class TestFacadeReset:
    def test_reset(self):
        core = MagicMock()
        core.hooks = MagicMock()
        core.memory = MagicMock()
        agent = KoboiAgent(core=core)
        agent.reset()
        core.reset.assert_called_once()

    def test_reset_no_core(self):
        agent = KoboiAgent(core=None)
        agent.reset()  # should not raise
