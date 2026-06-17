"""Tests for koboi/hooks/mode_hook.py -- Mode-aware hook."""
from __future__ import annotations

import pytest

from koboi.hooks.chain import HookContext, HookEvent
from koboi.hooks.mode_hook import ModeHook
from koboi.modes import AgentMode, ModeManager


@pytest.fixture
def mode_hook():
    return ModeHook(ModeManager(AgentMode.CHAT))


class TestModeHook:
    def test_handles(self, mode_hook):
        events = mode_hook.handles()
        assert HookEvent.PRE_INPUT in events
        assert HookEvent.PRE_TOOL_USE in events

    @pytest.mark.asyncio
    async def test_pre_input_injects_suffix(self, mode_hook):
        ctx = HookContext(event=HookEvent.PRE_INPUT)
        result = await mode_hook.execute(ctx)
        assert result.inject_message is not None
        assert "CHAT mode" in result.inject_message

    @pytest.mark.asyncio
    async def test_pre_tool_use_read_only_allowed(self, mode_hook):
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="read")
        result = await mode_hook.execute(ctx)
        assert result.metadata.get("mode_blocked") is None

    @pytest.mark.asyncio
    async def test_pre_tool_use_write_blocked_in_chat(self, mode_hook):
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="write_file")
        result = await mode_hook.execute(ctx)
        assert result.metadata.get("mode_blocked") is True
        assert "CHAT mode" in result.metadata["mode_block_reason"]

    @pytest.mark.asyncio
    async def test_plan_mode_blocks_write(self):
        hook = ModeHook(ModeManager(AgentMode.PLAN))
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="shell.execute")
        result = await hook.execute(ctx)
        assert result.metadata.get("mode_blocked") is True
        assert "PLAN mode" in result.metadata["mode_block_reason"]

    @pytest.mark.asyncio
    async def test_act_mode_allows_all(self):
        hook = ModeHook(ModeManager(AgentMode.ACT))
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="shell.execute")
        result = await hook.execute(ctx)
        assert result.metadata.get("mode_blocked") is None

    @pytest.mark.asyncio
    async def test_auto_mode_allows_all(self):
        hook = ModeHook(ModeManager(AgentMode.AUTO))
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="write_file")
        result = await hook.execute(ctx)
        assert result.metadata.get("mode_blocked") is None

    @pytest.mark.asyncio
    async def test_no_tool_name_passes(self, mode_hook):
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name=None)
        result = await mode_hook.execute(ctx)
        assert result.metadata.get("mode_blocked") is None

    @pytest.mark.asyncio
    async def test_other_event_passes(self, mode_hook):
        ctx = HookContext(event=HookEvent.POST_TOOL_USE)
        result = await mode_hook.execute(ctx)
        assert result is ctx

    def test_is_read_only_exact(self):
        assert ModeHook._is_read_only("read") is True
        assert ModeHook._is_read_only("calculator") is True
        assert ModeHook._is_read_only("web_search") is True
        assert ModeHook._is_read_only("write_file") is False

    def test_is_read_only_prefix(self):
        assert ModeHook._is_read_only("read.file") is True
        assert ModeHook._is_read_only("search.code") is True

    @pytest.mark.asyncio
    async def test_pre_input_plan_suffix(self):
        hook = ModeHook(ModeManager(AgentMode.PLAN))
        ctx = HookContext(event=HookEvent.PRE_INPUT)
        result = await hook.execute(ctx)
        assert "PLAN mode" in result.inject_message

    @pytest.mark.asyncio
    async def test_yolo_mode_allows_all(self):
        hook = ModeHook(ModeManager(AgentMode.YOLO))
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="shell.execute")
        result = await hook.execute(ctx)
        assert result.metadata.get("mode_blocked") is None

    @pytest.mark.asyncio
    async def test_yolo_mode_injects_suffix(self):
        hook = ModeHook(ModeManager(AgentMode.YOLO))
        ctx = HookContext(event=HookEvent.PRE_INPUT)
        result = await hook.execute(ctx)
        assert "YOLO mode" in result.inject_message
