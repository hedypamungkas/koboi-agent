"""Tests for rich_task_hook, notification_hook, rich_subagent_hook."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from koboi.hooks.chain import HookContext, HookEvent
from koboi.hooks.rich_task_hook import RichTaskHook, _TASK_TOOLS
from koboi.hooks.notification_hook import NotificationHook, _EVENT_MESSAGES
from koboi.hooks.rich_subagent_hook import RichSubagentHook


def _make_ctx(event=HookEvent.POST_TOOL_USE, tool_name="task_create", tool_result="task created"):
    ctx = MagicMock(spec=HookContext)
    ctx.event = event
    ctx.tool_name = tool_name
    ctx.tool_result = tool_result
    ctx.metadata = {}
    ctx.llm_response = None
    return ctx


class TestRichTaskHook:
    def test_handles_post_tool_use(self):
        hook = RichTaskHook()
        assert hook.handles() == [HookEvent.POST_TOOL_USE]

    async def test_ignores_non_task_tools(self):
        hook = RichTaskHook(console=MagicMock())
        ctx = _make_ctx(tool_name="calculator")
        result = await hook.execute(ctx)
        hook._console.print.assert_not_called()

    async def test_no_console_does_nothing(self):
        hook = RichTaskHook(console=None)
        ctx = _make_ctx(tool_name="task_create", tool_result="created")
        result = await hook.execute(ctx)
        assert result is ctx

    async def test_task_create_blocked(self):
        console = MagicMock()
        hook = RichTaskHook(console=console)
        ctx = _make_ctx(tool_name="task_create", tool_result="blocked by task 1")
        await hook.execute(ctx)
        console.print.assert_called_once()
        assert "blocked" in console.print.call_args[0][0]

    async def test_task_create_normal(self):
        console = MagicMock()
        hook = RichTaskHook(console=console)
        ctx = _make_ctx(tool_name="task_create", tool_result="task 1 created")
        await hook.execute(ctx)
        console.print.assert_called_once()
        assert "Task created" in console.print.call_args[0][0]

    async def test_task_update_completed(self):
        console = MagicMock()
        hook = RichTaskHook(console=console)
        ctx = _make_ctx(tool_name="task_update", tool_result="task completed")
        await hook.execute(ctx)
        assert "completed" in console.print.call_args[0][0]

    async def test_task_update_in_progress(self):
        console = MagicMock()
        hook = RichTaskHook(console=console)
        ctx = _make_ctx(tool_name="task_update", tool_result="task in_progress")
        await hook.execute(ctx)
        assert "started" in console.print.call_args[0][0]

    async def test_task_update_cannot_start(self):
        console = MagicMock()
        hook = RichTaskHook(console=console)
        ctx = _make_ctx(tool_name="task_update", tool_result="Cannot start task")
        await hook.execute(ctx)
        assert "blocked" in console.print.call_args[0][0]

    async def test_task_update_other(self):
        console = MagicMock()
        hook = RichTaskHook(console=console)
        ctx = _make_ctx(tool_name="task_update", tool_result="some other update")
        await hook.execute(ctx)
        assert "updated" in console.print.call_args[0][0]

    async def test_task_list_ignored(self):
        """task_list is in _TASK_TOOLS but has no special handling."""
        console = MagicMock()
        hook = RichTaskHook(console=console)
        ctx = _make_ctx(tool_name="task_list", tool_result="task list")
        await hook.execute(ctx)
        # task_list has no handler branch, so console.print is not called


class TestNotificationHook:
    def test_handles_default_events(self):
        hook = NotificationHook()
        assert hook.handles() == [HookEvent.POST_OUTPUT]

    def test_handles_custom_events(self):
        hook = NotificationHook(events=[HookEvent.DOOM_LOOP_DETECTED])
        assert hook.handles() == [HookEvent.DOOM_LOOP_DETECTED]

    def test_event_messages_defined(self):
        assert HookEvent.POST_OUTPUT in _EVENT_MESSAGES
        assert HookEvent.DOOM_LOOP_DETECTED in _EVENT_MESSAGES

    @patch("koboi.notifications.notify")
    @patch("koboi.notifications.play_sound")
    async def test_execute_post_output(self, mock_sound, mock_notify):
        hook = NotificationHook()
        ctx = _make_ctx(event=HookEvent.POST_OUTPUT)
        ctx.llm_response = MagicMock()
        ctx.llm_response.content = "Hello world"
        await hook.execute(ctx)
        mock_notify.assert_called_once()

    @patch("koboi.notifications.notify")
    @patch("koboi.notifications.play_sound")
    async def test_execute_doom_loop(self, mock_sound, mock_notify):
        hook = NotificationHook(events=[HookEvent.DOOM_LOOP_DETECTED])
        ctx = _make_ctx(event=HookEvent.DOOM_LOOP_DETECTED)
        await hook.execute(ctx)
        mock_notify.assert_called_once()

    @patch("koboi.notifications.notify")
    @patch("koboi.notifications.play_sound")
    async def test_execute_agent_completed(self, mock_sound, mock_notify):
        hook = NotificationHook(events=[HookEvent.AGENT_COMPLETED])
        ctx = _make_ctx(event=HookEvent.AGENT_COMPLETED)
        ctx.metadata = {"agent_name": "test-agent"}
        await hook.execute(ctx)
        mock_notify.assert_called_once()

    @patch("koboi.notifications.notify")
    @patch("koboi.notifications.play_sound")
    async def test_execute_with_sound(self, mock_sound, mock_notify):
        hook = NotificationHook(sound=True, sound_name="Glass")
        ctx = _make_ctx(event=HookEvent.POST_OUTPUT)
        ctx.llm_response = None
        await hook.execute(ctx)
        mock_sound.assert_called_once_with("Glass")

    @patch("koboi.notifications.notify")
    @patch("koboi.notifications.play_sound")
    async def test_execute_unknown_event(self, mock_sound, mock_notify):
        hook = NotificationHook(events=[HookEvent.SESSION_END])
        ctx = _make_ctx(event=HookEvent.SESSION_END)
        await hook.execute(ctx)
        mock_notify.assert_called_once()


class TestRichSubagentHook:
    def test_handles_dispatch_and_complete(self):
        hook = RichSubagentHook()
        events = hook.handles()
        assert HookEvent.AGENT_DISPATCHED in events
        assert HookEvent.AGENT_COMPLETED in events

    async def test_no_console_does_nothing(self):
        hook = RichSubagentHook(console=None)
        ctx = MagicMock(spec=HookContext)
        ctx.event = HookEvent.AGENT_DISPATCHED
        ctx.metadata = {"subagent_label": "agent1"}
        result = await hook.execute(ctx)
        assert result is ctx

    async def test_no_subagent_label_skips(self):
        hook = RichSubagentHook(console=MagicMock())
        ctx = MagicMock(spec=HookContext)
        ctx.event = HookEvent.AGENT_DISPATCHED
        ctx.metadata = {}
        await hook.execute(ctx)
        hook._console.print.assert_not_called()

    async def test_dispatch_prints(self):
        console = MagicMock()
        hook = RichSubagentHook(console=console)
        ctx = MagicMock(spec=HookContext)
        ctx.event = HookEvent.AGENT_DISPATCHED
        ctx.metadata = {
            "subagent_label": "agent1",
            "subagent_index": 0,
            "subagent_total": 2,
            "subagent_task": "Do something important",
        }
        await hook.execute(ctx)
        console.print.assert_called_once()
        assert "dispatched" in console.print.call_args[0][0]

    async def test_complete_success(self):
        console = MagicMock()
        hook = RichSubagentHook(console=console)
        ctx = MagicMock(spec=HookContext)
        ctx.event = HookEvent.AGENT_COMPLETED
        ctx.metadata = {
            "subagent_label": "agent1",
            "subagent_index": 0,
            "subagent_total": 1,
            "subagent_task": "task",
            "subagent_elapsed": 1.5,
            "subagent_success": True,
        }
        await hook.execute(ctx)
        assert "completed" in console.print.call_args[0][0]

    async def test_complete_failure(self):
        console = MagicMock()
        hook = RichSubagentHook(console=console)
        ctx = MagicMock(spec=HookContext)
        ctx.event = HookEvent.AGENT_COMPLETED
        ctx.metadata = {
            "subagent_label": "agent1",
            "subagent_index": 0,
            "subagent_total": 1,
            "subagent_task": "task",
            "subagent_elapsed": 0.5,
            "subagent_success": False,
            "subagent_error": "timeout",
        }
        await hook.execute(ctx)
        assert "failed" in console.print.call_args[0][0]
