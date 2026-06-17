"""Tests for koboi.hooks.carryover_hook module."""
from __future__ import annotations

from koboi.hooks.chain import HookContext, HookEvent
from koboi.hooks.carryover_hook import CarryoverHook
from koboi.harness.carryover import CarryoverState


class TestCarryoverHookHandles:
    def test_handles_subscribes_to_correct_events(self):
        hook = CarryoverHook()
        events = hook.handles()
        assert HookEvent.SESSION_START in events
        assert HookEvent.SESSION_END in events
        assert HookEvent.POST_TOOL_USE in events
        assert HookEvent.POST_COMPACT in events

    def test_ignores_unrelated_events(self):
        hook = CarryoverHook()
        events = hook.handles()
        assert HookEvent.PRE_INPUT not in events
        assert HookEvent.PRE_LLM_CALL not in events


class TestCarryoverHookSessionStart:
    async def test_session_start_attaches_state_to_context(self):
        state = CarryoverState()
        hook = CarryoverHook(state=state)
        ctx = HookContext(event=HookEvent.SESSION_START)
        assert ctx.carryover is None

        result = await hook.execute(ctx)
        assert result.carryover is state

    async def test_session_start_attaches_default_state_when_none_provided(self):
        hook = CarryoverHook()
        ctx = HookContext(event=HookEvent.SESSION_START)
        result = await hook.execute(ctx)
        assert isinstance(result.carryover, CarryoverState)

    async def test_session_start_preserves_existing_carryover(self):
        existing_state = CarryoverState()
        existing_state.add_goal("test goal")
        hook = CarryoverHook()
        ctx = HookContext(event=HookEvent.SESSION_START, carryover=existing_state)
        result = await hook.execute(ctx)
        assert result.carryover is hook.state


class TestCarryoverHookPostToolUse:
    async def test_post_tool_use_records_tool_usage(self):
        hook = CarryoverHook()
        ctx = HookContext(
            event=HookEvent.POST_TOOL_USE,
            tool_name="get_weather",
            tool_arguments='{"city": "Jakarta"}',
            tool_result="Sunny, 28C",
            iteration=2,
        )
        await hook.execute(ctx)

        summary = hook.state.summary()
        assert summary["tool_calls"] == 1
        assert summary["unique_tools"] == 1
        assert "get_weather" in hook.state.invoked_tools
        assert hook.state.invoked_tools["get_weather"] == 1

    async def test_post_tool_use_records_failed_tool(self):
        hook = CarryoverHook()
        ctx = HookContext(
            event=HookEvent.POST_TOOL_USE,
            tool_name="delete_file",
            tool_arguments='{"path": "/tmp/x"}',
            tool_result="Error: file not found",
            iteration=1,
        )
        await hook.execute(ctx)

        assert hook.state.invoked_tools["delete_file"] == 1
        assert len(hook.state.work_log) == 1
        assert hook.state.work_log[0].success is False

    async def test_post_tool_use_records_success_when_no_error(self):
        hook = CarryoverHook()
        ctx = HookContext(
            event=HookEvent.POST_TOOL_USE,
            tool_name="read_file",
            tool_arguments='{"path": "/tmp/a"}',
            tool_result="File contents here",
            iteration=0,
        )
        await hook.execute(ctx)

        assert len(hook.state.work_log) == 1
        assert hook.state.work_log[0].success is True

    async def test_post_tool_use_skips_when_no_tool_name(self):
        hook = CarryoverHook()
        ctx = HookContext(
            event=HookEvent.POST_TOOL_USE,
            tool_name=None,
            iteration=0,
        )
        await hook.execute(ctx)
        assert hook.state.summary()["tool_calls"] == 0

    async def test_post_tool_use_multiple_calls_accumulate(self):
        hook = CarryoverHook()

        for i in range(3):
            ctx = HookContext(
                event=HookEvent.POST_TOOL_USE,
                tool_name="search",
                tool_arguments=f'{{"q": "query{i}"}}',
                tool_result=f"result {i}",
                iteration=i,
            )
            await hook.execute(ctx)

        assert hook.state.invoked_tools["search"] == 3
        assert len(hook.state.work_log) == 3


class TestCarryoverHookPostCompact:
    async def test_post_compact_injects_carryover_message(self):
        state = CarryoverState()
        state.add_goal("Deploy application")
        state.record_tool_use("git_push", "{}", "ok", iteration=0)

        hook = CarryoverHook(state=state)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Deploy it"},
        ]
        ctx = HookContext(
            event=HookEvent.POST_COMPACT,
            messages=messages,
            iteration=1,
        )
        result = await hook.execute(ctx)

        assert len(result.messages) == 3
        injected = result.messages[1]
        assert injected["role"] == "system"
        assert "<harness-carryover>" in injected["content"]
        assert "Deploy application" in injected["content"]

    async def test_post_compact_does_not_duplicate_carryover(self):
        state = CarryoverState()
        state.add_goal("Build project")

        hook = CarryoverHook(state=state)
        carryover_msg = state.to_context_message()
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "system", "content": carryover_msg},
            {"role": "user", "content": "Build it"},
        ]
        ctx = HookContext(
            event=HookEvent.POST_COMPACT,
            messages=messages,
            iteration=0,
        )
        result = await hook.execute(ctx)

        carryover_msgs = [
            m for m in result.messages
            if m.get("role") == "system" and "<harness-carryover>" in m.get("content", "")
        ]
        assert len(carryover_msgs) == 1

    async def test_post_compact_no_injection_when_empty_state(self):
        hook = CarryoverHook()
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
        ]
        ctx = HookContext(
            event=HookEvent.POST_COMPACT,
            messages=messages,
            iteration=0,
        )
        result = await hook.execute(ctx)

        assert len(result.messages) == 2

    async def test_post_compact_no_injection_when_messages_is_none(self):
        state = CarryoverState()
        state.add_goal("Test goal")
        hook = CarryoverHook(state=state)
        ctx = HookContext(
            event=HookEvent.POST_COMPACT,
            messages=None,
            iteration=0,
        )
        result = await hook.execute(ctx)
        assert result.messages is None


class TestCarryoverHookSessionEnd:
    async def test_session_end_sets_summary_metadata(self):
        state = CarryoverState()
        state.add_goal("Complete task A")
        state.record_tool_use("search", '{"q": "docs"}', "found 3 results", iteration=0)
        state.record_tool_use("read_file", '{"path": "/a"}', "contents", iteration=1)

        hook = CarryoverHook(state=state)
        ctx = HookContext(event=HookEvent.SESSION_END, metadata={})
        result = await hook.execute(ctx)

        assert "carryover_summary" in result.metadata
        summary = result.metadata["carryover_summary"]
        assert summary["goals"] == 1
        assert summary["tool_calls"] == 2
        assert summary["unique_tools"] == 2
        assert summary["log_entries"] == 2

    async def test_session_end_summary_on_empty_state(self):
        hook = CarryoverHook()
        ctx = HookContext(event=HookEvent.SESSION_END, metadata={})
        result = await hook.execute(ctx)

        summary = result.metadata["carryover_summary"]
        assert summary["goals"] == 0
        assert summary["tool_calls"] == 0
        assert summary["unique_tools"] == 0

    async def test_session_end_preserves_existing_metadata(self):
        hook = CarryoverHook()
        ctx = HookContext(
            event=HookEvent.SESSION_END,
            metadata={"existing_key": "existing_value"},
        )
        result = await hook.execute(ctx)

        assert result.metadata["existing_key"] == "existing_value"
        assert "carryover_summary" in result.metadata
