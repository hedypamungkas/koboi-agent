"""Tests for koboi/hooks/telemetry_hook.py — TelemetryHook (42% → >85%)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from koboi.hooks.chain import HookContext, HookEvent
from koboi.hooks.telemetry_hook import TelemetryHook
from koboi.harness.telemetry import TelemetryCollector
from koboi.types import AgentResponse, TokenUsage


class TestTelemetryHookInitialization:
    def test_handles_all_events(self):
        """TelemetryHook should handle all HookEvents."""
        collector = TelemetryCollector()
        hook = TelemetryHook(telemetry=collector)
        assert set(hook.handles()) == set(HookEvent)

    def test_stores_telemetry_collector(self):
        """TelemetryHook should store the provided TelemetryCollector."""
        collector = TelemetryCollector(session_id="test-session")
        hook = TelemetryHook(telemetry=collector)
        assert hook.telemetry is collector


class TestTelemetryHookSessionEvents:
    async def test_session_start_calls_telemetry_session_start(self):
        """SESSION_START event should call telemetry.session_start()."""
        collector = TelemetryCollector()
        hook = TelemetryHook(telemetry=collector)

        ctx = HookContext(event=HookEvent.SESSION_START)
        await hook.execute(ctx)

        # Verify session was started
        assert collector.snapshot.start_time > 0

    async def test_session_end_calls_telemetry_session_end(self):
        """SESSION_END event should call telemetry.session_end()."""
        collector = TelemetryCollector()
        hook = TelemetryHook(telemetry=collector)

        ctx = HookContext(event=HookEvent.SESSION_END)
        await hook.execute(ctx)

        # Verify session was ended
        assert collector.snapshot.end_time > 0


class TestTelemetryHookToolEvents:
    async def test_pre_tool_use_records_tool_call(self):
        """PRE_TOOL_USE event should call telemetry.record_tool_call()."""
        collector = TelemetryCollector()
        hook = TelemetryHook(telemetry=collector)

        ctx = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            tool_name="test_tool",
            iteration=0,
        )
        await hook.execute(ctx)

        assert collector.snapshot.total_tool_calls == 1
        assert "test_tool" in collector.snapshot.unique_tools_used

    async def test_pre_tool_use_with_no_tool_name(self):
        """PRE_TOOL_USE without tool_name should not record."""
        collector = TelemetryCollector()
        hook = TelemetryHook(telemetry=collector)

        ctx = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            tool_name=None,
        )
        await hook.execute(ctx)

        assert collector.snapshot.total_tool_calls == 0

    async def test_post_tool_use_success_calls_record_tool_success(self):
        """POST_TOOL_USE with successful result should call record_tool_success()."""
        collector = TelemetryCollector()
        hook = TelemetryHook(telemetry=collector)

        ctx = HookContext(
            event=HookEvent.POST_TOOL_USE,
            tool_result="Success: operation completed",
        )
        await hook.execute(ctx)

        assert collector.snapshot.tools_succeeded == 1
        assert collector.snapshot.tools_failed == 0

    async def test_post_tool_use_error_calls_record_tool_failure(self):
        """POST_TOOL_USE with error result should call record_tool_failure()."""
        collector = TelemetryCollector()
        hook = TelemetryHook(telemetry=collector)

        ctx = HookContext(
            event=HookEvent.POST_TOOL_USE,
            tool_result="Error: something went wrong",
        )
        await hook.execute(ctx)

        assert collector.snapshot.tools_succeeded == 0
        assert collector.snapshot.tools_failed == 1

    async def test_post_tool_use_records_permission_decision(self):
        """POST_TOOL_USE should record permission decision if available."""
        collector = TelemetryCollector()
        hook = TelemetryHook(telemetry=collector)

        ctx = HookContext(
            event=HookEvent.POST_TOOL_USE,
            tool_name="test_tool",
        )
        ctx.metadata["policy_decision"] = {
            "action": "confirmed",
            "matched_rule": "destructive_operations",
        }
        await hook.execute(ctx)

        assert len(collector.snapshot.permissions) == 1
        perm = collector.snapshot.permissions[0]
        assert perm.action == "confirmed"
        assert perm.rule_name == "destructive_operations"


class TestTelemetryHookLLMEvents:
    async def test_pre_llm_call_starts_iteration(self):
        """PRE_LLM_CALL event should call telemetry.iteration_start()."""
        collector = TelemetryCollector()
        hook = TelemetryHook(telemetry=collector)

        messages = [
            {"role": "user", "content": "Hello " * 100},  # ~500 chars
        ]
        ctx = HookContext(
            event=HookEvent.PRE_LLM_CALL,
            messages=messages,
        )
        await hook.execute(ctx)

        # Verify iteration was started (tokens estimated)
        # Messages are ~500 chars, /4 = ~125 tokens
        assert collector._tokens_at_iteration_start > 0

    async def test_post_llm_call_ends_iteration(self):
        """POST_LLM_CALL event should call telemetry.iteration_end()."""
        collector = TelemetryCollector()
        hook = TelemetryHook(telemetry=collector)

        response = AgentResponse(
            content="Response",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        )
        ctx = HookContext(
            event=HookEvent.POST_LLM_CALL,
            iteration=3,
            llm_response=response,
        )
        await hook.execute(ctx)

        assert len(collector.snapshot.iterations) == 1
        assert collector.snapshot.iterations[0].iteration == 3

    async def test_post_llm_call_without_usage(self):
        """POST_LLM_CALL without usage should still record iteration."""
        collector = TelemetryCollector()
        hook = TelemetryHook(telemetry=collector)

        response = AgentResponse(content="Response")
        ctx = HookContext(
            event=HookEvent.POST_LLM_CALL,
            iteration=1,
            llm_response=response,
        )
        await hook.execute(ctx)

        assert len(collector.snapshot.iterations) == 1


class TestTelemetryHookDoomLoopEvent:
    async def test_doom_loop_detected_records_event(self):
        """DOOM_LOOP_DETECTED event should call telemetry.record_doom_loop()."""
        collector = TelemetryCollector()
        hook = TelemetryHook(telemetry=collector)

        ctx = HookContext(event=HookEvent.DOOM_LOOP_DETECTED)
        await hook.execute(ctx)

        assert collector.snapshot.doom_loops_detected == 1

    async def test_multiple_doom_loops_accumulate(self):
        """Multiple doom loop events should accumulate in telemetry."""
        collector = TelemetryCollector()
        hook = TelemetryHook(telemetry=collector)

        ctx = HookContext(event=HookEvent.DOOM_LOOP_DETECTED)
        await hook.execute(ctx)
        await hook.execute(ctx)
        await hook.execute(ctx)

        assert collector.snapshot.doom_loops_detected == 3


class TestTelemetryHookNoOpEvents:
    async def test_pre_input_is_noop(self):
        """PRE_INPUT event should not modify telemetry state."""
        collector = TelemetryCollector()
        hook = TelemetryHook(telemetry=collector)

        initial_state = collector.snapshot.__dict__.copy()
        ctx = HookContext(event=HookEvent.PRE_INPUT)
        await hook.execute(ctx)

        # State should be unchanged
        assert collector.snapshot.__dict__ == initial_state

    async def test_post_output_is_noop(self):
        """POST_OUTPUT event should not modify telemetry state."""
        collector = TelemetryCollector()
        hook = TelemetryHook(telemetry=collector)

        initial_state = collector.snapshot.__dict__.copy()
        ctx = HookContext(event=HookEvent.POST_OUTPUT)
        await hook.execute(ctx)

        # State should be unchanged
        assert collector.snapshot.__dict__ == initial_state

    async def test_pre_compact_is_noop(self):
        """PRE_COMPACT event should not modify telemetry state."""
        collector = TelemetryCollector()
        hook = TelemetryHook(telemetry=collector)

        initial_state = collector.snapshot.__dict__.copy()
        ctx = HookContext(event=HookEvent.PRE_COMPACT)
        await hook.execute(ctx)

        # State should be unchanged
        assert collector.snapshot.__dict__ == initial_state

    async def test_post_compact_is_noop(self):
        """POST_COMPACT event should not modify telemetry state."""
        collector = TelemetryCollector()
        hook = TelemetryHook(telemetry=collector)

        initial_state = collector.snapshot.__dict__.copy()
        ctx = HookContext(event=HookEvent.POST_COMPACT)
        await hook.execute(ctx)

        # State should be unchanged
        assert collector.snapshot.__dict__ == initial_state


class TestTelemetryHookIntegration:
    async def test_full_session_workflow(self):
        """Test a complete session with various events."""
        collector = TelemetryCollector(session_id="test-full-session")
        hook = TelemetryHook(telemetry=collector)

        # Session start
        await hook.execute(HookContext(event=HookEvent.SESSION_START))

        # First iteration
        messages = [{"role": "user", "content": "Hello"}]
        await hook.execute(
            HookContext(
                event=HookEvent.PRE_LLM_CALL,
                messages=messages,
            )
        )

        response = AgentResponse(
            content="Hi!",
            usage=TokenUsage(prompt_tokens=5, completion_tokens=2),
        )
        await hook.execute(
            HookContext(
                event=HookEvent.POST_LLM_CALL,
                iteration=0,
                llm_response=response,
            )
        )

        # Tool use
        await hook.execute(
            HookContext(
                event=HookEvent.PRE_TOOL_USE,
                tool_name="test_tool",
            )
        )

        await hook.execute(
            HookContext(
                event=HookEvent.POST_TOOL_USE,
                tool_result="Success",
            )
        )

        # Session end
        await hook.execute(HookContext(event=HookEvent.SESSION_END))

        # Verify state
        assert collector.snapshot.session_id == "test-full-session"
        assert collector.snapshot.total_iterations == 1
        assert collector.snapshot.total_tool_calls == 1
        assert collector.snapshot.tools_succeeded == 1

    async def test_context_unchanged_after_execution(self):
        """Hook execution should not modify the context unexpectedly."""
        collector = TelemetryCollector()
        hook = TelemetryHook(telemetry=collector)

        original_metadata = {"key": "value"}
        ctx = HookContext(
            event=HookEvent.SESSION_START,
            metadata=original_metadata.copy(),
        )
        result = await hook.execute(ctx)

        assert result is ctx
        assert result.metadata == original_metadata

    async def test_multiple_tools_tracked_correctly(self):
        """Multiple different tools should all be tracked."""
        collector = TelemetryCollector()
        hook = TelemetryHook(telemetry=collector)

        # Use tool1
        await hook.execute(
            HookContext(
                event=HookEvent.PRE_TOOL_USE,
                tool_name="tool1",
            )
        )

        # Use tool2
        await hook.execute(
            HookContext(
                event=HookEvent.PRE_TOOL_USE,
                tool_name="tool2",
            )
        )

        # Use tool1 again
        await hook.execute(
            HookContext(
                event=HookEvent.PRE_TOOL_USE,
                tool_name="tool1",
            )
        )

        assert collector.snapshot.total_tool_calls == 3
        assert len(collector.snapshot.unique_tools_used) == 2
        assert "tool1" in collector.snapshot.unique_tools_used
        assert "tool2" in collector.snapshot.unique_tools_used
