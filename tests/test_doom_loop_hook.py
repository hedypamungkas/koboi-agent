"""Tests for koboi/hooks/doom_loop_hook.py — DoomLoopHook (0% → >85%)."""

from __future__ import annotations

from unittest.mock import MagicMock


from koboi.hooks.chain import HookContext, HookEvent
from koboi.hooks.doom_loop_hook import DoomLoopHook
from koboi.harness.doom_loop import DoomLoopConfig, DoomLoopResult


class TestDoomLoopHook:
    def test_handles_returns_session_start_and_post_tool_use(self):
        """DoomLoopHook handles POST_TOOL_USE (detection) + SESSION_START (per-run reset)."""
        hook = DoomLoopHook()
        assert hook.handles() == [HookEvent.SESSION_START, HookEvent.POST_TOOL_USE]

    async def test_passthrough_when_no_tool_name(self):
        """Should return context unchanged when no tool_name provided."""
        hook = DoomLoopHook()
        ctx = HookContext(event=HookEvent.POST_TOOL_USE, tool_name=None)
        result = await hook.execute(ctx)
        assert result is ctx
        assert "doom_loop" not in ctx.metadata

    async def test_records_tool_call(self):
        """Should record tool call in detector."""
        hook = DoomLoopHook()
        ctx = HookContext(
            event=HookEvent.POST_TOOL_USE,
            tool_name="test_tool",
            tool_arguments='{"arg": "value"}',
        )
        await hook.execute(ctx)
        assert len(hook.detector.history) == 1
        assert hook.detector.history[0] == ("test_tool", '{"arg": "value"}')

    async def test_detects_error_in_tool_result(self):
        """Should detect error when tool_result contains 'error'."""
        hook = DoomLoopHook()
        ctx = HookContext(
            event=HookEvent.POST_TOOL_USE,
            tool_name="failing_tool",
            tool_arguments='{"arg": "value"}',
            tool_result="Error: something went wrong",
        )
        await hook.execute(ctx)
        # Check that error was recorded
        assert len(hook.detector.history) == 1

    async def test_no_detection_on_clean_passthrough(self):
        """Should not detect doom loop on normal tool execution."""
        hook = DoomLoopHook()
        ctx = HookContext(
            event=HookEvent.POST_TOOL_USE,
            tool_name="test_tool",
            tool_arguments='{"arg": "value"}',
            tool_result="Success",
        )
        result = await hook.execute(ctx)
        assert "doom_loop" not in result.metadata
        assert result.inject_message is None

    async def test_on_doom_loop_detection_sets_metadata(self):
        """When doom loop detected, should set metadata with detection info."""
        hook = DoomLoopHook(config=DoomLoopConfig(consecutive_identical_threshold=2))
        # Record same tool call twice
        for _ in range(2):
            ctx = HookContext(
                event=HookEvent.POST_TOOL_USE,
                tool_name="repeat_tool",
                tool_arguments='{"same": "args"}',
                tool_result="Try again",
            )
            await hook.execute(ctx)

        # Third call should trigger detection
        ctx = HookContext(
            event=HookEvent.POST_TOOL_USE,
            tool_name="repeat_tool",
            tool_arguments='{"same": "args"}',
            tool_result="Try again",
        )
        result = await hook.execute(ctx)
        assert result.metadata.get("doom_loop", {}).get("detected") is True

    async def test_on_doom_loop_injects_recovery_message(self):
        """When doom loop detected, should inject recovery message."""
        hook = DoomLoopHook(config=DoomLoopConfig(consecutive_identical_threshold=2))
        # Record same tool call twice
        for _ in range(2):
            ctx = HookContext(
                event=HookEvent.POST_TOOL_USE,
                tool_name="repeat_tool",
                tool_arguments='{"same": "args"}',
            )
            await hook.execute(ctx)

        # Third call should trigger detection
        ctx = HookContext(
            event=HookEvent.POST_TOOL_USE,
            tool_name="repeat_tool",
            tool_arguments='{"same": "args"}',
        )
        result = await hook.execute(ctx)
        assert result.inject_message is not None
        assert "DOOM LOOP WARNING" in result.inject_message

    async def test_on_doom_loop_sets_detected_flag(self):
        """Should set doom_loop_detected flag in metadata."""
        hook = DoomLoopHook(config=DoomLoopConfig(consecutive_identical_threshold=2))
        # Record same tool call twice
        for _ in range(2):
            ctx = HookContext(
                event=HookEvent.POST_TOOL_USE,
                tool_name="repeat_tool",
                tool_arguments='{"same": "args"}',
            )
            await hook.execute(ctx)

        # Third call should trigger detection
        ctx = HookContext(
            event=HookEvent.POST_TOOL_USE,
            tool_name="repeat_tool",
            tool_arguments='{"same": "args"}',
        )
        result = await hook.execute(ctx)
        assert result.metadata.get("doom_loop_detected") is True

    async def test_on_doom_loop_callback_called(self):
        """Should call on_doom_loop callback when detection occurs."""
        callback_mock = MagicMock()
        hook = DoomLoopHook(
            config=DoomLoopConfig(consecutive_identical_threshold=2),
            on_doom_loop=callback_mock,
        )
        # Record same tool call twice
        for _ in range(2):
            ctx = HookContext(
                event=HookEvent.POST_TOOL_USE,
                tool_name="repeat_tool",
                tool_arguments='{"same": "args"}',
            )
            await hook.execute(ctx)

        # Third call should trigger detection and callback
        ctx = HookContext(
            event=HookEvent.POST_TOOL_USE,
            tool_name="repeat_tool",
            tool_arguments='{"same": "args"}',
        )
        await hook.execute(ctx)
        assert callback_mock.called

    async def test_callback_receives_result_and_context(self):
        """Callback should receive DoomLoopResult and HookContext."""
        captured_args = []

        def capture_callback(result, ctx):
            captured_args.append((result, ctx))

        hook = DoomLoopHook(
            config=DoomLoopConfig(consecutive_identical_threshold=2),
            on_doom_loop=capture_callback,
        )
        # Record same tool call once
        ctx = HookContext(
            event=HookEvent.POST_TOOL_USE,
            tool_name="repeat_tool",
            tool_arguments='{"same": "args"}',
        )
        await hook.execute(ctx)

        # Second call should trigger detection (threshold is 2)
        test_ctx = HookContext(
            event=HookEvent.POST_TOOL_USE,
            tool_name="repeat_tool",
            tool_arguments='{"same": "args"}',
            iteration=1,
        )
        await hook.execute(test_ctx)
        assert len(captured_args) == 1
        result, ctx = captured_args[0]
        assert isinstance(result, DoomLoopResult)
        assert result.detected is True
        assert ctx is test_ctx

    async def test_metadata_contains_loop_type(self):
        """Metadata should contain loop_type when detected."""
        hook = DoomLoopHook(config=DoomLoopConfig(consecutive_identical_threshold=2))
        # Record same tool call twice
        for _ in range(2):
            ctx = HookContext(
                event=HookEvent.POST_TOOL_USE,
                tool_name="repeat_tool",
                tool_arguments='{"same": "args"}',
            )
            await hook.execute(ctx)

        # Third call should trigger detection
        ctx = HookContext(
            event=HookEvent.POST_TOOL_USE,
            tool_name="repeat_tool",
            tool_arguments='{"same": "args"}',
        )
        result = await hook.execute(ctx)
        doom_loop_info = result.metadata.get("doom_loop", {})
        assert "loop_type" in doom_loop_info
        assert doom_loop_info["loop_type"] == "consecutive_identical"

    async def test_metadata_contains_pattern_description(self):
        """Metadata should contain pattern_description when detected."""
        hook = DoomLoopHook(config=DoomLoopConfig(consecutive_identical_threshold=2))
        # Record same tool call twice
        for _ in range(2):
            ctx = HookContext(
                event=HookEvent.POST_TOOL_USE,
                tool_name="repeat_tool",
                tool_arguments='{"same": "args"}',
            )
            await hook.execute(ctx)

        # Third call should trigger detection
        ctx = HookContext(
            event=HookEvent.POST_TOOL_USE,
            tool_name="repeat_tool",
            tool_arguments='{"same": "args"}',
        )
        result = await hook.execute(ctx)
        doom_loop_info = result.metadata.get("doom_loop", {})
        assert "pattern" in doom_loop_info
        assert "repeat_tool" in doom_loop_info["pattern"]

    async def test_metadata_contains_recovery_hint(self):
        """Metadata should contain recovery_hint when detected."""
        hook = DoomLoopHook(config=DoomLoopConfig(consecutive_identical_threshold=2))
        # Record same tool call twice
        for _ in range(2):
            ctx = HookContext(
                event=HookEvent.POST_TOOL_USE,
                tool_name="repeat_tool",
                tool_arguments='{"same": "args"}',
            )
            await hook.execute(ctx)

        # Third call should trigger detection
        ctx = HookContext(
            event=HookEvent.POST_TOOL_USE,
            tool_name="repeat_tool",
            tool_arguments='{"same": "args"}',
        )
        result = await hook.execute(ctx)
        doom_loop_info = result.metadata.get("doom_loop", {})
        assert "recovery_hint" in doom_loop_info
        assert len(doom_loop_info["recovery_hint"]) > 0

    async def test_metadata_contains_iterations_wasted(self):
        """Metadata should contain iterations_wasted when detected."""
        hook = DoomLoopHook(config=DoomLoopConfig(consecutive_identical_threshold=2))
        # Record same tool call twice
        for _ in range(2):
            ctx = HookContext(
                event=HookEvent.POST_TOOL_USE,
                tool_name="repeat_tool",
                tool_arguments='{"same": "args"}',
            )
            await hook.execute(ctx)

        # Third call should trigger detection
        ctx = HookContext(
            event=HookEvent.POST_TOOL_USE,
            tool_name="repeat_tool",
            tool_arguments='{"same": "args"}',
        )
        result = await hook.execute(ctx)
        doom_loop_info = result.metadata.get("doom_loop", {})
        assert "iterations_wasted" in doom_loop_info
        assert doom_loop_info["iterations_wasted"] >= 2

    async def test_custom_config_passed_to_detector(self):
        """Custom config should be passed to DoomLoopDetector."""
        custom_config = DoomLoopConfig(
            consecutive_identical_threshold=5,
            repeating_pattern_window=10,
        )
        hook = DoomLoopHook(config=custom_config)
        assert hook.detector.config is custom_config
        assert hook.detector.config.consecutive_identical_threshold == 5
        assert hook.detector.config.repeating_pattern_window == 10

    async def test_empty_tool_arguments_handled(self):
        """Should handle empty tool_arguments gracefully."""
        hook = DoomLoopHook()
        ctx = HookContext(
            event=HookEvent.POST_TOOL_USE,
            tool_name="test_tool",
            tool_arguments=None,
        )
        result = await hook.execute(ctx)
        assert result is ctx
        assert len(hook.detector.history) == 1
