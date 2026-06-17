"""Tests for koboi.hooks module."""

from __future__ import annotations

from unittest.mock import MagicMock

from koboi.hooks.chain import HookEvent, HookContext, Hook, HookChain
from koboi.hooks.builtin import LoggingHook, AuditHook
from koboi.hooks.callback_hook import CallbackHook


class SimpleHook(Hook):
    def __init__(self, events=None):
        self._events = events or list(HookEvent)
        self.log = []

    def handles(self) -> list[HookEvent]:
        return self._events

    async def execute(self, ctx: HookContext) -> HookContext:
        self.log.append(ctx.event)
        return ctx


class AbortHook(Hook):
    def handles(self) -> list[HookEvent]:
        return [HookEvent.PRE_TOOL_USE]

    async def execute(self, ctx: HookContext) -> HookContext:
        ctx.abort = True
        return ctx


class TestHookChain:
    async def test_dispatch(self):
        hook = SimpleHook()
        chain = HookChain([hook])
        await chain.emit(HookContext(event=HookEvent.SESSION_START))
        assert HookEvent.SESSION_START in hook.log

    async def test_multiple_hooks(self):
        h1 = SimpleHook([HookEvent.SESSION_START])
        h2 = SimpleHook([HookEvent.SESSION_START])
        chain = HookChain([h1, h2])
        await chain.emit(HookContext(event=HookEvent.SESSION_START))
        assert len(h1.log) == 1
        assert len(h2.log) == 1

    async def test_event_filtering(self):
        hook = SimpleHook([HookEvent.PRE_TOOL_USE])
        chain = HookChain([hook])
        await chain.emit(HookContext(event=HookEvent.SESSION_START))
        assert len(hook.log) == 0
        await chain.emit(HookContext(event=HookEvent.PRE_TOOL_USE))
        assert len(hook.log) == 1

    async def test_abort(self):
        h1 = AbortHook()
        h2 = SimpleHook([HookEvent.PRE_TOOL_USE])
        chain = HookChain([h1, h2])
        ctx = await chain.emit(HookContext(event=HookEvent.PRE_TOOL_USE))
        assert ctx.abort is True
        assert len(h2.log) == 0

    async def test_add_hook(self):
        chain = HookChain()
        hook = SimpleHook()
        chain.add(hook)
        await chain.emit(HookContext(event=HookEvent.SESSION_START))
        assert len(hook.log) == 1


class TestHookContext:
    def test_defaults(self):
        ctx = HookContext(event=HookEvent.SESSION_START)
        assert ctx.event == HookEvent.SESSION_START
        assert ctx.iteration == 0
        assert ctx.abort is False
        assert ctx.inject_message is None

    def test_custom_fields(self):
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="test", iteration=5)
        assert ctx.tool_name == "test"
        assert ctx.iteration == 5


class TestLoggingHook:
    def test_handles_all_events(self):
        """LoggingHook should handle all HookEvents."""
        mock_logger = MagicMock()
        hook = LoggingHook(logger=mock_logger)
        assert set(hook.handles()) == set(HookEvent)

    async def test_execute_with_logger(self):
        """LoggingHook should call logger.log() with formatted message."""
        mock_logger = MagicMock()
        hook = LoggingHook(logger=mock_logger)
        ctx = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            tool_name="test_tool",
            iteration=3,
        )
        result = await hook.execute(ctx)
        assert result is ctx
        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args[0][0]
        assert "pre_tool_use" in call_args
        assert "tool=test_tool" in call_args
        assert "iter=3" in call_args

    async def test_execute_without_logger(self):
        """LoggingHook should not crash when logger is None."""
        hook = LoggingHook(logger=None)
        ctx = HookContext(event=HookEvent.SESSION_START)
        result = await hook.execute(ctx)
        assert result is ctx
        # Should not raise any exception

    async def test_execute_verbose_flag(self):
        """LoggingHook verbose flag should be stored but not affect basic execution."""
        mock_logger = MagicMock()
        hook = LoggingHook(logger=mock_logger, verbose=True)
        assert hook.verbose is True
        ctx = HookContext(event=HookEvent.SESSION_START)
        result = await hook.execute(ctx)
        assert result is ctx
        mock_logger.log.assert_called_once()


class TestAuditHook:
    def test_handles_specific_events(self):
        """AuditHook should handle PRE_TOOL_USE, POST_TOOL_USE, DOOM_LOOP_DETECTED."""
        mock_audit = MagicMock()
        hook = AuditHook(audit_trail=mock_audit)
        expected_events = {
            HookEvent.PRE_TOOL_USE,
            HookEvent.POST_TOOL_USE,
            HookEvent.DOOM_LOOP_DETECTED,
        }
        assert set(hook.handles()) == expected_events

    async def test_execute_records_audit_entry(self):
        """AuditHook should record AuditEntry with correct information."""
        from koboi.types import AuditEntry

        mock_audit = MagicMock()
        hook = AuditHook(audit_trail=mock_audit)
        ctx = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            tool_name="test_tool",
            tool_arguments='{"arg": "value"}',
            tool_result="Success",
            iteration=2,
        )
        result = await hook.execute(ctx)
        assert result is ctx
        mock_audit.record.assert_called_once()
        call_args = mock_audit.record.call_args[0][0]
        assert isinstance(call_args, AuditEntry)
        assert call_args.tool_name == "test_tool"
        assert call_args.arguments == '{"arg": "value"}'
        assert call_args.result == "Success"
        assert call_args.event_type == "harness_pre_tool_use"

    async def test_execute_with_long_arguments_truncated(self):
        """AuditHook should truncate long arguments and results."""
        from koboi.types import AuditEntry

        mock_audit = MagicMock()
        hook = AuditHook(audit_trail=mock_audit)
        long_args = "x" * 300
        long_result = "y" * 300
        ctx = HookContext(
            event=HookEvent.POST_TOOL_USE,
            tool_name="test_tool",
            tool_arguments=long_args,
            tool_result=long_result,
        )
        await hook.execute(ctx)
        call_args = mock_audit.record.call_args[0][0]
        assert len(call_args.arguments) <= 200
        assert len(call_args.result) <= 200


class TestCallbackHook:
    def test_handles_all_events_by_default(self):
        """CallbackHook should handle all events when no custom events provided."""
        sync_callback = MagicMock(return_value=None)
        hook = CallbackHook(callback=sync_callback)
        assert set(hook.handles()) == set(HookEvent)

    def test_handles_custom_events(self):
        """CallbackHook should handle only specified custom events."""
        sync_callback = MagicMock(return_value=None)
        custom_events = [HookEvent.PRE_TOOL_USE, HookEvent.POST_TOOL_USE]
        hook = CallbackHook(callback=sync_callback, events=custom_events)
        assert set(hook.handles()) == set(custom_events)

    async def test_execute_with_sync_callback(self):
        """CallbackHook should execute synchronous callbacks."""

        def sync_callback(ctx):
            # Callback must return the context
            return ctx

        hook = CallbackHook(callback=sync_callback)
        ctx = HookContext(event=HookEvent.SESSION_START)
        result = await hook.execute(ctx)
        assert result is ctx

    async def test_execute_with_async_callback(self):
        """CallbackHook should execute async callbacks."""

        async def async_callback(ctx):
            ctx.metadata["async_called"] = True
            return ctx

        hook = CallbackHook(callback=async_callback)
        ctx = HookContext(event=HookEvent.SESSION_START)
        result = await hook.execute(ctx)
        assert result.metadata.get("async_called") is True

    async def test_callback_can_modify_context(self):
        """Callback should be able to modify and return context."""

        def modifying_callback(ctx):
            ctx.metadata["modified"] = True
            return ctx

        hook = CallbackHook(callback=modifying_callback)
        ctx = HookContext(event=HookEvent.PRE_INPUT)
        result = await hook.execute(ctx)
        assert result.metadata.get("modified") is True

    async def test_callback_can_return_different_context(self):
        """Callback should be able to return a different context instance."""

        def new_context_callback(ctx):
            new_ctx = HookContext(event=ctx.event, iteration=999)
            return new_ctx

        hook = CallbackHook(callback=new_context_callback)
        ctx = HookContext(event=HookEvent.SESSION_START, iteration=1)
        result = await hook.execute(ctx)
        assert result.iteration == 999
        assert result is not ctx


class CrashingHook(Hook):
    def __init__(self, events=None):
        self._events = events or [HookEvent.SESSION_START]

    def handles(self) -> list[HookEvent]:
        return self._events

    async def execute(self, ctx: HookContext) -> HookContext:
        raise ValueError("boom")


class TestHookErrorIsolation:
    async def test_crashing_hook_aborts_chain(self):
        h1 = CrashingHook()
        h2 = SimpleHook([HookEvent.SESSION_START])
        chain = HookChain([h1, h2])
        ctx = await chain.emit(HookContext(event=HookEvent.SESSION_START))
        assert ctx.abort is True
        assert len(h2.log) == 0

    async def test_crashing_hook_records_outcome(self):
        from koboi.hooks.chain import HookOutcome

        h1 = CrashingHook()
        chain = HookChain([h1])
        ctx = await chain.emit(HookContext(event=HookEvent.SESSION_START))
        assert len(ctx.hook_outcomes) == 1
        assert ctx.hook_outcomes[0][1] == HookOutcome.ERRORED
        assert "boom" in ctx.hook_outcomes[0][2]

    async def test_crashing_hook_preserves_previous_ctx(self):
        h1 = SimpleHook([HookEvent.SESSION_START])
        h2 = CrashingHook()
        h3 = SimpleHook([HookEvent.SESSION_START])
        chain = HookChain([h1, h2, h3])
        ctx = await chain.emit(HookContext(event=HookEvent.SESSION_START))
        assert len(h1.log) == 1
        assert ctx.abort is True
        assert len(h3.log) == 0

    async def test_abort_before_crashing_hook_stops_chain(self):
        h1 = AbortHook()
        h2 = CrashingHook([HookEvent.PRE_TOOL_USE])
        h3 = SimpleHook([HookEvent.PRE_TOOL_USE])
        chain = HookChain([h1, h2, h3])
        ctx = await chain.emit(HookContext(event=HookEvent.PRE_TOOL_USE))
        assert ctx.abort is True
        assert len(h3.log) == 0

    async def test_multiple_crashing_hooks_abort_on_first(self):
        h1 = CrashingHook()
        h2 = CrashingHook()
        h3 = SimpleHook([HookEvent.SESSION_START])
        chain = HookChain([h1, h2, h3])
        ctx = await chain.emit(HookContext(event=HookEvent.SESSION_START))
        assert ctx.abort is True
        assert len(h3.log) == 0


class TestHookPriority:
    async def test_priority_ordering(self):
        order = []

        class TrackingHook(Hook):
            def __init__(self, name, p, events):
                self.name = name
                self.priority = p
                self._events = events

            def handles(self):
                return self._events

            async def execute(self, ctx):
                order.append(self.name)
                return ctx

        low = TrackingHook("low", 10, [HookEvent.SESSION_START])
        high = TrackingHook("high", 90, [HookEvent.SESSION_START])
        chain = HookChain([high, low])
        await chain.emit(HookContext(event=HookEvent.SESSION_START))
        assert order == ["low", "high"]

    async def test_stable_sort_same_priority(self):
        order = []

        class TrackingHook(Hook):
            def __init__(self, name, events):
                self.name = name
                self._events = events

            def handles(self):
                return self._events

            async def execute(self, ctx):
                order.append(self.name)
                return ctx

        h1 = TrackingHook("first", [HookEvent.SESSION_START])
        h2 = TrackingHook("second", [HookEvent.SESSION_START])
        chain = HookChain([h1, h2])
        await chain.emit(HookContext(event=HookEvent.SESSION_START))
        assert order == ["first", "second"]

    def test_default_priority_is_50(self):
        hook = SimpleHook()
        assert hook.priority == 50

    async def test_add_resorts_chain(self):
        order = []

        class TrackingHook(Hook):
            def __init__(self, name, p, events):
                self.name = name
                self.priority = p
                self._events = events

            def handles(self):
                return self._events

            async def execute(self, ctx):
                order.append(self.name)
                return ctx

        chain = HookChain()
        high = TrackingHook("high", 90, [HookEvent.SESSION_START])
        low = TrackingHook("low", 10, [HookEvent.SESSION_START])
        chain.add(high)
        chain.add(low)
        await chain.emit(HookContext(event=HookEvent.SESSION_START))
        assert order == ["low", "high"]

    def test_builtin_priorities(self):
        assert LoggingHook.priority == 0
        mock_audit = MagicMock()
        hook = AuditHook(audit_trail=mock_audit)
        assert hook.priority == 80
