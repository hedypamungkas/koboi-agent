"""Tests for koboi/hooks/context_hook.py — ContextHook (0% → >85%)."""
from __future__ import annotations

import pytest

from koboi.hooks.chain import HookContext, HookEvent
from koboi.hooks.context_hook import ContextHook


class TestContextHook:
    def test_handles_returns_post_compact(self):
        """ContextHook should handle POST_COMPACT event."""
        hook = ContextHook()
        assert hook.handles() == [HookEvent.POST_COMPACT]

    async def test_empty_messages_passthrough(self):
        """Empty messages list should return context unchanged."""
        hook = ContextHook()
        ctx = HookContext(event=HookEvent.POST_COMPACT, messages=[])
        result = await hook.execute(ctx)
        assert result is ctx
        assert ctx.messages == []

    async def test_system_messages_always_preserved(self):
        """System messages should be preserved when preserve_system=True."""
        hook = ContextHook(preserve_system=True)
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        ctx = HookContext(event=HookEvent.POST_COMPACT, messages=messages)
        result = await hook.execute(ctx)
        assert result.messages[0]["role"] == "system"
        assert result.messages[0]["content"] == "You are helpful"

    async def test_system_messages_not_preserved_when_false(self):
        """System messages should not be preserved when preserve_system=False."""
        hook = ContextHook(preserve_system=False)
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        ctx = HookContext(event=HookEvent.POST_COMPACT, messages=messages)
        result = await hook.execute(ctx)
        # With preserve_recent=5, all messages fit
        assert len(result.messages) == 2
        # But system message is not prioritized
        assert result.messages == messages

    async def test_recent_messages_preserved(self):
        """Most recent messages should be preserved based on preserve_recent."""
        hook = ContextHook(preserve_recent=3, max_messages=3, preserve_system=False)
        messages = [
            {"role": "user", "content": f"Message {i}"} for i in range(10)
        ]
        ctx = HookContext(event=HookEvent.POST_COMPACT, messages=messages)
        result = await hook.execute(ctx)
        # Should keep last 3 messages (max_messages=3 limits the total)
        assert len(result.messages) == 3
        assert result.messages[0]["content"] == "Message 7"
        assert result.messages[2]["content"] == "Message 9"

    async def test_recent_messages_greater_than_total(self):
        """When preserve_recent > total messages, all messages should be kept."""
        hook = ContextHook(preserve_recent=10)
        messages = [
            {"role": "user", "content": f"Message {i}"} for i in range(5)
        ]
        ctx = HookContext(event=HookEvent.POST_COMPACT, messages=messages)
        result = await hook.execute(ctx)
        assert len(result.messages) == 5

    async def test_middle_messages_fill_up_to_max_messages(self):
        """Middle messages should fill remaining slots up to max_messages."""
        hook = ContextHook(max_messages=5, preserve_system=False, preserve_recent=2)
        messages = [
            {"role": "user", "content": f"Message {i}"} for i in range(10)
        ]
        ctx = HookContext(event=HookEvent.POST_COMPACT, messages=messages)
        result = await hook.execute(ctx)
        # Should have 5 messages: 2 recent + 3 middle (first 3)
        assert len(result.messages) == 5
        # Recent messages come first (in original order)
        assert result.messages[0]["content"] == "Message 8"
        assert result.messages[1]["content"] == "Message 9"
        # Then middle messages (oldest first)
        assert result.messages[2]["content"] == "Message 0"
        assert result.messages[3]["content"] == "Message 1"
        assert result.messages[4]["content"] == "Message 2"

    async def test_metadata_set_after_execution(self):
        """Metadata should be set with context management info."""
        hook = ContextHook()
        messages = [{"role": "user", "content": f"Message {i}"} for i in range(10)]
        ctx = HookContext(event=HookEvent.POST_COMPACT, messages=messages)
        result = await hook.execute(ctx)
        assert result.metadata["context_managed"] is True
        assert result.metadata["context_before"] == 10
        # With preserve_system=True (default) and preserve_recent=5, we keep:
        # - 0 system messages + 5 recent + fill up to 50 max_messages
        # So all 10 messages are kept (they all fit in max_messages=50)
        assert result.metadata["context_after"] == 10

    async def test_messages_trimmed_when_managed_less_than_original(self):
        """Messages should be trimmed when managed < original count."""
        hook = ContextHook(max_messages=3, preserve_system=False, preserve_recent=2)
        messages = [
            {"role": "user", "content": f"Message {i}"} for i in range(10)
        ]
        ctx = HookContext(event=HookEvent.POST_COMPACT, messages=messages)
        original_length = len(ctx.messages)
        result = await hook.execute(ctx)
        assert len(result.messages) < original_length
        assert len(result.messages) == 3

    async def test_messages_not_trimmed_when_fit_in_max(self):
        """Messages should not be trimmed when they fit within max_messages."""
        hook = ContextHook(max_messages=100)
        messages = [
            {"role": "user", "content": f"Message {i}"} for i in range(10)
        ]
        ctx = HookContext(event=HookEvent.POST_COMPACT, messages=messages)
        result = await hook.execute(ctx)
        assert len(result.messages) == 10

    async def test_system_and_recent_combination(self):
        """System messages and recent messages should both be preserved."""
        hook = ContextHook(preserve_system=True, preserve_recent=2, max_messages=10)
        messages = [
            {"role": "system", "content": "System prompt"},
            *[{"role": "user", "content": f"Message {i}"} for i in range(10)]
        ]
        ctx = HookContext(event=HookEvent.POST_COMPACT, messages=messages)
        result = await hook.execute(ctx)
        # We have 11 total messages (1 system + 10 user) but max_messages=10
        # Order: System, recent (Message 8, 9), middle (Message 0-6)
        assert len(result.messages) == 10
        assert result.messages[0]["role"] == "system"
        # Recent messages come right after system
        assert result.messages[1]["content"] == "Message 8"
        assert result.messages[2]["content"] == "Message 9"
        # Middle messages fill the rest
        assert result.messages[-1]["content"] == "Message 6"
        assert result.messages[-2]["content"] == "Message 5"

    async def test_no_duplicate_messages(self):
        """Messages should not appear twice even if they match multiple criteria."""
        hook = ContextHook(preserve_system=True, preserve_recent=5)
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User 1"},
            {"role": "user", "content": "User 2"},
        ]
        ctx = HookContext(event=HookEvent.POST_COMPACT, messages=messages)
        result = await hook.execute(ctx)
        # Count unique messages
        unique_messages = len(set(id(m) for m in result.messages))
        assert len(result.messages) == unique_messages

    async def test_custom_max_messages_default(self):
        """Default max_messages should be 50."""
        hook = ContextHook()
        assert hook.max_messages == 50

    async def test_custom_preserve_recent_default(self):
        """Default preserve_recent should be 5."""
        hook = ContextHook()
        assert hook.preserve_recent == 5
