"""Tests for SlidingWindowManager and context manager edge cases."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from koboi.context.manager import (
    ContextManager,
    NoopContextManager,
    TruncationManager,
    SmartTruncationManager,
    KeyFactsManager,
    SlidingWindowManager,
    ensure_tool_integrity,
)
from koboi.types import AgentResponse


class TestSlidingWindowManager:
    async def test_no_summarization_when_under_limit(self):
        mgr = SlidingWindowManager()
        messages = [{"role": "user", "content": "hi"}]
        result = await mgr.manage(messages, max_tokens=100000)
        assert result == messages

    async def test_summarization_with_client(self):
        client = MagicMock()
        client.complete = AsyncMock(return_value=AgentResponse(content="Summary of conversation"))
        mgr = SlidingWindowManager(client=client, keep_last=2)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "Tell me more."},
            {"role": "assistant", "content": "Python was created by Guido."},
            {"role": "user", "content": "When?"},
            {"role": "assistant", "content": "In 1991."},
        ]
        # Force token estimation to be high
        mgr.last_actual_tokens = 100000
        result = await mgr.manage(messages, max_tokens=1000)
        assert any("Summary" in m.get("content", "") for m in result)

    async def test_summarization_without_client(self):
        mgr = SlidingWindowManager(client=None, keep_last=2)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "a" * 1000},
            {"role": "assistant", "content": "b" * 1000},
            {"role": "user", "content": "c" * 1000},
            {"role": "assistant", "content": "d" * 1000},
        ]
        mgr.last_actual_tokens = 100000
        result = await mgr.manage(messages, max_tokens=100)
        # No summary, but truncation happened
        assert len(result) < len(messages)

    async def test_summarization_preserves_system(self):
        client = MagicMock()
        client.complete = AsyncMock(return_value=AgentResponse(content="summary"))
        mgr = SlidingWindowManager(client=client, keep_last=1)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "a" * 500},
            {"role": "assistant", "content": "b" * 500},
        ]
        mgr.last_actual_tokens = 100000
        result = await mgr.manage(messages, max_tokens=100)
        assert result[0]["role"] == "system"

    async def test_summarization_error_handling(self):
        client = MagicMock()
        client.complete = AsyncMock(side_effect=Exception("network error"))
        mgr = SlidingWindowManager(client=client, keep_last=2)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "a" * 500},
            {"role": "assistant", "content": "b" * 500},
            {"role": "user", "content": "c" * 500},
            {"role": "assistant", "content": "d" * 500},
        ]
        mgr.last_actual_tokens = 100000
        result = await mgr.manage(messages, max_tokens=100)
        # Should not raise, falls back gracefully
        assert len(result) > 0

    async def test_multiple_summarizations_accumulate(self):
        client = MagicMock()
        client.complete = AsyncMock(return_value=AgentResponse(content="updated summary"))
        mgr = SlidingWindowManager(client=client, keep_last=1)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "a" * 500},
            {"role": "assistant", "content": "b" * 500},
        ]
        mgr.last_actual_tokens = 100000
        await mgr.manage(messages, max_tokens=100)
        assert mgr._summary == "updated summary"


class TestKeyFactsManagerEdge:
    async def test_no_tool_results_in_old(self):
        mgr = KeyFactsManager(keep_last=2)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "a" * 100},
            {"role": "assistant", "content": "b" * 100},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
        ]
        mgr.last_actual_tokens = 100000
        result = await mgr.manage(messages, max_tokens=100)
        # No facts message since no tool results
        facts_msgs = [m for m in result if "Previously collected" in m.get("content", "")]
        assert len(facts_msgs) == 0


class TestEffectiveTokens:
    def test_uses_max_of_estimated_and_actual(self):
        mgr = NoopContextManager()
        mgr.last_actual_tokens = 5000
        messages = [{"role": "user", "content": "hi"}]
        tokens = mgr._effective_tokens(messages)
        assert tokens == 5000  # actual > estimated


class TestEnsureToolIntegrityEdge:
    def test_only_system_messages(self):
        messages = [{"role": "system", "content": "sys"}]
        result = ensure_tool_integrity(messages)
        assert any(m["role"] == "user" for m in result)

    def test_partial_tool_results(self):
        messages = [
            {
                "role": "assistant",
                "content": "calling",
                "tool_calls": [
                    {"id": "tc1", "function": {"name": "a"}},
                    {"id": "tc2", "function": {"name": "b"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "result1"},
            # tc2 result is missing
        ]
        result = ensure_tool_integrity(messages)
        # tc2 should be stripped from assistant's tool_calls
        assistant = [m for m in result if m.get("role") == "assistant"][0]
        if "tool_calls" in assistant:
            ids = {tc["id"] for tc in assistant["tool_calls"]}
            assert "tc2" not in ids

    def test_empty_tool_calls_stripped(self):
        messages = [
            {"role": "assistant", "content": "thinking", "tool_calls": []},
        ]
        result = ensure_tool_integrity(messages)
        assistant = [m for m in result if m.get("role") == "assistant"][0]
        # Empty tool_calls should be converted to content-only message
        assert assistant.get("content") == "thinking"
