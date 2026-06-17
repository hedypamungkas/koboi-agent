"""Tests for koboi/context/manager.py -- Context management."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from koboi.context.manager import (
    ensure_tool_integrity,
    NoopContextManager,
    TruncationManager,
    SmartTruncationManager,
    KeyFactsManager,
)


class TestEnsureToolIntegrity:
    def test_empty_messages(self):
        result = ensure_tool_integrity([])
        # Adds synthetic user message for empty input
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_passthrough_clean(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = ensure_tool_integrity(msgs)
        assert len(result) == 2

    def test_removes_orphaned_tool_result(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "tool", "tool_call_id": "nonexistent", "content": "result"},
        ]
        result = ensure_tool_integrity(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_keeps_valid_tool_result(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "tc1", "function": {"name": "read"}}]},
            {"role": "tool", "tool_call_id": "tc1", "content": "file content"},
        ]
        result = ensure_tool_integrity(msgs)
        # user + assistant + tool
        assert len(result) == 3

    def test_fixes_empty_tool_calls(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "thinking", "tool_calls": []},
        ]
        result = ensure_tool_integrity(msgs)
        # Empty tool_calls is falsy, passes through without the strip branch
        assistant_msgs = [m for m in result if m.get("role") == "assistant"]
        assert len(assistant_msgs) == 1

    def test_merges_consecutive_same_role(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
        result = ensure_tool_integrity(msgs)
        user_msgs = [m for m in result if m.get("role") == "user"]
        assert len(user_msgs) == 1
        assert "first" in user_msgs[0]["content"]
        assert "second" in user_msgs[0]["content"]

    def test_does_not_merge_system(self):
        msgs = [
            {"role": "system", "content": "sys1"},
            {"role": "system", "content": "sys2"},
        ]
        result = ensure_tool_integrity(msgs)
        system_msgs = [m for m in result if m.get("role") == "system"]
        assert len(system_msgs) == 2

    def test_inserts_synthetic_user_if_starts_with_assistant(self):
        msgs = [
            {"role": "assistant", "content": "hello"},
        ]
        result = ensure_tool_integrity(msgs)
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "[continuing analysis]"

    def test_only_system_messages(self):
        msgs = [
            {"role": "system", "content": "sys"},
        ]
        result = ensure_tool_integrity(msgs)
        assert any(m["role"] == "user" for m in result)

    def test_handles_missing_tool_results_for_calls(self):
        msgs = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "tc1", "function": {"name": "read"}}]},
        ]
        result = ensure_tool_integrity(msgs)
        # Should have content indicating tool was called
        assistant_msgs = [m for m in result if m.get("role") == "assistant"]
        assert len(assistant_msgs) == 1
        assert "read" in assistant_msgs[0].get("content", "")


class TestNoopContextManager:
    @pytest.mark.asyncio
    async def test_passthrough(self):
        mgr = NoopContextManager()
        msgs = [{"role": "user", "content": "hi"}]
        result = await mgr.manage(msgs, max_tokens=1000)
        assert result == msgs


class TestTruncationManager:
    @pytest.mark.asyncio
    async def test_no_truncation_needed(self):
        mgr = TruncationManager(keep_last=10)
        msgs = [{"role": "user", "content": "hi"}]
        result = await mgr.manage(msgs, max_tokens=100000)
        assert result == msgs

    @pytest.mark.asyncio
    async def test_truncates_old_messages(self):
        mgr = TruncationManager(keep_last=2)
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "resp1"},
            {"role": "user", "content": "msg2"},
            {"role": "assistant", "content": "resp2"},
        ]
        result = await mgr.manage(msgs, max_tokens=10)
        assert len(result) < len(msgs)
        # System should be preserved
        assert any(m.get("role") == "system" for m in result)

    @pytest.mark.asyncio
    async def test_with_logger(self):
        logger = MagicMock()
        mgr = TruncationManager(logger=logger, keep_last=2)
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "a" * 100},
            {"role": "assistant", "content": "b" * 100},
            {"role": "user", "content": "c" * 100},
            {"role": "assistant", "content": "d" * 100},
        ]
        await mgr.manage(msgs, max_tokens=10)
        logger.log_context_management.assert_called_once()


class TestSmartTruncationManager:
    @pytest.mark.asyncio
    async def test_keeps_first_user(self):
        mgr = SmartTruncationManager(keep_last=1)
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first user"},
            {"role": "assistant", "content": "resp1"},
            {"role": "user", "content": "second user"},
            {"role": "assistant", "content": "resp2"},
        ]
        result = await mgr.manage(msgs, max_tokens=10)
        # Should have system + first user + recent
        assert any(m.get("content") == "first user" for m in result)

    @pytest.mark.asyncio
    async def test_no_truncation_needed(self):
        mgr = SmartTruncationManager(keep_last=10)
        msgs = [{"role": "user", "content": "hi"}]
        result = await mgr.manage(msgs, max_tokens=100000)
        assert result == msgs


class TestKeyFactsManager:
    @pytest.mark.asyncio
    async def test_no_truncation_needed(self):
        mgr = KeyFactsManager(keep_last=10)
        msgs = [{"role": "user", "content": "hi"}]
        result = await mgr.manage(msgs, max_tokens=100000)
        assert result == msgs

    @pytest.mark.asyncio
    async def test_extracts_facts(self):
        mgr = KeyFactsManager(keep_last=1)
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "tool", "tool_call_id": "tc1", "content": "fact 1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        result = await mgr.manage(msgs, max_tokens=10)
        # Should have system + facts + recent
        assert any("fact" in m.get("content", "") for m in result)
