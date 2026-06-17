"""Tests for koboi.context module."""

from __future__ import annotations

from koboi.context.manager import (
    NoopContextManager,
    TruncationManager,
    SmartTruncationManager,
    ensure_tool_integrity,
)


def _make_messages(n: int, system: bool = False) -> list[dict]:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": "System prompt"})
    for i in range(n):
        msgs.append({"role": "user", "content": f"Message {i}"})
    return msgs


class TestNoopContextManager:
    async def test_passthrough(self):
        mgr = NoopContextManager()
        msgs = _make_messages(10)
        result = await mgr.manage(msgs, 1000)
        assert len(result) == 10


class TestTruncationManager:
    async def test_truncation(self):
        mgr = TruncationManager(keep_last=5)
        msgs = _make_messages(20)
        result = await mgr.manage(msgs, max_tokens=1)
        assert len(result) <= 5

    async def test_preserves_system(self):
        mgr = TruncationManager(keep_last=3)
        msgs = [{"role": "system", "content": "System"}] + [{"role": "user", "content": f"Msg {i}"} for i in range(10)]
        result = await mgr.manage(msgs, 1000)
        assert result[0]["role"] == "system"


class TestSmartTruncationManager:
    async def test_keeps_first_and_last(self):
        mgr = SmartTruncationManager(keep_last=4)
        msgs = [{"role": "system", "content": "Sys"}]
        msgs.append({"role": "user", "content": "First question"})
        for i in range(20):
            msgs.append({"role": "assistant", "content": f"Answer {i}"})
            msgs.append({"role": "user", "content": f"Q {i}"})
        result = await mgr.manage(msgs, 1000)
        assert result[0]["role"] == "system"


class TestToolIntegrity:
    def test_removes_orphaned_tool_results(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "tool", "tool_call_id": "tc_1", "content": "orphaned"},
            {"role": "user", "content": "World"},
        ]
        result = ensure_tool_integrity(msgs)
        assert all(m["role"] != "tool" for m in result)
        assert "Hello" in result[0]["content"]
        assert "World" in result[0]["content"]
