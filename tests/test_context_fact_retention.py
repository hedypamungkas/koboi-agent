"""Tests for mid-conversation fact retention in smart_truncation (#6) + key_facts (#7)."""

from __future__ import annotations


from koboi.context.manager import KeyFactsManager, SmartTruncationManager


class TestSmartTruncationFactRetention:
    async def test_mid_conversation_user_fact_preserved(self):
        SECRET = "MY FLIGHT IS BA2490 CONFIRMATION XYZ789"
        msgs = [{"role": "system", "content": "sys"}]
        msgs.append({"role": "user", "content": "first user (anchored)"})
        msgs.append({"role": "user", "content": SECRET})  # mid-convo fact
        msgs.append({"role": "assistant", "content": "ack"})
        for i in range(12):
            msgs.append({"role": "user", "content": f"chatter {i}"})
            msgs.append({"role": "assistant", "content": f"reply {i}"})

        mgr = SmartTruncationManager(keep_last=6)
        result = await mgr.manage(msgs, max_tokens=10)  # force compaction
        blob = " ".join(m.get("content", "") for m in result)
        assert SECRET in blob  # issue #6: was dropped before

    async def test_first_user_still_anchored(self):
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "FIRST"}]
        for i in range(10):
            msgs.append({"role": "user", "content": f"u{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})
        mgr = SmartTruncationManager(keep_last=4)
        result = await mgr.manage(msgs, max_tokens=10)
        assert any(m.get("content") == "FIRST" for m in result)

    async def test_truncation_caps_earlier_user_lines(self):
        long_msg = "X" * 1000
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "first"}]
        msgs.append({"role": "user", "content": long_msg})
        for i in range(10):
            msgs.append({"role": "user", "content": f"u{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})
        mgr = SmartTruncationManager(keep_last=4, summarization_truncation=50)
        result = await mgr.manage(msgs, max_tokens=10)
        note = next((m for m in result if "Earlier user messages" in m.get("content", "")), None)
        assert note is not None
        # the 1000-char message must be capped to 50 in the note
        assert "X" * 51 not in note["content"]
        assert "X" * 50 in note["content"]


class TestKeyFactsUserAssistant:
    async def test_user_assistant_tool_all_promoted(self):
        msgs = [{"role": "system", "content": "sys"}]
        msgs.append({"role": "user", "content": "USER_FACT: balance=42000"})
        msgs.append({"role": "assistant", "content": "ASSISTANT_REASONING"})
        msgs.append({"role": "tool", "tool_call_id": "t1", "content": "TOOL_RESULT: ok"})
        for i in range(4):
            msgs.append({"role": "user", "content": f"q{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})

        mgr = KeyFactsManager(keep_last=4)
        result = await mgr.manage(msgs, max_tokens=10)
        blob = " ".join(m.get("content", "") for m in result)
        assert "USER_FACT" in blob
        assert "ASSISTANT_REASONING" in blob
        assert "TOOL_RESULT" in blob

    async def test_tool_content_untruncated_by_default(self):
        long_tool = "Z" * 500
        msgs = [{"role": "system", "content": "sys"}, {"role": "tool", "tool_call_id": "t", "content": long_tool}]
        for i in range(4):
            msgs.append({"role": "user", "content": f"q{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})
        mgr = KeyFactsManager(keep_last=4)
        result = await mgr.manage(msgs, max_tokens=10)
        blob = " ".join(m.get("content", "") for m in result)
        assert long_tool in blob  # full content preserved (no default truncation)

    async def test_truncation_when_configured(self):
        long_tool = "Z" * 500
        msgs = [{"role": "system", "content": "sys"}, {"role": "tool", "tool_call_id": "t", "content": long_tool}]
        for i in range(4):
            msgs.append({"role": "user", "content": f"q{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})
        mgr = KeyFactsManager(keep_last=4, summarization_truncation=40)
        result = await mgr.manage(msgs, max_tokens=10)
        blob = " ".join(m.get("content", "") for m in result)
        assert "Z" * 41 not in blob
        assert "Z" * 40 in blob
