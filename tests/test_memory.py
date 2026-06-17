"""Tests for koboi.memory module."""
from __future__ import annotations

from koboi.memory import ConversationMemory


class TestConversationMemory:
    def test_add_user_message(self):
        mem = ConversationMemory()
        mem.add_user_message("Hello")
        msgs = mem.get_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello"

    def test_add_assistant_message(self):
        mem = ConversationMemory()
        mem.add_assistant_message("Hi there")
        msgs = mem.get_messages()
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["content"] == "Hi there"

    def test_add_tool_result(self):
        mem = ConversationMemory()
        mem.add_tool_result("tc_1", "result data")
        msgs = mem.get_messages()
        assert msgs[0]["role"] == "tool"
        assert msgs[0]["tool_call_id"] == "tc_1"

    def test_system_prompt_prepended(self):
        mem = ConversationMemory(system_prompt="You are helpful")
        mem.add_user_message("Hi")
        msgs = mem.get_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are helpful"
        assert msgs[1]["role"] == "user"

    def test_clear(self):
        mem = ConversationMemory()
        mem.add_user_message("Hello")
        mem.add_user_message("World")
        assert len(mem) == 2
        mem.clear()
        assert len(mem) == 0

    def test_conversation_flow(self):
        mem = ConversationMemory(system_prompt="System")
        mem.add_user_message("Q1")
        mem.add_assistant_message("A1")
        mem.add_user_message("Q2")
        msgs = mem.get_messages()
        assert len(msgs) == 4
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user", "assistant", "user"]

    def test_context_message(self):
        mem = ConversationMemory()
        mem.add_context_message("<skill>body</skill>", label="test_skill")
        msgs = mem.get_messages()
        assert msgs[0]["role"] == "system"
        assert "skill" in msgs[0]["content"]
