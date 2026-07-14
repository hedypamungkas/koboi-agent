"""koboi/loop.py -- branch coverage for AgentCore accessors + managed-messages injection."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from koboi.loop import AgentCore
from koboi.memory import ConversationMemory
from koboi.tools.registry import ToolRegistry


def _mock_client() -> MagicMock:
    c = MagicMock()
    c.model = "mock-model"
    return c


def _core(**kw) -> AgentCore:
    base = dict(client=_mock_client(), memory=ConversationMemory(), tools=ToolRegistry(), max_iterations=3)
    base.update(kw)
    return AgentCore(**base)


class TestGuardrailAccessors:
    def test_input_get_set(self):
        agent = _core()
        assert agent.input_guardrail is None  # empty -> None
        g = MagicMock(name="ing")
        agent.input_guardrail = g
        assert agent.input_guardrail is g
        agent.input_guardrail = None  # setter else-branch
        assert agent.input_guardrail is None

    def test_output_get_set(self):
        agent = _core()
        assert agent.output_guardrail is None
        g = MagicMock(name="outg")
        agent.output_guardrail = g
        assert agent.output_guardrail is g
        agent.output_guardrail = None
        assert agent.output_guardrail is None


class TestSkillsDiscoveryInjection:
    async def test_appended_to_existing_system_message(self):
        skills = MagicMock()
        skills.get_discovery_prompt.return_value = "DISCOVERY"
        agent = _core(memory=ConversationMemory(system_prompt="SYS"), skills=skills)
        msgs = await agent._get_managed_messages()
        assert any(m["role"] == "system" and "DISCOVERY" in m["content"] for m in msgs)
        assert agent._skills_discovery_appended is True

    async def test_inserted_when_no_system_message(self):
        skills = MagicMock()
        skills.get_routed_discovery_prompt.return_value = "DISCOVERY"
        agent = _core(memory=ConversationMemory(system_prompt=None), skills=skills)
        agent._last_user_message = "hello"
        msgs = await agent._get_managed_messages()
        assert msgs and msgs[0]["role"] == "system" and msgs[0]["content"] == "DISCOVERY"

    async def test_no_skills_no_change(self):
        agent = _core()
        msgs = await agent._get_managed_messages()
        assert isinstance(msgs, list)


class TestProactiveBlock:
    async def test_none_returns_empty(self):
        agent = _core()  # proactive_memory is None
        assert await agent._proactive_block("q") == ""

    async def test_core_block_and_recall_appended_to_system(self):
        pm = MagicMock()
        pm.core_block_enabled = True
        pm.recall_enabled = True
        pm.get_core_block.return_value = "CORE"
        pm.recall = AsyncMock(return_value="FACTS")
        agent = _core(memory=ConversationMemory(system_prompt="SYS"), proactive_memory=pm)
        agent._last_user_message = "q"
        msgs = await agent._get_managed_messages()
        sys_msg = next(m for m in msgs if m["role"] == "system")
        assert "CORE" in sys_msg["content"] and "FACTS" in sys_msg["content"]

    async def test_recall_exception_swallowed(self):
        pm = MagicMock()
        pm.core_block_enabled = False
        pm.recall_enabled = True
        pm.recall = AsyncMock(side_effect=RuntimeError("embed fail"))
        agent = _core(memory=ConversationMemory(system_prompt="SYS"), proactive_memory=pm)
        agent._last_user_message = "q"
        block = await agent._proactive_block("q")
        assert block == ""  # recall failed -> no parts

    async def test_proactive_inserted_when_no_system(self):
        pm = MagicMock()
        pm.core_block_enabled = True
        pm.recall_enabled = False
        pm.get_core_block.return_value = "CORE"
        agent = _core(memory=ConversationMemory(system_prompt=None), proactive_memory=pm)
        agent._last_user_message = "q"
        msgs = await agent._get_managed_messages()
        assert msgs[0]["role"] == "system" and msgs[0]["content"] == "CORE"
