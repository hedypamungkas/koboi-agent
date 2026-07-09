"""Tests for koboi/loop.py — AgentCore expanded coverage for streaming, skills, guardrails."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from koboi.loop import AgentCore
from koboi.hooks.chain import HookEvent, HookChain, HookContext
from koboi.memory import ConversationMemory
from koboi.tools.registry import ToolRegistry
from koboi.types import AgentResponse, TokenUsage, ToolCall
from koboi.llm.base import LLMClient
from koboi.events import TextDeltaEvent, CompleteEvent, ErrorEvent, ToolResultEvent
from koboi.exceptions import AgentGuardrailError


def _make_response(content=None, tool_calls=None):
    return AgentResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
    )


def _make_tool_call(name="test_tool", arguments="{}"):
    return ToolCall(id="tc_1", name=name, arguments=arguments)


class MockStreamClient(LLMClient):
    def __init__(self, responses=None):
        self.responses = responses or []
        self._idx = 0
        self._model = "mock-stream-model"

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, messages, tools=None, **kwargs):
        if self._idx < len(self.responses):
            r = self.responses[self._idx]
            self._idx += 1
            return r
        return _make_response(content="default")

    async def complete_stream(self, messages, tools=None, **kwargs):
        resp = await self.complete(messages, tools)
        if resp.content:
            yield TextDeltaEvent(content=resp.content)
        yield CompleteEvent(response=resp, content=resp.content or "")

    async def get_embeddings(self, text):
        return None

    async def close(self):
        pass


@pytest.fixture
def memory():
    return ConversationMemory()


@pytest.fixture
def tools():
    registry = ToolRegistry()
    registry.register(
        name="get_weather",
        description="Get weather",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        fn=lambda city: f"Weather in {city}: Sunny",
    )
    return registry


class TestAgentCoreStreaming:
    @pytest.mark.asyncio
    async def test_run_stream_basic(self, memory, tools):
        client = MockStreamClient([_make_response(content="Hello world")])
        core = AgentCore(client=client, memory=memory, tools=tools, max_iterations=5)

        collected = []
        async for event in core.run_stream("hi"):
            collected.append(event)

        assert any(isinstance(e, (TextDeltaEvent, CompleteEvent)) for e in collected)

    @pytest.mark.asyncio
    async def test_run_stream_with_tool_calls(self, memory, tools):
        client = MagicMock()
        call_count = 0

        async def mock_stream(messages, tools_arg=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                resp = _make_response(
                    content="checking", tool_calls=[_make_tool_call("get_weather", '{"city": "Tokyo"}')]
                )
                yield TextDeltaEvent(content="Let me check")
                yield CompleteEvent(response=resp, content="Let me check")
            else:
                resp = _make_response(content="Weather in Tokyo")
                yield CompleteEvent(response=resp, content="Weather in Tokyo")

        client.complete_stream = mock_stream

        core = AgentCore(client=client, memory=memory, tools=tools, max_iterations=5)

        collected = []
        async for event in core.run_stream("What's the weather in Tokyo?"):
            collected.append(event)

        assert any(isinstance(e, CompleteEvent) for e in collected)

    @pytest.mark.asyncio
    async def test_run_stream_max_iterations(self, memory):
        client = MagicMock()

        async def mock_stream(messages, tools_arg=None, **kwargs):
            resp = _make_response(tool_calls=[_make_tool_call()])
            yield CompleteEvent(response=resp, content="not done")

        client.complete_stream = mock_stream

        core = AgentCore(client=client, memory=memory, max_iterations=2)
        collected = []
        async for event in core.run_stream("loop"):
            collected.append(event)

        assert any(isinstance(e, ErrorEvent) for e in collected)

    @pytest.mark.asyncio
    async def test_run_stream_input_guardrail_blocks(self, memory):
        guardrail = AsyncMock()
        guardrail.check.return_value = MagicMock(passed=False, reason="Injection detected", sanitized_content=None)

        client = MockStreamClient([])
        core = AgentCore(client=client, memory=memory, input_guardrail=guardrail)

        collected = []
        async for event in core.run_stream("malicious input"):
            collected.append(event)

        assert len(collected) == 1
        assert isinstance(collected[0], ErrorEvent)

    @pytest.mark.asyncio
    async def test_run_stream_hook_abort(self, memory):
        hook_chain = MagicMock()
        hook_chain.emit = AsyncMock(
            return_value=HookContext(event=HookEvent.PRE_INPUT, abort=True, inject_message="Rejected")
        )

        client = MockStreamClient([])
        core = AgentCore(client=client, memory=memory, hook_chain=hook_chain)

        collected = []
        async for event in core.run_stream("test"):
            collected.append(event)

        assert len(collected) == 1
        assert isinstance(collected[0], ErrorEvent)

    @pytest.mark.asyncio
    async def test_run_stream_null_response(self, memory):
        client = MagicMock()

        async def mock_stream(messages, tools_arg=None, **kwargs):
            yield TextDeltaEvent(content="partial")
            # No CompleteEvent

        client.complete_stream = mock_stream

        core = AgentCore(client=client, memory=memory, max_iterations=1)
        collected = []
        async for event in core.run_stream("test"):
            collected.append(event)

        assert any(isinstance(e, ErrorEvent) for e in collected)


class TestAgentCoreSkills:
    def test_check_skill_activation_match(self, tools):
        skills = MagicMock()
        skills.get.return_value = MagicMock(skill_dir="/skills/coding")

        client = MagicMock()
        mem = ConversationMemory()
        core = AgentCore(client=client, memory=mem, tools=tools, skills=skills)

        result = core._check_skill_activation("Some text [ACTIVATE_SKILL: coding] more")
        assert result is not None
        assert result[0] == "coding"

    def test_check_skill_activation_no_match(self, tools):
        client = MagicMock()
        mem = ConversationMemory()
        core = AgentCore(client=client, memory=mem, tools=tools)
        result = core._check_skill_activation("No skill pattern here")
        assert result is None

    def test_check_skill_activation_no_skills(self, tools):
        client = MagicMock()
        mem = ConversationMemory()
        core = AgentCore(client=client, memory=mem, tools=tools, skills=None)
        result = core._check_skill_activation("[ACTIVATE_SKILL: coding]")
        assert result is None

    @pytest.mark.asyncio
    async def test_run_skill_activation(self, tools):
        skills = MagicMock()
        skills.get.return_value = MagicMock(skill_dir="/skills/coding")
        skills.is_activated.return_value = False
        skills.activate.return_value = "Skill body content"

        client = MagicMock()
        call_count = 0

        async def mock_complete(messages, tools_arg=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_response(content="[ACTIVATE_SKILL: coding]", tool_calls=[])
            return _make_response(content="Done with skill")

        client.complete = mock_complete
        client.get_embeddings = AsyncMock(return_value=None)
        client.close = AsyncMock()

        mem = ConversationMemory()
        core = AgentCore(client=client, memory=mem, tools=tools, skills=skills, max_iterations=3)
        result = await core.run("activate coding skill")
        assert result.success is True


class TestAgentCoreGuardrails:
    @pytest.mark.asyncio
    async def test_run_input_guardrail_sanitizes(self, tools):
        guardrail = AsyncMock()
        guardrail.check.return_value = MagicMock(passed=True, reason="", sanitized_content="clean input")

        client = MagicMock()
        client.complete = AsyncMock(return_value=_make_response(content="ok"))
        client.get_embeddings = AsyncMock(return_value=None)
        client.close = AsyncMock()

        mem = ConversationMemory()
        core = AgentCore(client=client, memory=mem, tools=tools, input_guardrail=guardrail, max_iterations=1)

        result = await core.run("dirty input")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_run_input_guardrail_blocks(self, tools):
        guardrail = AsyncMock()
        guardrail.check.return_value = MagicMock(passed=False, reason="Injection detected", sanitized_content=None)

        client = MagicMock()
        mem = ConversationMemory()
        core = AgentCore(client=client, memory=mem, tools=tools, input_guardrail=guardrail, max_iterations=1)

        with pytest.raises(AgentGuardrailError):
            await core.run("malicious")

    @pytest.mark.asyncio
    async def test_run_output_guardrail_warning(self, tools):
        output_guard = AsyncMock()
        output_guard.check.return_value = MagicMock(passed=False, reason="Contains API key")

        client = MagicMock()
        client.complete = AsyncMock(return_value=_make_response(content="secret data"))
        client.get_embeddings = AsyncMock(return_value=None)
        client.close = AsyncMock()

        mem = ConversationMemory()
        core = AgentCore(client=client, memory=mem, tools=tools, output_guardrail=output_guard, max_iterations=1)

        result = await core.run("get data")
        assert "GUARDRAIL WARNING" in result.content


class TestAgentCoreRateLimiter:
    @pytest.mark.asyncio
    async def test_run_rate_limited_tool(self, tools):
        rl = MagicMock()
        rl.check.return_value = MagicMock(passed=False, reason="Too many calls")

        client = MagicMock()
        call_count = 0

        async def mock_complete(messages, tools_arg=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_response(tool_calls=[_make_tool_call("get_weather", '{"city": "NYC"}')])
            return _make_response(content="Done")

        client.complete = mock_complete

        mem = ConversationMemory()
        core = AgentCore(client=client, memory=mem, tools=tools, rate_limiter=rl, max_iterations=3)

        result = await core.run("check weather")
        assert result.success is True


class TestAgentCoreApproval:
    @pytest.mark.asyncio
    async def test_run_tool_denied(self, tools):
        approval = MagicMock()
        approval.should_approve.return_value = False

        client = MagicMock()
        call_count = 0

        async def mock_complete(messages, tools_arg=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_response(tool_calls=[_make_tool_call("get_weather", '{"city": "NYC"}')])
            return _make_response(content="Can't do that")

        client.complete = mock_complete

        mem = ConversationMemory()
        core = AgentCore(client=client, memory=mem, tools=tools, approval_handler=approval, max_iterations=3)

        result = await core.run("check weather")
        assert result.success is True


class TestAgentCoreReset:
    def test_reset(self, tools):
        mem = ConversationMemory()
        mem.add_user_message("test")
        client = MagicMock()
        core = AgentCore(client=client, memory=mem, tools=tools)
        # After AgentCore init with non-empty memory, core.memory should be mem
        assert core.memory is mem
        assert len(mem) == 1
        core.reset()
        assert len(mem) == 0

    def test_reset_with_rate_limiter(self, tools):
        rl = MagicMock()
        mem = ConversationMemory()
        client = MagicMock()
        core = AgentCore(client=client, memory=mem, tools=tools, rate_limiter=rl)
        core.reset()
        rl.reset.assert_called_once()


class TestAgentCoreEmit:
    @pytest.mark.asyncio
    async def test_emit_sets_context(self, tools):
        chain = HookChain()
        mem = ConversationMemory()
        client = MagicMock()
        core = AgentCore(client=client, memory=mem, tools=tools, hook_chain=chain)

        ctx = await core._emit(HookEvent.SESSION_START)
        assert ctx.event == HookEvent.SESSION_START

    @pytest.mark.asyncio
    async def test_emit_with_inject_message(self, tools):
        hook = MagicMock()
        hook.handles.return_value = [HookEvent.PRE_INPUT]

        async def mock_execute(ctx):
            ctx.inject_message = "injected context"
            return ctx

        hook.execute = mock_execute
        chain = HookChain([hook])

        mem = ConversationMemory()
        mem.add_user_message("seed")  # Make memory truthy so AgentCore uses it
        client = MagicMock()
        core = AgentCore(client=client, memory=mem, tools=tools, hook_chain=chain)

        ctx = await core._emit(HookEvent.PRE_INPUT)
        assert ctx.inject_message == "injected context"
        # _emit adds inject_message as context message via add_context_message
        internal_msgs = mem._messages
        assert any("injected context" in m.get("content", "") for m in internal_msgs)


class TestAgentCoreChat:
    @pytest.mark.asyncio
    async def test_chat_delegates_to_run(self, tools):
        mem = ConversationMemory()
        client = MagicMock()
        client.complete = AsyncMock(return_value=_make_response(content="hi back"))
        client.get_embeddings = AsyncMock(return_value=None)
        client.close = AsyncMock()

        core = AgentCore(client=client, memory=mem, tools=tools, max_iterations=1)
        result = await core.chat("hi")
        assert result.success is True
        assert result.content == "hi back"


class TestAgentCoreContextManagement:
    @pytest.mark.asyncio
    async def test_managed_messages_with_context_manager(self, tools):
        mem = ConversationMemory()
        ctx_mgr = AsyncMock()
        ctx_mgr.manage.return_value = [{"role": "user", "content": "managed"}]

        client = MagicMock()
        core = AgentCore(client=client, memory=mem, tools=tools, context_manager=ctx_mgr)

        msgs = await core._get_managed_messages()
        assert len(msgs) == 1
        ctx_mgr.manage.assert_called_once()

    @pytest.mark.asyncio
    async def test_managed_messages_no_context_manager(self, tools):
        mem = ConversationMemory()
        mem.add_user_message("seed")
        client = MagicMock()
        core = AgentCore(client=client, memory=mem, tools=tools)
        mem.add_user_message("hello")
        msgs = await core._get_managed_messages()
        assert len(msgs) >= 1

    @pytest.mark.asyncio
    async def test_augment_memory_no_augmentation(self, tools):
        mem = ConversationMemory()
        client = MagicMock()
        core = AgentCore(client=client, memory=mem, tools=tools)
        result = await core._augment_memory("test")
        assert result == "test"

    @pytest.mark.asyncio
    async def test_augment_memory_with_augmentation(self, tools):
        mem = ConversationMemory()
        aug = AsyncMock()
        aug.augment_for_memory.return_value = "augmented test"

        client = MagicMock()
        core = AgentCore(client=client, memory=mem, tools=tools, augmentation=aug)

        result = await core._augment_memory("test")
        assert result == "augmented test"

    @pytest.mark.asyncio
    async def test_augment_llm_no_augmentation(self, tools):
        mem = ConversationMemory()
        client = MagicMock()
        core = AgentCore(client=client, memory=mem, tools=tools)
        msgs = [{"role": "user", "content": "test"}]
        result = await core._augment_llm(msgs)
        assert result == msgs

    def test_format_tool_calls_for_memory(self, tools):
        mem = ConversationMemory()
        client = MagicMock()
        core = AgentCore(client=client, memory=mem, tools=tools)
        tcs = [ToolCall(id="tc1", name="search", arguments='{"q": "test"}')]
        result = core._format_tool_calls_for_memory(tcs)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "search"


from koboi.loop import _extract_text


class TestExtractText:
    def test_text_blocks(self):
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"},
        ]
        assert _extract_text(content) == "Hello World"

    def test_image_blocks_ignored(self):
        content = [
            {"type": "text", "text": "Look at"},
            {"type": "image", "source": {"data": "base64"}},
            {"type": "text", "text": "this"},
        ]
        assert _extract_text(content) == "Look at this"

    def test_empty_content(self):
        assert _extract_text([]) == ""

    def test_no_text_blocks(self):
        content = [{"type": "image", "source": {}}]
        assert _extract_text(content) == ""

    def test_string_block_skipped(self):
        content = ["plain string", {"type": "text", "text": "real"}]
        assert _extract_text(content) == "real"


class TestAgentCoreVerbose:
    def test_log_verbose(self, tools):
        mem = ConversationMemory()
        client = MagicMock()
        core = AgentCore(client=client, memory=mem, tools=tools, verbose=True)
        with patch("koboi.loop._log") as mock_log:
            core._log("test message")
            mock_log.debug.assert_called_once_with("test message")

    def test_log_not_verbose(self, tools):
        mem = ConversationMemory()
        client = MagicMock()
        core = AgentCore(client=client, memory=mem, tools=tools, verbose=False)
        with patch("koboi.loop._log") as mock_log:
            core._log("test message")
            mock_log.debug.assert_not_called()


class TestAgentCoreAudit:
    def test_audit_with_trail(self, tools):
        mem = ConversationMemory()
        client = MagicMock()
        core = AgentCore(client=client, memory=mem, tools=tools)
        core.audit_trail = MagicMock()
        core._audit("test_event")
        core.audit_trail.record.assert_called_once()

    def test_audit_without_trail(self, tools):
        mem = ConversationMemory()
        client = MagicMock()
        core = AgentCore(client=client, memory=mem, tools=tools)
        core.audit_trail = None
        core._audit("test_event")  # should not raise


class TestAgentCoreSkillsDiscovery:
    @pytest.mark.asyncio
    async def test_skills_discovery_appended(self, tools):
        mem = ConversationMemory()
        mem.add_user_message("hello")
        client = MagicMock()
        skills = MagicMock()
        skills.get_routed_discovery_prompt.return_value = "\n\nSkills: code_review"

        core = AgentCore(client=client, memory=mem, tools=tools, skills=skills)
        core._last_user_message = "hello"
        await core._get_managed_messages()
        skills.get_routed_discovery_prompt.assert_called_once_with("hello")

    @pytest.mark.asyncio
    async def test_skills_discovery_no_user_message(self, tools):
        mem = ConversationMemory()
        client = MagicMock()
        skills = MagicMock()
        skills.get_discovery_prompt.return_value = "\n\nAll skills"

        core = AgentCore(client=client, memory=mem, tools=tools, skills=skills)
        core._last_user_message = ""
        await core._get_managed_messages()
        skills.get_discovery_prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_skills_discovery_already_appended(self, tools):
        mem = ConversationMemory()
        mem.add_user_message("hello")
        client = MagicMock()
        skills = MagicMock()

        core = AgentCore(client=client, memory=mem, tools=tools, skills=skills)
        core._skills_discovery_appended = True
        await core._get_managed_messages()
        skills.get_routed_discovery_prompt.assert_not_called()

    @pytest.mark.asyncio
    async def test_skills_discovery_no_system_message(self, tools):
        mem = ConversationMemory()
        mem.add_user_message("hello")
        client = MagicMock()
        skills = MagicMock()
        skills.get_routed_discovery_prompt.return_value = "\n\nSkills"

        core = AgentCore(client=client, memory=mem, tools=tools, skills=skills)
        core._last_user_message = "hello"
        msgs = await core._get_managed_messages()
        # Should insert system message at position 0
        assert msgs[0]["role"] == "system"


class TestAgentCoreApprovalAsync:
    @pytest.mark.asyncio
    async def test_approval_async_should_approve(self, tools):
        approval = MagicMock()
        approval.should_approve = AsyncMock(return_value=True)

        client = MagicMock()
        call_count = 0

        async def mock_complete(messages, tools_arg=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_response(tool_calls=[_make_tool_call("get_weather", '{"city": "NYC"}')])
            return _make_response(content="Approved")

        client.complete = mock_complete

        mem = ConversationMemory()
        core = AgentCore(client=client, memory=mem, tools=tools, approval_handler=approval, max_iterations=3)

        result = await core.run("check weather")
        approval.should_approve.assert_called_once()
        assert result.success is True


class TestAgentCoreRunStreamModeBlocked:
    @pytest.mark.asyncio
    async def test_stream_mode_blocked_tool(self, tools):
        hook = MagicMock()
        hook.handles.return_value = [HookEvent.PRE_TOOL_USE]

        async def mock_execute(ctx):
            ctx.metadata["mode_blocked"] = True
            ctx.metadata["mode_block_reason"] = "Not in act mode"
            return ctx

        hook.execute = mock_execute
        chain = HookChain([hook])

        client = MagicMock()
        call_count = 0

        async def mock_stream(messages, tools_arg=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                resp = _make_response(tool_calls=[_make_tool_call("get_weather", '{"city": "NYC"}')])
                yield CompleteEvent(response=resp, content="")
            else:
                resp = _make_response(content="Done")
                yield CompleteEvent(response=resp, content="Done")

        client.complete_stream = mock_stream

        mem = ConversationMemory()
        core = AgentCore(client=client, memory=mem, tools=tools, hook_chain=chain, max_iterations=3)

        collected = []
        async for event in core.run_stream("test"):
            collected.append(event)
        assert any(isinstance(e, (CompleteEvent, ToolResultEvent)) for e in collected)

        # Regression: the blocked tool must NOT appear in tools_used (the streaming
        # sibling of the tool_calls_made fix -- _stream_tools_used is gated on
        # `not pipeline_result.skipped` in loop.py's run_stream).
        complete = next(e for e in collected if isinstance(e, CompleteEvent))
        assert "get_weather" not in complete.tools_used
