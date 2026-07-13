"""A4: a POST_OUTPUT hook veto actually halts.

Previously ``_process_output`` emitted ``POST_OUTPUT`` and discarded the returned
``HookContext``, so a hook setting ``ctx.abort = True`` was silently ignored --
the output was persisted to memory and returned regardless. Now the POST_OUTPUT
emit mirrors the PRE_INPUT pattern (``loop.py``): capture ``ctx``, raise
``AgentAbortedError`` on ``ctx.abort`` before the memory write.

Streaming edge (documented, not fixed): on ``run_stream`` with no output
guardrails (``should_buffer=False``), ``TextDeltaEvent``\\ s are already yielded
live before POST_OUTPUT fires, so a veto cannot un-send streamed tokens -- it
only skips the memory write and raises. This matches the existing G8 output-
guardrail streaming semantics (``loop.py:696-697``).
"""

from __future__ import annotations

import pytest

from koboi.exceptions import AgentAbortedError
from koboi.hooks.chain import Hook, HookChain, HookContext, HookEvent
from koboi.loop import AgentCore
from koboi.memory import ConversationMemory
from koboi.tools.registry import ToolRegistry
from tests.conftest import MockClient, make_mock_response


class _AbortOnPostOutput(Hook):
    """POST_OUTPUT subscriber that vetoes the output."""

    def handles(self) -> list[HookEvent]:
        return [HookEvent.POST_OUTPUT]

    async def execute(self, ctx: HookContext) -> HookContext:
        ctx.abort = True
        ctx.inject_message = "vetoed by post-output hook"
        return ctx


class _NoOpPostOutput(Hook):
    """POST_OUTPUT subscriber that does nothing (control)."""

    def handles(self) -> list[HookEvent]:
        return [HookEvent.POST_OUTPUT]

    async def execute(self, ctx: HookContext) -> HookContext:
        return ctx


class TestPostOutputHookAbort:
    async def test_abort_raises_and_skips_memory(self):
        core = AgentCore(
            client=MockClient([make_mock_response(content="hallucinated answer")]),
            memory=ConversationMemory(),
            tools=ToolRegistry(),
            hook_chain=HookChain([_AbortOnPostOutput()]),
            max_iterations=1,
        )
        with pytest.raises(AgentAbortedError) as exc_info:
            await core.run("anything")
        assert "vetoed by post-output hook" in str(exc_info.value)
        # The aborted output must NOT be persisted to conversation memory.
        msgs = core.memory.get_messages()
        assert not any(m.get("role") == "assistant" for m in msgs), (
            "aborted output must not persist to memory"
        )

    async def test_no_abort_still_persists(self):
        # Control: a POST_OUTPUT hook that does NOT abort -> output persists normally.
        core = AgentCore(
            client=MockClient([make_mock_response(content="ok answer")]),
            memory=ConversationMemory(),
            tools=ToolRegistry(),
            hook_chain=HookChain([_NoOpPostOutput()]),
            max_iterations=1,
        )
        result = await core.run("q")
        assert result.content == "ok answer"
        assert any(m.get("role") == "assistant" for m in core.memory.get_messages())

    async def test_abort_on_streaming_raises_raw(self):
        # On run_stream the POST_OUTPUT abort raises AgentAbortedError out of the
        # async generator (not as an ErrorEvent) -- identical to AgentGuardrailError
        # from _process_output (G8). Already-streamed TextDeltas are consumed first
        # (cannot un-send); memory write is skipped.
        core = AgentCore(
            client=MockClient([make_mock_response(content="hallucinated answer")]),
            memory=ConversationMemory(),
            tools=ToolRegistry(),
            hook_chain=HookChain([_AbortOnPostOutput()]),
            max_iterations=1,
        )
        with pytest.raises(AgentAbortedError):
            async for _ in core.run_stream("anything"):
                pass
        assert not any(m.get("role") == "assistant" for m in core.memory.get_messages())
