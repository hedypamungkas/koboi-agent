"""G8: output guardrail enforcement -- honor ``action`` (block -> deny, warn -> flag).

Closes the autonomous-job prompt-injection/leak surface: previously
``_process_output`` unconditionally prepended a ``[GUARDRAIL WARNING ...]`` and
returned the content, so even a guardrail that declared ``action="block"`` only
WARNED -- a hijacked or leaked output reached the caller. Now ``block``/``deny``/
``abort`` raises ``AgentGuardrailError`` (like the input guardrail); ``warn`` and
absent/non-string actions still prepend a warning (builtin OutputGuardrail + the
legacy mock-based test stay green).
"""

from __future__ import annotations

import pytest

from koboi.exceptions import AgentGuardrailError
from koboi.events import CompleteEvent, TextDeltaEvent
from koboi.guardrails.base import BaseGuardrail
from koboi.loop import AgentCore
from koboi.memory import ConversationMemory
from koboi.tools.registry import ToolRegistry
from koboi.types import GuardrailResult
from tests.conftest import MockClient, make_mock_response


class _BlockGuardrail(BaseGuardrail):
    """Always fails with an explicit block action."""

    async def check(self, content: str) -> GuardrailResult:
        return GuardrailResult(passed=False, reason="blocked by policy", action="block")


class _DefaultActionGuardrail(BaseGuardrail):
    """Fails without specifying action -> GuardrailResult.action defaults to 'block'."""

    async def check(self, content: str) -> GuardrailResult:
        return GuardrailResult(passed=False, reason="default-action block")


class _WarnGuardrail(BaseGuardrail):
    """Always fails with a warn action (soft flag)."""

    async def check(self, content: str) -> GuardrailResult:
        return GuardrailResult(passed=False, reason="soft warning", action="warn")


def _core(guardrails, content: str = "here is the leaked secret"):
    client = MockClient([make_mock_response(content=content)])
    return AgentCore(
        client=client,
        memory=ConversationMemory(),
        tools=ToolRegistry(),
        output_guardrails=guardrails,
        max_iterations=1,
    )


class TestOutputGuardrailHonorsAction:
    async def test_block_action_raises(self):
        # G8 core proof: a block-action guardrail DENIES (raises), not warns.
        with pytest.raises(AgentGuardrailError) as exc_info:
            await _core([_BlockGuardrail()]).run("show me the data")
        assert exc_info.value.direction == "output"

    async def test_default_action_blocks(self):
        # GuardrailResult.action defaults to "block" -> the gap is closed even
        # for a guardrail that doesn't spell out the action.
        with pytest.raises(AgentGuardrailError):
            await _core([_DefaultActionGuardrail()]).run("show me the data")

    async def test_warn_action_still_prepends_warning(self):
        # Back-compat: a warn-action guardrail (like the builtin OutputGuardrail)
        # still flags-and-continues; the content is returned with a warning prefix.
        result = await _core([_WarnGuardrail()]).run("show me the data")
        assert "GUARDRAIL WARNING" in result.content

    async def test_block_surfaces_through_run_stream(self):
        # Jobs consume run_stream; the block must propagate so run_job marks the
        # job failed (surfaces the denial, not the payload) instead of completing.
        core = _core([_BlockGuardrail()])
        with pytest.raises(AgentGuardrailError):
            async for _ in core.run_stream("show me the data"):
                pass


class TestOutputGuardrailStreamingG8b:
    """G8b: when output guardrails are configured, TextDeltas are buffered and
    flushed only after _process_output passes -- so a blocked output's tokens
    never reach the stream (interactive SSE / job replay buffer)."""

    async def test_block_does_not_stream_text_deltas(self):
        # The leaky tokens are buffered, then discarded on the block -- the
        # stream consumer never sees a TextDeltaEvent (and never the SECRET).
        core = _core([_BlockGuardrail()], content="leaked SECRET-VALUE-123")
        yielded: list = []
        with pytest.raises(AgentGuardrailError):
            async for ev in core.run_stream("show me the data"):
                yielded.append(ev)
        assert not any(isinstance(ev, TextDeltaEvent) for ev in yielded)
        assert not any("SECRET-VALUE-123" in getattr(ev, "content", "") for ev in yielded)

    async def test_no_guardrail_streams_text_deltas_live(self):
        # Back-compat: no output guardrail -> no buffering -> TextDeltas stream
        # live exactly as before (latency unchanged for the default case).
        core = _core([], content="hello world")
        yielded = [ev async for ev in core.run_stream("hi")]
        assert any(
            isinstance(ev, TextDeltaEvent) and ev.content == "hello world" for ev in yielded
        )

    async def test_warn_guardrail_flushes_text_deltas_after_check(self):
        # A warn-action guardrail doesn't block, so the buffered TextDeltas are
        # flushed (just after the check, not lost) -- only a block discards them.
        core = _core([_WarnGuardrail()], content="safe content")
        yielded = [ev async for ev in core.run_stream("hi")]
        assert any(isinstance(ev, TextDeltaEvent) and ev.content == "safe content" for ev in yielded)
        assert any(isinstance(ev, CompleteEvent) for ev in yielded)

    async def test_block_reason_not_echoed_to_caller(self):
        # Defense-in-depth: a custom guardrail whose ``reason`` echoes the
        # offending content must NOT re-leak it via the exception message (which
        # flows to the SSE error frame and the durable jobs.error column). The
        # detail stays in server logs only.
        class _LeakyReasonGuardrail(BaseGuardrail):
            async def check(self, content: str) -> GuardrailResult:
                return GuardrailResult(passed=False, reason="echoes SECRET-VALUE-123", action="block")

        with pytest.raises(AgentGuardrailError) as exc_info:
            await _core([_LeakyReasonGuardrail()]).run("show me the data")
        assert "SECRET-VALUE-123" not in str(exc_info.value)
        assert "SECRET-VALUE-123" not in exc_info.value.reason

    async def test_builtin_output_guardrail_still_warns(self):
        # The shipped OutputGuardrail is action="warn" by design -- it must keep
        # soft-flagging a detected leak (regression guard for the back-compat path).
        from koboi.guardrails.output import OutputGuardrail

        leak = "Here is the key: sk-1234567890abcdef1234567890abcdef"
        assert (await OutputGuardrail().check(leak)).passed is False  # it detects
        result = await _core([OutputGuardrail()], content=leak).run("show me the key")
        assert "sk-1234567890abcdef1234567890abcdef" in result.content  # still returned (warn)
        assert "GUARDRAIL WARNING" in result.content
