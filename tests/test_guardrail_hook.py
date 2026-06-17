"""Tests for koboi/hooks/guardrail_hook.py — GuardrailHook (0% → >85%)."""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock
from collections.abc import Awaitable

import pytest

from koboi.hooks.chain import HookContext, HookEvent
from koboi.hooks.guardrail_hook import GuardrailHook
from koboi.types import GuardrailResult


class MockInputGuardrail:
    """Mock input guardrail for testing."""

    def __init__(self, passed=True, reason="OK", action="allow"):
        self._passed = passed
        self._reason = reason
        self._action = action

    async def check(self, text: str) -> GuardrailResult:
        return GuardrailResult(passed=self._passed, reason=self._reason, action=self._action)


class MockOutputGuardrail:
    """Mock output guardrail for testing."""

    def __init__(self, passed=True, reason="OK", action="allow"):
        self._passed = passed
        self._reason = reason
        self._action = action

    async def check(self, text: str) -> GuardrailResult:
        return GuardrailResult(passed=self._passed, reason=self._reason, action=self._action)


class TestGuardrailHookHandles:
    def test_handles_empty_when_both_guardrails_none(self):
        """handles() should return empty list when both guardrails are None."""
        hook = GuardrailHook(input_guardrail=None, output_guardrail=None)
        assert hook.handles() == []

    def test_handles_pre_input_when_input_guardrail_provided(self):
        """handles() should return PRE_INPUT when only input_guardrail provided."""
        mock_guardrail = MockInputGuardrail()
        hook = GuardrailHook(input_guardrail=mock_guardrail, output_guardrail=None)
        assert hook.handles() == [HookEvent.PRE_INPUT]

    def test_handles_post_output_when_output_guardrail_provided(self):
        """handles() should return POST_OUTPUT when only output_guardrail provided."""
        mock_guardrail = MockOutputGuardrail()
        hook = GuardrailHook(input_guardrail=None, output_guardrail=mock_guardrail)
        assert hook.handles() == [HookEvent.POST_OUTPUT]

    def test_handles_both_when_both_guardrails_provided(self):
        """handles() should return both events when both guardrails provided."""
        mock_input = MockInputGuardrail()
        mock_output = MockOutputGuardrail()
        hook = GuardrailHook(input_guardrail=mock_input, output_guardrail=mock_output)
        assert set(hook.handles()) == {HookEvent.PRE_INPUT, HookEvent.POST_OUTPUT}


class TestGuardrailHookInputCheck:
    async def test_input_check_passthrough_when_no_messages(self):
        """Should passthrough when no messages available."""
        mock_guardrail = MockInputGuardrail()
        hook = GuardrailHook(input_guardrail=mock_guardrail)
        ctx = HookContext(event=HookEvent.PRE_INPUT, messages=None)
        result = await hook.execute(ctx)
        assert result is ctx

    async def test_input_check_passthrough_when_empty_messages(self):
        """Should passthrough when messages list is empty."""
        mock_guardrail = MockInputGuardrail()
        hook = GuardrailHook(input_guardrail=mock_guardrail)
        ctx = HookContext(event=HookEvent.PRE_INPUT, messages=[])
        result = await hook.execute(ctx)
        assert result is ctx

    async def test_input_check_passthrough_when_no_user_message(self):
        """Should passthrough when no user message found."""
        mock_guardrail = MockInputGuardrail()
        hook = GuardrailHook(input_guardrail=mock_guardrail)
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "assistant", "content": "Hello"},
        ]
        ctx = HookContext(event=HookEvent.PRE_INPUT, messages=messages)
        result = await hook.execute(ctx)
        assert result is ctx

    async def test_input_check_finds_user_message(self):
        """Should find and check the last user message."""
        mock_guardrail = MockInputGuardrail(passed=True, reason="OK")
        hook = GuardrailHook(input_guardrail=mock_guardrail)
        messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Last message"},
        ]
        ctx = HookContext(event=HookEvent.PRE_INPUT, messages=messages)
        result = await hook.execute(ctx)
        assert result.metadata["input_guardrail_result"]["passed"] is True

    async def test_input_check_passed_sets_metadata(self):
        """Should set metadata when input check passes."""
        mock_guardrail = MockInputGuardrail(passed=True, reason="OK", action="allow")
        hook = GuardrailHook(input_guardrail=mock_guardrail)
        messages = [{"role": "user", "content": "Hello"}]
        ctx = HookContext(event=HookEvent.PRE_INPUT, messages=messages)
        result = await hook.execute(ctx)
        assert result.metadata["input_guardrail_result"]["passed"] is True
        assert result.metadata["input_guardrail_result"]["reason"] == "OK"

    async def test_input_check_failed_sets_abort(self):
        """Should set abort flag when input check fails."""
        mock_guardrail = MockInputGuardrail(passed=False, reason="Blocked content", action="block")
        hook = GuardrailHook(input_guardrail=mock_guardrail)
        messages = [{"role": "user", "content": "Bad content"}]
        ctx = HookContext(event=HookEvent.PRE_INPUT, messages=messages)
        result = await hook.execute(ctx)
        assert result.abort is True
        assert result.metadata["guardrail_blocked"] is True

    async def test_input_check_failed_injects_message(self):
        """Should inject message when input check fails."""
        mock_guardrail = MockInputGuardrail(passed=False, reason="Profanity detected", action="block")
        hook = GuardrailHook(input_guardrail=mock_guardrail)
        messages = [{"role": "user", "content": "Bad words"}]
        ctx = HookContext(event=HookEvent.PRE_INPUT, messages=messages)
        result = await hook.execute(ctx)
        assert "Input blocked" in result.inject_message
        assert "Profanity detected" in result.inject_message

    async def test_input_check_metadata_includes_action(self):
        """Metadata should include action from guardrail result."""
        mock_guardrail = MockInputGuardrail(passed=False, reason="Blocked", action="block")
        hook = GuardrailHook(input_guardrail=mock_guardrail)
        messages = [{"role": "user", "content": "Test"}]
        ctx = HookContext(event=HookEvent.PRE_INPUT, messages=messages)
        result = await hook.execute(ctx)
        assert result.metadata["input_guardrail_result"]["action"] == "block"


class TestGuardrailHookOutputCheck:
    async def test_output_check_passthrough_when_no_llm_response(self):
        """Should passthrough when no llm_response available."""
        mock_guardrail = MockOutputGuardrail()
        hook = GuardrailHook(output_guardrail=mock_guardrail)
        ctx = HookContext(event=HookEvent.POST_OUTPUT, llm_response=None)
        result = await hook.execute(ctx)
        assert result is ctx

    async def test_output_check_passthrough_when_empty_content(self):
        """Should passthrough when llm_response has no content."""
        mock_guardrail = MockOutputGuardrail()
        hook = GuardrailHook(output_guardrail=mock_guardrail)

        class EmptyResponse:
            content = ""

        ctx = HookContext(event=HookEvent.POST_OUTPUT, llm_response=EmptyResponse())
        result = await hook.execute(ctx)
        assert result is ctx

    async def test_output_check_extracts_content_from_response(self):
        """Should extract content from llm_response for checking."""
        mock_guardrail = MockOutputGuardrail(passed=True, reason="OK")
        hook = GuardrailHook(output_guardrail=mock_guardrail)

        class MockResponse:
            content = "This is the output"

        ctx = HookContext(event=HookEvent.POST_OUTPUT, llm_response=MockResponse())
        result = await hook.execute(ctx)
        assert result.metadata["output_guardrail_result"]["passed"] is True

    async def test_output_check_passed_sets_metadata(self):
        """Should set metadata when output check passes."""
        mock_guardrail = MockOutputGuardrail(passed=True, reason="OK", action="allow")
        hook = GuardrailHook(output_guardrail=mock_guardrail)

        class MockResponse:
            content = "Safe output"

        ctx = HookContext(event=HookEvent.POST_OUTPUT, llm_response=MockResponse())
        result = await hook.execute(ctx)
        assert result.metadata["output_guardrail_result"]["passed"] is True
        assert result.metadata["output_guardrail_result"]["reason"] == "OK"

    async def test_output_check_failed_does_not_abort(self):
        """Should NOT set abort flag when output check fails (only warns)."""
        mock_guardrail = MockOutputGuardrail(passed=False, reason="Unsafe content", action="warn")
        hook = GuardrailHook(output_guardrail=mock_guardrail)

        class MockResponse:
            content = "Unsafe output"

        ctx = HookContext(event=HookEvent.POST_OUTPUT, llm_response=MockResponse())
        result = await hook.execute(ctx)
        assert result.abort is False

    async def test_output_check_failed_sets_warning_metadata(self):
        """Should set output_warning in metadata when output check fails."""
        mock_guardrail = MockOutputGuardrail(passed=False, reason="PII detected", action="warn")
        hook = GuardrailHook(output_guardrail=mock_guardrail)

        class MockResponse:
            content = "SSN: 123-45-6789"

        ctx = HookContext(event=HookEvent.POST_OUTPUT, llm_response=MockResponse())
        result = await hook.execute(ctx)
        assert result.metadata["output_warning"] == "PII detected"

    async def test_output_check_metadata_includes_action(self):
        """Metadata should include action from guardrail result."""
        mock_guardrail = MockOutputGuardrail(passed=False, reason="Flagged", action="warn")
        hook = GuardrailHook(output_guardrail=mock_guardrail)

        class MockResponse:
            content = "Output"

        ctx = HookContext(event=HookEvent.POST_OUTPUT, llm_response=MockResponse())
        result = await hook.execute(ctx)
        assert result.metadata["output_guardrail_result"]["action"] == "warn"


class TestGuardrailHookIntegration:
    async def test_both_guardrails_can_coexist(self):
        """Both input and output guardrails should work together."""
        mock_input = MockInputGuardrail(passed=True, reason="OK")
        mock_output = MockOutputGuardrail(passed=True, reason="OK")
        hook = GuardrailHook(input_guardrail=mock_input, output_guardrail=mock_output)

        # Test input path
        messages = [{"role": "user", "content": "Hello"}]
        ctx_input = HookContext(event=HookEvent.PRE_INPUT, messages=messages)
        result_input = await hook.execute(ctx_input)
        assert result_input.metadata["input_guardrail_result"]["passed"] is True

        # Test output path
        class MockResponse:
            content = "Output"

        ctx_output = HookContext(event=HookEvent.POST_OUTPUT, llm_response=MockResponse())
        result_output = await hook.execute(ctx_output)
        assert result_output.metadata["output_guardrail_result"]["passed"] is True

    async def test_different_events_call_correct_check(self):
        """PRE_INPUT should call input check, POST_OUTPUT should call output check."""
        mock_input = MockInputGuardrail(passed=True, reason="Input OK")
        mock_output = MockOutputGuardrail(passed=True, reason="Output OK")
        hook = GuardrailHook(input_guardrail=mock_input, output_guardrail=mock_output)

        # PRE_INPUT event
        messages = [{"role": "user", "content": "Test"}]
        ctx1 = HookContext(event=HookEvent.PRE_INPUT, messages=messages)
        result1 = await hook.execute(ctx1)
        assert "input_guardrail_result" in result1.metadata

        # Test POST_OUTPUT event
        class MockResponse:
            content = "Output"

        ctx2 = HookContext(event=HookEvent.POST_OUTPUT, llm_response=MockResponse())
        result2 = await hook.execute(ctx2)
        assert "output_guardrail_result" in result2.metadata
