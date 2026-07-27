"""Tests for koboi/hooks/rate_limit_hook.py — RateLimitEmitHook (contribution #5)."""

from __future__ import annotations

from unittest.mock import MagicMock

from koboi.hooks.chain import HookContext, HookEvent
from koboi.hooks.rate_limit_hook import RateLimitEmitHook
from koboi.types import AgentResponse


class TestRateLimitEmitHook:
    def test_handles_returns_post_llm_call(self):
        """RateLimitEmitHook handles POST_LLM_CALL event."""
        hook = RateLimitEmitHook()
        assert hook.handles() == [HookEvent.POST_LLM_CALL]

    async def test_passthrough_when_no_headers(self):
        """Should return context unchanged when no headers available."""
        hook = RateLimitEmitHook()
        ctx = HookContext(event=HookEvent.POST_LLM_CALL)
        result = await hook.execute(ctx)
        assert result is ctx

    async def test_emits_telemetry_when_anthropic_headers_present(self):
        """Should emit telemetry when Anthropic rate-limit headers are present."""
        emit_mock = MagicMock()
        hook = RateLimitEmitHook(emit_func=emit_mock)

        ctx = HookContext(
            event=HookEvent.POST_LLM_CALL,
            llm_response=AgentResponse(content="test", model="claude-3-opus"),
            metadata={
                "response_headers": {
                    "retry-after": "60",
                    "anthropic-ratelimit-unified-requests-remaining": "100",
                    "anthropic-ratelimit-unified-tokens-remaining": "50000",
                    "anthropic-ratelimit-unified-requests-reset": "2025-01-01T00:00:00Z",
                }
            },
        )

        await hook.execute(ctx)

        emit_mock.assert_called_once()
        call_args = emit_mock.call_args
        assert call_args[0][0] == "rate_limit.info"
        data = call_args[0][1]
        assert data["provider"] == "anthropic"
        assert data["retry_after"] == 60.0
        assert data["remaining_requests"] == 100
        assert data["remaining_tokens"] == 50000
        assert data["reset_at"] == "2025-01-01T00:00:00Z"

    async def test_emits_telemetry_when_openai_headers_present(self):
        """Should emit telemetry when OpenAI rate-limit headers are present."""
        emit_mock = MagicMock()
        hook = RateLimitEmitHook(emit_func=emit_mock)

        ctx = HookContext(
            event=HookEvent.POST_LLM_CALL,
            llm_response=AgentResponse(content="test", model="gpt-4"),
            metadata={
                "response_headers": {
                    "retry-after": "30",
                    "x-ratelimit-remaining-requests": "50",
                    "x-ratelimit-remaining-tokens": "25000",
                    "x-ratelimit-reset": "1735689600",
                }
            },
        )

        await hook.execute(ctx)

        emit_mock.assert_called_once()
        call_args = emit_mock.call_args
        assert call_args[0][0] == "rate_limit.info"
        data = call_args[0][1]
        assert data["provider"] == "openai"
        assert data["retry_after"] == 30.0
        assert data["remaining_requests"] == 50
        assert data["remaining_tokens"] == 25000

    async def test_skips_emission_when_no_rate_limit_headers(self):
        """Should not emit when only non-rate-limit headers present."""
        emit_mock = MagicMock()
        hook = RateLimitEmitHook(emit_func=emit_mock)

        ctx = HookContext(
            event=HookEvent.POST_LLM_CALL,
            metadata={"response_headers": {"content-type": "application/json"}},
        )

        await hook.execute(ctx)

        emit_mock.assert_not_called()

    async def test_parses_retry_after_as_http_date(self):
        """Should parse retry-after as HTTP-date string when not numeric."""
        emit_mock = MagicMock()
        hook = RateLimitEmitHook(emit_func=emit_mock)

        ctx = HookContext(
            event=HookEvent.POST_LLM_CALL,
            metadata={
                "response_headers": {
                    "retry-after": "Wed, 21 Oct 2015 07:28:00 GMT",
                    "x-ratelimit-remaining-requests": "10",
                }
            },
        )

        await hook.execute(ctx)

        emit_mock.assert_called_once()
        data = emit_mock.call_args[0][1]
        assert data["retry_after"] == "Wed, 21 Oct 2015 07:28:00 GMT"
        assert data["remaining_requests"] == 10

    async def test_handles_mixed_provider_headers(self):
        """Should handle mixed Anthropic/OpenAI headers (priority to Anthropic)."""
        emit_mock = MagicMock()
        hook = RateLimitEmitHook(emit_func=emit_mock)

        ctx = HookContext(
            event=HookEvent.POST_LLM_CALL,
            llm_response=AgentResponse(content="test", model="claude-3-opus"),
            metadata={
                "response_headers": {
                    "anthropic-ratelimit-unified-requests-remaining": "80",
                    "x-ratelimit-remaining-requests": "90",
                }
            },
        )

        await hook.execute(ctx)

        emit_mock.assert_called_once()
        data = emit_mock.call_args[0][1]
        # Anthropic headers take priority when both present
        assert data["remaining_requests"] == 80

    async def test_default_emit_logs_info(self):
        """Default emit function should log at INFO level."""
        from unittest.mock import Mock, patch

        hook = RateLimitEmitHook()

        # Mock the logger
        logger_mock = Mock()
        with patch("koboi.hooks.rate_limit_hook._logger", logger_mock):
            ctx = HookContext(
                event=HookEvent.POST_LLM_CALL,
                metadata={"response_headers": {"retry-after": "120", "x-ratelimit-remaining-requests": "5"}},
            )

            await hook.execute(ctx)

            logger_mock.info.assert_called_once()
            call_args = logger_mock.info.call_args
            assert "[rate_limit]" in call_args[0][0]

    async def test_returns_context_unchanged_after_emission(self):
        """Should return the same context object after emitting telemetry."""
        hook = RateLimitEmitHook()
        ctx = HookContext(
            event=HookEvent.POST_LLM_CALL,
            metadata={"response_headers": {"retry-after": "60"}},
        )
        result = await hook.execute(ctx)
        assert result is ctx
