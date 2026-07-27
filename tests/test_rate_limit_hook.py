"""Tests for koboi/hooks/rate_limit_hook.py -- RateLimitEmitHook.

Headers are sourced from ``AgentResponse.response_headers`` (what the LLM adapters
populate), NOT from ``ctx.metadata`` -- the production loop fires POST_LLM_CALL with
``llm_response=`` only. Tests below build the context the way the loop does.
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

from koboi.hooks.chain import HookContext, HookEvent
from koboi.hooks.rate_limit_hook import RateLimitEmitHook
from koboi.types import AgentResponse


def _ctx(*, headers: dict | None = None, model: str | None = None, metadata: dict | None = None) -> HookContext:
    """Build a POST_LLM_CALL context shaped like the loop's ``_emit``."""
    return HookContext(
        event=HookEvent.POST_LLM_CALL,
        llm_response=AgentResponse(content="test", model=model, response_headers=headers or {}),
        metadata=metadata,
    )


class TestRateLimitEmitHook:
    def test_handles_returns_post_llm_call(self):
        """RateLimitEmitHook handles POST_LLM_CALL event."""
        assert RateLimitEmitHook().handles() == [HookEvent.POST_LLM_CALL]

    async def test_passthrough_when_no_response(self):
        """Should return context unchanged when there is no llm_response."""
        hook = RateLimitEmitHook()
        ctx = HookContext(event=HookEvent.POST_LLM_CALL)  # no llm_response, like a stray emit
        result = await hook.execute(ctx)
        assert result is ctx

    async def test_emits_telemetry_when_anthropic_headers_present(self):
        """Should emit telemetry when Anthropic rate-limit headers are present."""
        emit_mock = MagicMock()
        hook = RateLimitEmitHook(emit_func=emit_mock)

        await hook.execute(
            _ctx(
                model="claude-3-opus",
                headers={
                    "retry-after": "60",
                    "anthropic-ratelimit-unified-requests-remaining": "100",
                    "anthropic-ratelimit-unified-tokens-remaining": "50000",
                    "anthropic-ratelimit-unified-requests-reset": "2025-01-01T00:00:00Z",
                },
            )
        )

        emit_mock.assert_called_once()
        event_type, data = emit_mock.call_args[0]
        assert event_type == "rate_limit.info"
        assert data["provider"] == "anthropic"
        assert data["retry_after"] == 60.0
        assert data["remaining_requests"] == 100
        assert data["remaining_tokens"] == 50000
        assert data["reset_at"] == "2025-01-01T00:00:00Z"

    async def test_emits_telemetry_when_openai_headers_present(self):
        """Should emit telemetry when OpenAI rate-limit headers are present."""
        emit_mock = MagicMock()
        hook = RateLimitEmitHook(emit_func=emit_mock)

        await hook.execute(
            _ctx(
                model="gpt-4",
                headers={
                    "retry-after": "30",
                    "x-ratelimit-remaining-requests": "50",
                    "x-ratelimit-remaining-tokens": "25000",
                    "x-ratelimit-reset": "1735689600",
                },
            )
        )

        emit_mock.assert_called_once()
        event_type, data = emit_mock.call_args[0]
        assert event_type == "rate_limit.info"
        assert data["provider"] == "openai"
        assert data["retry_after"] == 30.0
        assert data["remaining_requests"] == 50
        assert data["remaining_tokens"] == 25000

    async def test_skips_emission_when_no_rate_limit_headers(self):
        """Should not emit when only non-rate-limit headers are present."""
        emit_mock = MagicMock()
        hook = RateLimitEmitHook(emit_func=emit_mock)

        await hook.execute(_ctx(headers={"content-type": "application/json"}))

        emit_mock.assert_not_called()

    async def test_parses_retry_after_as_http_date(self):
        """Should parse retry-after as HTTP-date string when not numeric."""
        emit_mock = MagicMock()
        hook = RateLimitEmitHook(emit_func=emit_mock)

        await hook.execute(
            _ctx(
                headers={
                    "retry-after": "Wed, 21 Oct 2015 07:28:00 GMT",
                    "x-ratelimit-remaining-requests": "10",
                }
            )
        )

        emit_mock.assert_called_once()
        data = emit_mock.call_args[0][1]
        assert data["retry_after"] == "Wed, 21 Oct 2015 07:28:00 GMT"
        assert data["remaining_requests"] == 10

    async def test_handles_mixed_provider_headers(self):
        """Should handle mixed Anthropic/OpenAI headers (priority to Anthropic)."""
        emit_mock = MagicMock()
        hook = RateLimitEmitHook(emit_func=emit_mock)

        await hook.execute(
            _ctx(
                model="claude-3-opus",
                headers={
                    "anthropic-ratelimit-unified-requests-remaining": "80",
                    "x-ratelimit-remaining-requests": "90",
                },
            )
        )

        emit_mock.assert_called_once()
        data = emit_mock.call_args[0][1]
        # Anthropic headers take priority when both present
        assert data["remaining_requests"] == 80

    async def test_default_emit_logs_info(self):
        """Default emit function should log at INFO level."""
        hook = RateLimitEmitHook()

        logger_mock = Mock()
        with patch("koboi.hooks.rate_limit_hook._logger", logger_mock):
            await hook.execute(_ctx(headers={"retry-after": "120", "x-ratelimit-remaining-requests": "5"}))

            logger_mock.info.assert_called_once()
            assert "[rate_limit]" in logger_mock.info.call_args[0][0]

    async def test_returns_context_unchanged_after_emission(self):
        """Should return the same context object after emitting telemetry."""
        hook = RateLimitEmitHook()
        ctx = _ctx(headers={"retry-after": "60"})
        result = await hook.execute(ctx)
        assert result is ctx

    async def test_does_not_read_metadata_response_headers(self):
        """Regression guard: the hook must read headers from the AgentResponse, not
        ctx.metadata. A context carrying rate-limit headers ONLY in metadata (the
        old, never-populated-by-the-loop key) must NOT emit.
        """
        emit_mock = MagicMock()
        hook = RateLimitEmitHook(emit_func=emit_mock)

        # Loop never writes ctx.metadata["response_headers"]; if it did, the hook
        # must still ignore it and read ctx.llm_response.response_headers instead.
        ctx = HookContext(
            event=HookEvent.POST_LLM_CALL,
            llm_response=AgentResponse(content="x", model="gpt-4", response_headers={}),
            metadata={"response_headers": {"x-ratelimit-remaining-requests": "99"}},
        )
        await hook.execute(ctx)

        emit_mock.assert_not_called()
