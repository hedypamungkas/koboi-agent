"""koboi/hooks/rate_limit_hook.py -- Hook that emits rate-limit telemetry from LLM response headers.

Captures provider rate-limit information from LLM response headers and emits it as a
structured event. This enables external monitoring systems to track rate limit status
without requiring direct access to the agent's internal logs.

Opt-in by default (must be enabled via config) to avoid changing existing behavior.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from koboi.hooks.chain import Hook, HookContext, HookEvent

_logger = logging.getLogger(__name__)


class RateLimitEmitHook(Hook):
    """Hook that captures and emits rate-limit telemetry from LLM response headers.

    Subscribes to POST_LLM_CALL events and extracts rate-limit information from
    response headers (carried on the ``AgentResponse.response_headers`` field), then
    logs it for external consumption by monitoring/webhook systems.

    Default: disabled (opt-in via ``hooks.rate_limit_emit: true``).
    """

    priority = 20  # Run early (infrastructure/telemetry tier)

    def __init__(self, emit_func: Callable[[str, dict], None] | None = None):
        """Initialize the hook.

        Args:
            emit_func: Optional custom emit function. If None, uses _logger.info.
                      Signature: emit_func(event_type: str, data: dict) -> None
        """
        self._emit = emit_func or self._default_emit

    def handles(self) -> list[HookEvent]:
        return [HookEvent.POST_LLM_CALL]

    async def execute(self, ctx: HookContext) -> HookContext:
        """Extract rate-limit headers from the LLM response and emit telemetry.

        Headers are read from ``ctx.llm_response.response_headers`` (populated by the
        LLM adapters from the transport's most-recent response headers).

        Emits a ``rate_limit.info`` event with fields:
            - provider: str (e.g., "openai", "anthropic")
            - retry_after: float | str | None (seconds, or an HTTP-date string)
            - remaining_requests: int | None
            - remaining_tokens: int | None
            - reset_at: str | None (ISO timestamp or human-readable)
        """
        # Rate-limit headers are attached to the AgentResponse by the LLM adapter
        # (``response_headers`` field, populated from the transport's
        # ``last_response_headers``). The loop fires POST_LLM_CALL with the response
        # object; it does NOT copy headers into ctx.metadata, so read them here.
        headers = None
        if ctx.llm_response is not None:
            headers = getattr(ctx.llm_response, "response_headers", None)
        if not isinstance(headers, dict) or not headers:
            _logger.debug("rate-limit hook: response carried no headers; skipping emit")
            return ctx

        # Extract provider from response if available
        provider = "unknown"
        if ctx.llm_response and hasattr(ctx.llm_response, "model"):
            model = ctx.llm_response.model or ""
            # Simple heuristic: model name prefixes indicate provider
            if model.startswith("gpt-"):
                provider = "openai"
            elif model.startswith("claude-"):
                provider = "anthropic"

        # Parse headers (Anthropic and OpenAI use different formats)
        retry_after = self._parse_retry_after(headers)
        remaining_requests = self._parse_header_int(
            headers, "anthropic-ratelimit-unified-requests-remaining"
        ) or self._parse_header_int(headers, "x-ratelimit-remaining-requests")

        remaining_tokens = self._parse_header_int(
            headers, "anthropic-ratelimit-unified-tokens-remaining"
        ) or self._parse_header_int(headers, "x-ratelimit-remaining-tokens")

        reset_at = headers.get("anthropic-ratelimit-unified-requests-reset") or headers.get("x-ratelimit-reset")

        # Only emit if we have at least one rate-limit metric
        if retry_after is not None or remaining_requests is not None or remaining_tokens is not None:
            event_data = {
                "provider": provider,
                "retry_after": retry_after,
                "remaining_requests": remaining_requests,
                "remaining_tokens": remaining_tokens,
                "reset_at": reset_at,
            }
            self._emit("rate_limit.info", event_data)

        return ctx

    def _parse_retry_after(self, headers: dict) -> float | str | None:
        """Parse retry-after header (seconds as a number, or an HTTP-date string).

        Note: providers only emit ``retry-after`` on a 429, which the HTTP transport
        raises as ``LLMRateLimitError`` *before* a POST_LLM_CALL fires -- so in
        practice this field is populated only when a provider/proxy emits it on a
        non-error response. The always-useful signal (remaining/reset) lives on 2xx.
        """
        val = headers.get("retry-after")
        if not val:
            return None

        try:
            # Try parsing as float/int (e.g., "60" -> 60.0)
            return float(val)
        except ValueError:
            # If not a number, it's an HTTP-date (e.g., "Wed, 21 Oct 2015 07:28:00 GMT")
            # For simplicity, we return the string as-is (logging systems can parse)
            return val

    def _parse_header_int(self, headers: dict, key: str) -> int | None:
        """Parse a header value as int, returning None if missing/invalid."""
        val = headers.get(key)
        if not val:
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    def _default_emit(self, event_type: str, data: dict) -> None:
        """Default emit implementation: log at INFO level.

        Structured format (space-delimited key=value) for log parsing.
        """
        # Filter out None values for cleaner logs
        filtered_data = {k: v for k, v in data.items() if v is not None}
        _logger.info("[rate_limit] %s %s", event_type, " ".join(f"{k}={v}" for k, v in filtered_data.items()))
