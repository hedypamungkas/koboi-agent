"""koboi/hooks/rate_limit_hook.py -- Hook that emits rate-limit telemetry from LLM response headers.

Captures provider rate-limit information from LLM response headers and emits it as a
structured event. This enables external monitoring systems to track rate limit status
without requiring direct access to the agent's internal logs.

Opt-in by default (must be enabled via config) to avoid changing existing behavior.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from koboi.hooks.chain import Hook, HookContext, HookEvent

if TYPE_CHECKING:
    pass

_logger = logging.getLogger(__name__)


class RateLimitEmitHook(Hook):
    """Hook that captures and emits rate-limit telemetry from LLM response headers.

    Subscribes to POST_LLM_CALL events and extracts rate-limit information from
    response headers (when available via metadata), then logs it for external
    consumption by monitoring/webhook systems.

    Default: disabled (opt-in via ``hooks.rate_limit_emit: true``).
    """

    priority = 20  # Run early (infrastructure/telemetry tier)

    def __init__(self, emit_func: callable | None = None):
        """Initialize the hook.

        Args:
            emit_func: Optional custom emit function. If None, uses _logger.info.
                      Signature: emit_func(event_type: str, data: dict) -> None
        """
        self._emit = emit_func or self._default_emit

    def handles(self) -> list[HookEvent]:
        return [HookEvent.POST_LLM_CALL]

    async def execute(self, ctx: HookContext) -> HookContext:
        """Extract rate-limit headers from response metadata and emit telemetry.

        Headers are expected to be attached to ctx.metadata by the LLM adapter
        (e.g., in ``koboi/llm/openai_adapter.py``) under the key ``response_headers``.

        Emits a ``rate_limit.info`` event with fields:
            - provider: str (e.g., "openai", "anthropic")
            - retry_after: float | None (seconds until retry allowed)
            - remaining_requests: int | None
            - remaining_tokens: int | None
            - reset_at: str | None (ISO timestamp or human-readable)
        """
        # Access headers from metadata (set by LLM adapter)
        headers = ctx.metadata.get("response_headers") if ctx.metadata else None
        if not headers or not isinstance(headers, dict):
            # No headers available (not an error, just skip)
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

        reset_at = headers.get("anthropic-ratelimit-unified-requests-reset") or headers.get(
            "x-ratelimit-reset"
        )

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

    def _parse_retry_after(self, headers: dict) -> float | None:
        """Parse retry-after header (can be seconds int or HTTP-date)."""
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
