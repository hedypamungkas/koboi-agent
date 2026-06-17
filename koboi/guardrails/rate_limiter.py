"""Rate limiting for tool calls per session, per tool, and per minute."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import TYPE_CHECKING

from koboi.types import GuardrailResult, RateLimitConfig

if TYPE_CHECKING:
    from koboi.logger import AgentLogger


class RateLimiter:
    """Limit number of tool calls per session and per minute."""

    def __init__(
        self,
        config: RateLimitConfig | None = None,
        logger: AgentLogger | None = None,
    ):
        self.config = config or RateLimitConfig()
        self.logger = logger
        self._total_calls: int = 0
        self._per_tool_calls: dict[str, int] = defaultdict(int)
        self._call_timestamps: list[float] = []

    def check(self, tool_name: str) -> GuardrailResult:
        now = time.time()

        if self._total_calls >= self.config.max_tool_calls_per_session:
            return GuardrailResult(
                passed=False,
                reason=f"Rate limit: session max ({self.config.max_tool_calls_per_session}) reached",
                action="block",
            )

        if self.config.max_calls_per_tool and tool_name in self.config.max_calls_per_tool:
            limit = self.config.max_calls_per_tool[tool_name]
            if self._per_tool_calls[tool_name] >= limit:
                return GuardrailResult(
                    passed=False,
                    reason=f"Rate limit: tool '{tool_name}' max ({limit}) reached",
                    action="block",
                )

        cutoff = now - self.config.rate_window_seconds
        self._call_timestamps = [t for t in self._call_timestamps if t > cutoff]
        if len(self._call_timestamps) >= self.config.max_calls_per_minute:
            return GuardrailResult(
                passed=False,
                reason=f"Rate limit: {self.config.max_calls_per_minute} calls/min reached",
                action="block",
            )

        return GuardrailResult(passed=True)

    def record(self, tool_name: str) -> None:
        self._total_calls += 1
        self._per_tool_calls[tool_name] += 1
        self._call_timestamps.append(time.time())

    def reset(self) -> None:
        self._total_calls = 0
        self._per_tool_calls.clear()
        self._call_timestamps.clear()
