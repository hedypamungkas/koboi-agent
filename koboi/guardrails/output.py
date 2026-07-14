"""Output validation for agent responses before returning to user."""

from __future__ import annotations

from typing import TYPE_CHECKING

from koboi.guardrails.base import PatternGuardrail
from koboi.types import GuardrailResult

if TYPE_CHECKING:
    from koboi.logger import AgentLogger


class OutputGuardrail(PatternGuardrail):
    """Check agent output before returning to user."""

    PATTERNS: list[tuple[str, str]] = [
        (r"(?i)(api[_-]?key|password|secret|token)\s*[:=]\s*\S+", "Possible sensitive data leak"),
        (r"sk-[a-zA-Z0-9]{20,}", "Possible API key exposure"),
        (r"(?i)\b\d{16}\b", "Possible credit card number"),
    ]
    DEFAULT_ACTION = "warn"

    def __init__(
        self,
        custom_patterns: list[tuple[str, str]] | None = None,
        logger: AgentLogger | None = None,
        **kwargs: object,
    ):
        super().__init__(
            patterns=self.PATTERNS,
            default_action="warn",
            custom_patterns=custom_patterns,
            logger=logger,
        )

    async def check(self, agent_output: str, context: list[str] | None = None) -> GuardrailResult:
        if not agent_output:
            return GuardrailResult(passed=True)

        pattern_result = await self.check_patterns(agent_output)
        if pattern_result is not None:
            return pattern_result

        return GuardrailResult(passed=True)
