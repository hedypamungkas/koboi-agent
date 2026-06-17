"""koboi/guardrails/base.py -- Base guardrail ABC and pattern-based guardrail."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from koboi.types import GuardrailResult


class BaseGuardrail(ABC):
    """Base class for content guardrails.

    Subclasses implement check() to inspect text content (user input
    or agent output) and return a GuardrailResult.
    """

    @abstractmethod
    async def check(self, content: str) -> GuardrailResult: ...


class PatternGuardrail(BaseGuardrail):
    """Base class for guardrails that match content against regex patterns.

    Subclasses override PATTERNS and DEFAULT_ACTION. The check_patterns()
    method iterates all patterns and returns GuardrailResult on first match.
    """

    PATTERNS: list[tuple[str, str]] = []
    DEFAULT_ACTION: str = "block"

    def __init__(
        self,
        patterns: list[tuple[str, str]] | None = None,
        default_action: str | None = None,
        custom_patterns: list[tuple[str, str]] | None = None,
        logger: object | None = None,
        **kwargs: object,
    ):
        self.patterns = list(patterns or self.PATTERNS)
        if custom_patterns:
            self.patterns.extend(custom_patterns)
        self.default_action = default_action or self.DEFAULT_ACTION
        self.logger = logger

    async def check_patterns(self, content: str) -> GuardrailResult | None:
        """Return GuardrailResult(passed=False) on first pattern match, or None."""
        for pattern, description in self.patterns:
            if re.search(pattern, content):
                return GuardrailResult(
                    passed=False,
                    reason=description,
                    action=self.default_action,
                )
        return None
