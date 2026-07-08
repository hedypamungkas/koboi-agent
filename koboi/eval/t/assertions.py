"""koboi/eval/t/assertions.py -- Severity, matchers, and the recorded-assertion model behind the `t` eval surface."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any
from collections.abc import Callable


class Severity(str, Enum):
    """How an assertion failure affects a test's outcome.

    GATE: any failure forces the test's ``EvalResult.passed = False`` (hard contract).
    SOFT: only contributes to ``overall_score``; cannot fail the test on its own.
    """

    GATE = "gate"
    SOFT = "soft"


@dataclass
class AssertionOutcome:
    """Result of evaluating one recorded assertion.

    ``value`` (0..1) feeds the test's ``overall_score``; ``passed`` combined with
    the assertion's severity decides gate-failure.
    """

    passed: bool
    value: float
    reason: str


class Matcher(ABC):
    """Composable value matcher used by ``t.check``."""

    @abstractmethod
    def matches(self, actual: Any) -> bool: ...

    @abstractmethod
    def describe(self) -> str: ...


@dataclass(frozen=True)
class Equals(Matcher):
    """Strict equality."""

    expected: Any

    def matches(self, actual: Any) -> bool:
        return actual == self.expected

    def describe(self) -> str:
        return f"equals {self.expected!r}"


@dataclass(frozen=True)
class Contains(Matcher):
    """Substring (str) or membership (container) match."""

    needle: Any
    case_insensitive: bool = True

    def matches(self, actual: Any) -> bool:
        if actual is None:
            return False
        if isinstance(actual, str) and isinstance(self.needle, str):
            if self.case_insensitive:
                return self.needle.lower() in actual.lower()
            return self.needle in actual
        try:
            return self.needle in actual
        except TypeError:
            return False

    def describe(self) -> str:
        return f"contains {self.needle!r}"


@dataclass(frozen=True)
class Regex(Matcher):
    """``re.search`` against the stringified value."""

    pattern: str
    flags: int = 0

    def matches(self, actual: Any) -> bool:
        return re.search(self.pattern, str(actual), self.flags) is not None

    def describe(self) -> str:
        return f"matches /{self.pattern}/"


@dataclass(frozen=True)
class Truth(Matcher):
    """Asserts the value is truthy (used when ``t.check(value)`` has no matcher)."""

    def matches(self, actual: Any) -> bool:
        return bool(actual)

    def describe(self) -> str:
        return "is truthy"


@dataclass(frozen=True)
class Matches(Matcher):
    """Arbitrary predicate ``fn(actual) -> bool``."""

    fn: Callable[[Any], bool]
    description: str = "matches predicate"

    def matches(self, actual: Any) -> bool:
        try:
            return bool(self.fn(actual))
        except Exception:
            return False

    def describe(self) -> str:
        return self.description


def coerce_matcher(matcher: Any) -> Matcher:
    """Turn a bare value/callable into a :class:`Matcher`; pass Matcher instances through."""
    if isinstance(matcher, Matcher):
        return matcher
    if callable(matcher):
        return Matches(fn=matcher)
    return Equals(expected=matcher)


def describe_value(value: Any, limit: int = 80) -> str:
    """Repr a value, truncated for readable assertion reasons."""
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class RecordedAssertion:
    """One assertion recorded by a ``t.*`` call, evaluated lazily at collect time."""

    name: str
    severity: Severity
    evaluate: Callable[[], AssertionOutcome]

    def outcome(self) -> AssertionOutcome:
        return self.evaluate()


def binary_outcome(severity: Severity, ok: bool, reason: str) -> AssertionOutcome:
    """Build an :class:`AssertionOutcome` for a pass/fail assertion.

    Pass -> value 1.0. Gate failure -> value 0.0. Soft failure -> value 0.5
    (advisory; does not gate).
    """
    if ok:
        return AssertionOutcome(passed=True, value=1.0, reason=reason)
    value = 0.0 if severity is Severity.GATE else 0.5
    return AssertionOutcome(passed=False, value=value, reason=reason)
