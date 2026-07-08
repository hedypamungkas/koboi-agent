"""koboi/eval/scorers/gaia_scorer.py -- GAIA-style exact-match scorer.

Verifies answers using normalized exact matching with support for
numeric tolerance, unit stripping, and custom verification functions.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from koboi.types import EvalCase, EvalScore
from koboi.eval.scorers.base import BaseScorer


class GAIAVerificationScorer(BaseScorer):
    """GAIA-style exact-match scorer with answer normalization.

    Normalizes answers by:
    - Stripping whitespace
    - Case-insensitive comparison
    - Removing units (%, $, km, etc.)
    - Numeric tolerance for floating point
    - Accepting equivalent formats (1,000 == 1000)
    """

    UNITS_PATTERN = re.compile(
        r"\s*(%|percent|dollars?|USD|EUR|km|miles?|meters?|kg|pounds?|lbs?|hours?|hrs?|minutes?|mins?|seconds?|secs?|days?|years?|months?|weeks?)$",
        re.IGNORECASE,
    )

    def __init__(self, numeric_tolerance: float = 0.01):
        self.numeric_tolerance = numeric_tolerance

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        if not case.expected_answer:
            return EvalScore("gaia_verification", 0.0, "No expected_answer")

        # If case has custom verification_fn, use it
        if case.verification_fn:
            return self._run_custom_verification(
                case.verification_fn,
                output,
                case.expected_answer,
            )

        # Default: normalized exact match
        return self._normalized_match(output, case.expected_answer)

    def _run_custom_verification(
        self,
        fn: Callable,
        output: str,
        expected: str,
    ) -> EvalScore:
        """Run custom verification function."""
        try:
            passed = fn(output, expected)
            return EvalScore(
                "gaia_verification",
                1.0 if passed else 0.0,
                f"Custom verification: {'pass' if passed else 'fail'}",
            )
        except Exception as e:
            return EvalScore("gaia_verification", 0.0, f"Verification error: {e}")

    def _normalized_match(self, actual: str, expected: str) -> EvalScore:
        """Normalize and compare answers."""
        norm_actual = self._normalize_answer(actual)
        norm_expected = self._normalize_answer(expected)

        # Exact string match after normalization
        if norm_actual == norm_expected:
            return EvalScore("gaia_verification", 1.0, "Exact match (normalized)")

        # Try numeric comparison
        numeric_result = self._try_numeric_compare(norm_actual, norm_expected)
        if numeric_result is True:
            return EvalScore("gaia_verification", 1.0, "Numeric match (within tolerance)")
        if numeric_result is False:
            return EvalScore("gaia_verification", 0.0, "Numeric mismatch")

        # Try substring containment
        if norm_expected in norm_actual or norm_actual in norm_expected:
            return EvalScore("gaia_verification", 0.8, "Partial match (substring)")

        return EvalScore(
            "gaia_verification",
            0.0,
            f"Mismatch: expected='{norm_expected[:100]}', got='{norm_actual[:100]}'",
        )

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        """Strip, lowercase, remove common units and formatting."""
        s = answer.strip().lower()

        # Remove trailing units
        s = GAIAVerificationScorer.UNITS_PATTERN.sub("", s)

        # Remove commas in numbers: 1,000 -> 1000
        s = re.sub(r"(\d),(\d)", r"\1\2", s)

        # Remove currency symbols
        s = re.sub(r"[$€£]", "", s)

        # Normalize whitespace
        s = re.sub(r"\s+", " ", s).strip()

        return s

    def _try_numeric_compare(self, actual: str, expected: str) -> bool | None:
        """Try numeric comparison with tolerance.

        Returns:
            True if numeric match within tolerance.
            False if both are numeric but don't match.
            None if either is not numeric.
        """
        try:
            a = float(actual.replace(",", ""))
            e = float(expected.replace(",", ""))
            if abs(a - e) <= self.numeric_tolerance * max(abs(e), 1.0):
                return True
            return False
        except (ValueError, TypeError):
            return None
