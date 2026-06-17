"""koboi/eval/scorers/bfcl_scorer.py -- BFCL-style tool-calling accuracy scorer.

Compares actual tool calls against expected tool calls using function name
matching and argument correctness (structural comparison).
"""
from __future__ import annotations

import json
from typing import Any

from koboi.types import EvalCase, EvalScore
from koboi.eval.scorers.base import BaseScorer


class ToolCallingScorer(BaseScorer):
    """BFCL-style tool-calling accuracy scorer.

    Compares actual tool calls against expected_tool_calls using:
    1. Function name matching
    2. Argument correctness (structural comparison)
    3. Handles multiple/parallel call ordering (order-independent by default)
    """

    def __init__(self, strict_args: bool = True, ignore_order: bool = True):
        self.strict_args = strict_args
        self.ignore_order = ignore_order

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        expected = case.expected_tool_calls
        if not expected:
            return EvalScore("tool_calling_accuracy", 1.0, "No expected tool calls")

        actual_calls: list[dict] = []

        # Try telemetry first
        telemetry = context.get("telemetry")
        if telemetry:
            actual_calls = self._extract_from_telemetry(telemetry)

        # Fallback: extract from tool_calls in context
        if not actual_calls:
            tool_calls = context.get("tool_calls", [])
            if tool_calls:
                actual_calls = [
                    {"name": tc.name, "arguments": self._parse_args(tc.arguments)}
                    for tc in tool_calls
                ]

        if not actual_calls:
            return EvalScore("tool_calling_accuracy", 0.0, "No telemetry or tool calls available")

        return self._compare_calls(expected, actual_calls)

    def _compare_calls(self, expected: list[dict], actual: list[dict]) -> EvalScore:
        """Compare expected vs actual tool calls."""
        if not actual and expected:
            return EvalScore(
                "tool_calling_accuracy", 0.0,
                f"Expected {len(expected)} tool calls, got 0",
            )

        # Name matching
        expected_names = [c.get("name", "") for c in expected]
        actual_names = [c.get("name", "") for c in actual]

        if self.ignore_order:
            name_matches = _count_matches_unordered(expected_names, actual_names)
        else:
            name_matches = _count_matches_ordered(expected_names, actual_names)

        name_score = name_matches / len(expected) if expected else 1.0

        # Argument matching (only for matched names)
        arg_scores: list[float] = []
        matched_pairs = _pair_calls(expected, actual, self.ignore_order)

        for exp_call, act_call in matched_pairs:
            exp_args = exp_call.get("arguments", {})
            act_args = act_call.get("arguments", {})
            arg_scores.append(self._compare_args(exp_args, act_args))

        avg_arg_score = sum(arg_scores) / len(arg_scores) if arg_scores else 0.0

        # Composite: 50% name match + 50% argument match
        composite = 0.5 * name_score + 0.5 * avg_arg_score

        details = (
            f"Names: {name_matches}/{len(expected)} matched, "
            f"Args: {avg_arg_score:.2f} avg accuracy"
        )
        return EvalScore("tool_calling_accuracy", round(composite, 3), details)

    def _compare_args(self, expected: dict, actual: dict) -> float:
        """Compare arguments between expected and actual tool calls."""
        if not expected and not actual:
            return 1.0
        if not expected or not actual:
            return 0.0

        if self.strict_args:
            return self._strict_compare(expected, actual)
        return self._fuzzy_compare(expected, actual)

    @staticmethod
    def _strict_compare(expected: dict, actual: dict) -> float:
        """Strict structural comparison of arguments."""
        all_keys = set(expected.keys()) | set(actual.keys())
        if not all_keys:
            return 1.0

        matches = 0
        for key in all_keys:
            exp_val = expected.get(key)
            act_val = actual.get(key)
            if exp_val is not None and act_val is not None:
                if _values_equal(exp_val, act_val):
                    matches += 1
            elif exp_val is None and act_val is None:
                matches += 1

        return matches / len(all_keys)

    @staticmethod
    def _fuzzy_compare(expected: dict, actual: dict) -> float:
        """Fuzzy comparison - checks if expected keys/values are present in actual."""
        if not expected:
            return 1.0

        matches = 0
        for key, exp_val in expected.items():
            act_val = actual.get(key)
            if act_val is not None and _values_equal(exp_val, act_val):
                matches += 1

        return matches / len(expected)

    @staticmethod
    def _parse_args(args: str | dict) -> dict:
        """Parse arguments from string or dict."""
        if isinstance(args, dict):
            return args
        if isinstance(args, str):
            try:
                return json.loads(args)
            except (json.JSONDecodeError, ValueError):
                return {}
        return {}

    @staticmethod
    def _extract_from_telemetry(telemetry) -> list[dict]:
        """Extract tool calls from telemetry snapshot."""
        calls: list[dict] = []
        snapshot = getattr(telemetry, 'snapshot', None)
        if not snapshot:
            return calls

        # TelemetryCollector tracks tool calls in iterations
        iterations = getattr(snapshot, 'iterations', [])
        for iteration in iterations:
            tools = getattr(iteration, 'tools', []) or []
            for tool in tools:
                calls.append({
                    "name": getattr(tool, 'name', ''),
                    "arguments": getattr(tool, 'arguments', {}),
                })

        return calls


def _values_equal(a: Any, b: Any) -> bool:
    """Compare two values with type coercion."""
    if a == b:
        return True
    # Try numeric comparison
    try:
        if float(a) == float(b):
            return True
    except (ValueError, TypeError):
        pass
    # Try string comparison
    return str(a).strip().lower() == str(b).strip().lower()


def _count_matches_unordered(expected: list[str], actual: list[str]) -> int:
    """Count matching names, order-independent."""
    actual_copy = list(actual)
    matches = 0
    for name in expected:
        if name in actual_copy:
            matches += 1
            actual_copy.remove(name)
    return matches


def _count_matches_ordered(expected: list[str], actual: list[str]) -> int:
    """Count matching names in order."""
    matches = 0
    for i, name in enumerate(expected):
        if i < len(actual) and actual[i] == name:
            matches += 1
    return matches


def _pair_calls(
    expected: list[dict],
    actual: list[dict],
    ignore_order: bool,
) -> list[tuple[dict, dict]]:
    """Pair expected calls with actual calls by name."""
    pairs: list[tuple[dict, dict]] = []
    actual_pool = list(actual)

    for exp in expected:
        exp_name = exp.get("name", "")
        match_idx = None

        for i, act in enumerate(actual_pool):
            if act.get("name", "") == exp_name:
                match_idx = i
                break

        if match_idx is not None:
            pairs.append((exp, actual_pool.pop(match_idx)))
        else:
            # No match found - pair with empty
            pairs.append((exp, {"name": "", "arguments": {}}))

    return pairs
