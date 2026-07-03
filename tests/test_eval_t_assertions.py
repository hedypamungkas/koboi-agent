"""Tests for koboi.eval.t.assertions -- Severity, matchers, recorded assertions."""

from __future__ import annotations

from koboi.eval.t.assertions import (
    AssertionOutcome,
    Contains,
    Equals,
    Matches,
    RecordedAssertion,
    Regex,
    Severity,
    Truth,
    binary_outcome,
    coerce_matcher,
    describe_value,
)


class TestSeverity:
    def test_values(self):
        assert Severity.GATE.value == "gate"
        assert Severity.SOFT.value == "soft"

    def test_string_equality(self):
        assert Severity.GATE == "gate"


class TestMatchers:
    def test_equals(self):
        assert Equals(4).matches(4) is True
        assert Equals(4).matches("4") is False
        assert "equals 4" in Equals(4).describe()

    def test_contains_str_case_insensitive_default(self):
        assert Contains("answer").matches("The Answer is 4") is True
        assert Contains("Answer", case_insensitive=False).matches("the answer") is False
        assert Contains("missing").matches("hello") is False

    def test_contains_none_safe(self):
        assert Contains("x").matches(None) is False

    def test_contains_list_membership(self):
        assert Contains("a").matches(["a", "b"]) is True
        assert Contains("z").matches(["a", "b"]) is False

    def test_regex(self):
        assert Regex(r"\d+").matches("answer 42") is True
        assert Regex(r"^\d+$").matches("abc") is False

    def test_truth(self):
        assert Truth().matches(1) is True
        assert Truth().matches(0) is False
        assert Truth().matches("") is False

    def test_matches_callable_and_exception(self):
        assert Matches(lambda x: x > 5).matches(10) is True
        assert Matches(lambda x: x > 5).matches(1) is False
        # An exception inside the predicate must not propagate -- it is a failed match.
        assert Matches(lambda x: 1 / 0).matches(1) is False


class TestCoerceMatcher:
    def test_matcher_passthrough(self):
        matcher = Equals(1)
        assert coerce_matcher(matcher) is matcher

    def test_callable_becomes_matches(self):
        matcher = coerce_matcher(lambda x: x is True)
        assert isinstance(matcher, Matches)
        assert matcher.matches(True) is True

    def test_bare_value_becomes_equals(self):
        matcher = coerce_matcher(42)
        assert isinstance(matcher, Equals)
        assert matcher.matches(42) is True


def test_describe_value_truncates():
    assert describe_value("ok") == "'ok'"
    truncated = describe_value("x" * 200, limit=20)
    assert len(truncated) <= 20
    assert truncated.endswith("…")


class TestBinaryOutcome:
    def test_pass(self):
        outcome = binary_outcome(Severity.GATE, True, "ok")
        assert outcome.passed is True and outcome.value == 1.0

    def test_gate_failure(self):
        outcome = binary_outcome(Severity.GATE, False, "nope")
        assert outcome.passed is False and outcome.value == 0.0

    def test_soft_failure(self):
        outcome = binary_outcome(Severity.SOFT, False, "nope")
        assert outcome.passed is False and outcome.value == 0.5


class TestRecordedAssertion:
    def test_outcome_evaluates_lazily(self):
        calls = {"n": 0}

        def evaluate():
            calls["n"] += 1
            return AssertionOutcome(True, 1.0, "ok")

        assertion = RecordedAssertion(name="x", severity=Severity.GATE, evaluate=evaluate)
        assert calls["n"] == 0  # not evaluated at construction
        outcome = assertion.outcome()
        assert outcome.passed is True
        assert calls["n"] == 1  # evaluated exactly once
