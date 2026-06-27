"""Tests for framework-specific eval scorers: BFCL, GAIA, SWE-bench, RAGAS, DeepEval."""

from __future__ import annotations

import pytest

from koboi.eval.scorers.bfcl_scorer import (
    ToolCallingScorer,
    _values_equal,
    _count_matches_unordered,
    _count_matches_ordered,
    _pair_calls,
)
from koboi.eval.scorers.gaia_scorer import GAIAVerificationScorer
from koboi.eval.scorers.swe_bench_scorer import (
    PatchGenerationScorer,
    _extract_filenames,
    _diff_stats,
)
from koboi.types import EvalCase


def _case(**kwargs):
    defaults = dict(name="t", user_message="q", max_iterations=10)
    defaults.update(kwargs)
    return EvalCase(**defaults)


# --- BFCL Scorer ---


class TestToolCallingScorer:
    @pytest.mark.asyncio
    async def test_no_expected_calls(self):
        s = ToolCallingScorer()
        score = await s.score(_case(), "out", {})
        assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_no_actual_calls(self):
        s = ToolCallingScorer()
        case = _case(expected_tool_calls=[{"name": "read", "arguments": {"path": "f.py"}}])
        score = await s.score(case, "out", {})
        assert score.value == 0.0

    @pytest.mark.asyncio
    async def test_perfect_match(self):
        from unittest.mock import MagicMock

        s = ToolCallingScorer()
        case = _case(expected_tool_calls=[{"name": "read", "arguments": {"path": "f.py"}}])
        telemetry = MagicMock()
        iteration = MagicMock()
        tool_mock = MagicMock()
        tool_mock.name = "read"
        tool_mock.arguments = {"path": "f.py"}
        iteration.tools = [tool_mock]
        telemetry.snapshot.iterations = [iteration]
        score = await s.score(case, "out", {"telemetry": telemetry})
        assert score.value > 0.5

    def test_values_equal_exact(self):
        assert _values_equal("a", "a") is True
        assert _values_equal(1, 1) is True

    def test_values_equal_numeric(self):
        assert _values_equal("1.0", 1.0) is True
        assert _values_equal("1", 1) is True

    def test_values_equal_string(self):
        assert _values_equal("Hello", "hello") is True

    def test_values_not_equal(self):
        assert _values_equal("a", "b") is False

    def test_count_matches_unordered(self):
        assert _count_matches_unordered(["a", "b"], ["b", "a"]) == 2
        assert _count_matches_unordered(["a", "b"], ["a"]) == 1
        assert _count_matches_unordered(["a"], ["b"]) == 0

    def test_count_matches_ordered(self):
        assert _count_matches_ordered(["a", "b"], ["a", "b"]) == 2
        assert _count_matches_ordered(["a", "b"], ["b", "a"]) == 0

    def test_pair_calls(self):
        expected = [{"name": "read"}, {"name": "write"}]
        actual = [{"name": "write", "arguments": {}}, {"name": "read", "arguments": {}}]
        pairs = _pair_calls(expected, actual, ignore_order=True)
        assert len(pairs) == 2

    def test_strict_compare(self):
        s = ToolCallingScorer(strict_args=True)
        assert s._compare_args({"a": 1}, {"a": 1}) == 1.0
        assert s._compare_args({"a": 1}, {"a": 2}) == 0.0

    def test_fuzzy_compare(self):
        s = ToolCallingScorer(strict_args=False)
        assert s._compare_args({"a": 1}, {"a": 1, "b": 2}) == 1.0

    def test_parse_args_string(self):
        assert ToolCallingScorer._parse_args('{"a": 1}') == {"a": 1}
        assert ToolCallingScorer._parse_args("invalid") == {}

    def test_parse_args_dict(self):
        assert ToolCallingScorer._parse_args({"a": 1}) == {"a": 1}


# --- GAIA Scorer ---


class TestGAIAVerificationScorer:
    @pytest.mark.asyncio
    async def test_no_expected_answer(self):
        s = GAIAVerificationScorer()
        score = await s.score(_case(), "output", {})
        assert score.value == 0.0

    @pytest.mark.asyncio
    async def test_exact_match(self):
        s = GAIAVerificationScorer()
        case = _case(expected_answer="42")
        score = await s.score(case, "42", {})
        assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        s = GAIAVerificationScorer()
        case = _case(expected_answer="Hello World")
        score = await s.score(case, "hello world", {})
        assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_numeric_match(self):
        s = GAIAVerificationScorer()
        case = _case(expected_answer="1000")
        score = await s.score(case, "1,000", {})
        assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_numeric_mismatch(self):
        s = GAIAVerificationScorer()
        case = _case(expected_answer="1000")
        score = await s.score(case, "2000", {})
        assert score.value == 0.0

    @pytest.mark.asyncio
    async def test_unit_stripping(self):
        s = GAIAVerificationScorer()
        case = _case(expected_answer="50 percent")
        score = await s.score(case, "50%", {})
        assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_partial_match(self):
        s = GAIAVerificationScorer()
        case = _case(expected_answer="42")
        score = await s.score(case, "the answer is 42", {})
        assert score.value == 0.8

    @pytest.mark.asyncio
    async def test_mismatch(self):
        s = GAIAVerificationScorer()
        case = _case(expected_answer="apple")
        score = await s.score(case, "orange", {})
        assert score.value == 0.0

    @pytest.mark.asyncio
    async def test_custom_verification(self):
        s = GAIAVerificationScorer()
        case = _case(expected_answer="42")
        case.verification_fn = lambda a, e: "42" in a
        score = await s.score(case, "the answer is 42", {})
        assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_custom_verification_error(self):
        s = GAIAVerificationScorer()
        case = _case(expected_answer="42")
        case.verification_fn = lambda a, e: 1 / 0
        score = await s.score(case, "42", {})
        assert score.value == 0.0

    def test_normalize_answer(self):
        s = GAIAVerificationScorer()
        assert s._normalize_answer("  $1,000.00 USD  ") == "1000.00"

    def test_try_numeric_compare(self):
        s = GAIAVerificationScorer()
        assert s._try_numeric_compare("42", "42") is True
        assert s._try_numeric_compare("42", "43") is False
        assert s._try_numeric_compare("hello", "world") is None


# --- SWE-bench Scorer ---


class TestPatchGenerationScorer:
    @pytest.mark.asyncio
    async def test_no_expected_patch(self):
        s = PatchGenerationScorer()
        score = await s.score(_case(), "out", {})
        assert score.value == 0.0

    @pytest.mark.asyncio
    async def test_no_patch_in_output(self):
        s = PatchGenerationScorer()
        case = _case(expected_answer="diff --git a/f.py b/f.py")
        score = await s.score(case, "no patch here", {})
        assert score.value == 0.0

    @pytest.mark.asyncio
    async def test_with_patch(self):
        s = PatchGenerationScorer()
        patch = "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1,3 +1,4 @@\n+new line"
        case = _case(expected_answer=patch)
        score = await s.score(case, patch, {})
        assert score.value > 0.5

    def test_extract_patch_diff_git(self):
        output = "some text\ndiff --git a/file.py b/file.py\n--- a/file.py"
        result = PatchGenerationScorer._extract_patch(output)
        assert result is not None
        assert "diff --git" in result

    def test_extract_patch_markers(self):
        output = "some text\n--- a/file.py\n+++ b/file.py"
        result = PatchGenerationScorer._extract_patch(output)
        assert result is not None

    def test_extract_patch_hunks(self):
        output = "text\n@@ -1,3 +1,4 @@\n+line"
        result = PatchGenerationScorer._extract_patch(output)
        assert result is not None

    def test_extract_patch_none(self):
        assert PatchGenerationScorer._extract_patch("just plain text") is None

    def test_file_overlap(self):
        gen = "diff --git a/f.py b/f.py"
        exp = "diff --git a/f.py b/f.py"
        assert PatchGenerationScorer._file_overlap(gen, exp) == 1.0

    def test_file_overlap_different(self):
        gen = "diff --git a/a.py b/a.py"
        exp = "diff --git a/b.py b/b.py"
        score = PatchGenerationScorer._file_overlap(gen, exp)
        assert score == 0.0

    def test_structural_similarity(self):
        patch = "@@ -1,3 +1,4 @@\n+line1\n+line2"
        assert PatchGenerationScorer._structural_similarity(patch, patch) > 0.5


class TestDiffHelpers:
    def test_extract_filenames(self):
        patch = "diff --git a/src/main.py b/src/main.py\n--- a/src/main.py\n+++ b/src/main.py"
        files = _extract_filenames(patch)
        assert "src/main.py" in files

    def test_diff_stats(self):
        patch = "@@ -1,3 +1,5 @@\n+line1\n+line2\n-old"
        stats = _diff_stats(patch)
        assert stats["hunks"] == 1
        assert stats["additions"] == 2
        assert stats["deletions"] == 1


# --- RAGAS Scorer (fail-open) ---


class TestRAGASScorer:
    @pytest.mark.asyncio
    async def test_not_installed(self):
        """If ragas is not installed, should return 0.0 gracefully."""
        import koboi.eval.scorers.ragas_scorer as mod

        original = mod._RAGAS_AVAILABLE
        mod._RAGAS_AVAILABLE = False
        try:
            from koboi.eval.scorers.ragas_scorer import RAGASScorer

            s = RAGASScorer("faithfulness")
            score = await s.score(_case(), "out", {})
            assert score.value == 0.0
            assert "not installed" in score.reason
        finally:
            mod._RAGAS_AVAILABLE = original


# --- DeepEval Scorer (fail-open) ---


class TestDeepEvalScorer:
    @pytest.mark.asyncio
    async def test_not_installed(self):
        """If deepeval is not installed, should return 0.0 gracefully."""
        import koboi.eval.scorers.deepeval_scorer as mod

        original = mod._DEEPEVAL_AVAILABLE
        mod._DEEPEVAL_AVAILABLE = False
        try:
            from koboi.eval.scorers.deepeval_scorer import DeepEvalScorer

            s = DeepEvalScorer("task_completion")
            score = await s.score(_case(), "out", {})
            assert score.value == 0.0
            assert "not installed" in score.reason
        finally:
            mod._DEEPEVAL_AVAILABLE = original

    @pytest.mark.asyncio
    async def test_safety_not_installed(self):
        import koboi.eval.scorers.deepeval_scorer as mod

        original = mod._DEEPEVAL_AVAILABLE
        mod._DEEPEVAL_AVAILABLE = False
        try:
            from koboi.eval.scorers.deepeval_scorer import DeepEvalSafetyScorer

            s = DeepEvalSafetyScorer()
            score = await s.score(_case(), "out", {})
            assert score.value == 0.5
        finally:
            mod._DEEPEVAL_AVAILABLE = original

    @pytest.mark.asyncio
    async def test_agentic_not_installed(self):
        import koboi.eval.scorers.deepeval_scorer as mod

        original = mod._DEEPEVAL_AVAILABLE
        mod._DEEPEVAL_AVAILABLE = False
        try:
            from koboi.eval.scorers.deepeval_scorer import DeepEvalAgenticScorer

            s = DeepEvalAgenticScorer()
            score = await s.score(_case(), "out", {})
            assert score.value == 0.0
        finally:
            mod._DEEPEVAL_AVAILABLE = original
