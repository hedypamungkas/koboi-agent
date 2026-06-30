"""tests/test_scenario_assertions -- Buckets E+F: harness keyword semantics.

Pure-logic tests (no LLM, no server) for the e2e assertion helpers:

- ``_kw_match``: number normalization (``"1,260"`` matches ``"1260"``).
- ``ScenarioExecutor._evaluate_keywords`` / ``_count_assertions``:
  * sequential AND over ``expect_keywords`` + OR over ``expect_any_of``;
  * concurrent fan-out: each session must match >=1 keyword (OR per session).
"""

from __future__ import annotations

from tests.e2e.framework.scenario import Scenario, ScenarioExecutor, Turn, TurnResult, _kw_match


def _tr(content: str) -> TurnResult:
    return TurnResult(
        message="m",
        events=[],
        content=content,
        tool_calls=[],
        tool_results=[],
        token_usage=None,
        latency_seconds=0.0,
        timestamp="",
    )


class TestKwMatch:
    def test_plain_substring_case_insensitive(self):
        assert _kw_match("Express shipping is $9.99", "9.99")

    def test_number_normalization_comma(self):
        # calc_compound_verify: keyword "1260" vs model output "1,260".
        assert _kw_match("15% of 8400 is 1,260.", "1260")

    def test_number_normalization_reverse(self):
        assert _kw_match("result is 1260", "1,260")

    def test_no_match(self):
        assert not _kw_match("hello world", "zzz")


class TestEvaluateSequential:
    def test_expect_any_of_passes_on_one_match(self):
        # rag_acme_erp_price: "$15,000/year" should satisfy any_of ["15,000","15000"].
        sc = Scenario("s", "rag", [Turn("q", expect_any_of=["15,000", "15000"])])
        ok, err = ScenarioExecutor._evaluate_keywords(sc, [_tr("annual price is $15,000/year")])
        assert ok and err is None

    def test_expect_any_of_fails_on_none(self):
        sc = Scenario("s", "rag", [Turn("q", expect_any_of=["15,000", "15000"])])
        ok, err = ScenarioExecutor._evaluate_keywords(sc, [_tr("price is twelve thousand")])
        assert not ok and err is not None and "None of" in err

    def test_expect_keywords_are_AND(self):
        sc = Scenario("s", "rag", [Turn("q", expect_keywords=["Standard", "Deluxe"])])
        assert ScenarioExecutor._evaluate_keywords(sc, [_tr("Standard and Deluxe rooms")])[0]
        assert not ScenarioExecutor._evaluate_keywords(sc, [_tr("Standard rooms only")])[0]

    def test_number_normalization_applies_to_keywords(self):
        sc = Scenario("s", "multi_tool", [Turn("q", expect_keywords=["1260"])])
        assert ScenarioExecutor._evaluate_keywords(sc, [_tr("15% of 8400 is 1,260.")])[0]

    def test_skill_off_by_one_accepts_description(self):
        # skill_codereview_off_by_one: any_of synonyms for the skipped element.
        sc = Scenario("s", "skills", [Turn("q", expect_any_of=["off", "missing", "skips", "omits", "first element"])])
        assert ScenarioExecutor._evaluate_keywords(sc, [_tr("range(1,n) yields Missing Index 0")])[0]


class TestEvaluateConcurrent:
    def _stress_color(self) -> Scenario:
        return Scenario(
            "s",
            "stress",
            [Turn("name a primary color", expect_keywords=["blue", "red", "green", "yellow"])],
            concurrent=5,
        )

    def test_each_session_one_color_passes(self):
        sc = self._stress_color()
        results = [_tr(c) for c in ["blue", "red", "red", "green", "yellow"]]
        ok, err = ScenarioExecutor._evaluate_keywords(sc, results)
        assert ok and err is None

    def test_one_session_no_color_fails(self):
        sc = self._stress_color()
        results = [_tr(c) for c in ["blue", "purple", "red", "green", "yellow"]]
        ok, err = ScenarioExecutor._evaluate_keywords(sc, results)
        assert not ok and err is not None and "Concurrent session 2" in err

    def test_concurrent_counts(self):
        sc = self._stress_color()
        results = [_tr(c) for c in ["blue", "purple", "red", "green", "yellow"]]
        checked, passed = ScenarioExecutor._count_assertions(sc, results)
        assert checked == 5
        assert passed == 4  # "purple" is not a primary color


class TestCountSequential:
    def test_counts_include_any_of(self):
        sc = Scenario("s", "rag", [Turn("q", expect_keywords=["a"], expect_any_of=["x", "y"])])
        checked, passed = ScenarioExecutor._count_assertions(sc, [_tr("a and x")])
        assert checked == 2  # 1 keyword + 1 any_of
        assert passed == 2

    def test_any_of_no_match_not_counted_as_passed(self):
        sc = Scenario("s", "rag", [Turn("q", expect_keywords=["alpha"], expect_any_of=["xray", "yankee"])])
        checked, passed = ScenarioExecutor._count_assertions(sc, [_tr("matched alpha only")])
        assert checked == 2
        assert passed == 1  # keyword matched, any_of did not
