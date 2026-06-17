"""Tests for koboi/eval/regression.py -- Regression tracking."""
from __future__ import annotations

from koboi.eval.regression import RegressionTracker, RegressionReport
from koboi.types import EvalResult, EvalScore


def _make_result(name: str, score: float) -> EvalResult:
    return EvalResult(
        case_name=name,
        output="output",
        scores=[EvalScore("s", score, "")],
        overall_score=score,
        passed=score >= 0.6,
        duration_seconds=1.0,
    )


class TestRegressionReport:
    def test_has_regression(self):
        r = RegressionReport()
        assert r.has_regression is False
        r.regressed.append("case1")
        assert r.has_regression is True

    def test_summary_no_regressions(self):
        r = RegressionReport(improved=["a"], unchanged=["b"])
        text = r.summary()
        assert "Improved:   1" in text
        assert "Unchanged:  1" in text

    def test_summary_with_regressions(self):
        r = RegressionReport(regressed=["c"], score_delta={"c": -0.15})
        text = r.summary()
        assert "Regressed:  1" in text
        assert "c:" in text


class TestRegressionTracker:
    def test_save_and_load(self, tmp_path):
        tracker = RegressionTracker(str(tmp_path / "baselines"))
        results = [_make_result("case1", 0.8), _make_result("case2", 0.9)]
        path = tracker.save_baseline("my_suite", results)
        assert path.exists()
        loaded = tracker.load_baseline("my_suite")
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0]["case_name"] == "case1"

    def test_load_nonexistent(self, tmp_path):
        tracker = RegressionTracker(str(tmp_path / "baselines"))
        assert tracker.load_baseline("nope") is None

    def test_compare_improved(self, tmp_path):
        tracker = RegressionTracker(str(tmp_path))
        current = [_make_result("c1", 0.9)]
        baseline = [{"case_name": "c1", "overall_score": 0.6}]
        report = tracker.compare(current, baseline, threshold=0.05)
        assert "c1" in report.improved

    def test_compare_regressed(self, tmp_path):
        tracker = RegressionTracker(str(tmp_path))
        current = [_make_result("c1", 0.4)]
        baseline = [{"case_name": "c1", "overall_score": 0.8}]
        report = tracker.compare(current, baseline, threshold=0.05)
        assert "c1" in report.regressed

    def test_compare_unchanged(self, tmp_path):
        tracker = RegressionTracker(str(tmp_path))
        current = [_make_result("c1", 0.8)]
        baseline = [{"case_name": "c1", "overall_score": 0.81}]
        report = tracker.compare(current, baseline, threshold=0.05)
        assert "c1" in report.unchanged

    def test_compare_new_case(self, tmp_path):
        tracker = RegressionTracker(str(tmp_path))
        current = [_make_result("new_case", 0.9)]
        baseline = []
        report = tracker.compare(current, baseline)
        assert "new_case" in report.new_cases

    def test_compare_removed_case(self, tmp_path):
        tracker = RegressionTracker(str(tmp_path))
        current = []
        baseline = [{"case_name": "old_case", "overall_score": 0.8}]
        report = tracker.compare(current, baseline)
        assert "old_case" in report.removed_cases
