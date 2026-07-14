"""Tests for A5 -- conformal-style threshold selector (calibration on-ramp)."""

from __future__ import annotations

from koboi.eval.scorers.calibration import select_conformal_threshold


class TestSelectConformalThreshold:
    def test_separable_set_returns_threshold(self):
        # Low-confidence all wrong; high-confidence all right.
        samples = [(0.9, True), (0.85, True), (0.8, True), (0.3, False), (0.2, False)]
        thresh = select_conformal_threshold(samples, target_correctness=0.9)
        assert thresh is not None
        # The threshold should sit in the gap (0.3..0.8); kept-set (>= thresh) all correct.
        kept = [ok for c, ok in samples if c >= thresh]
        assert all(kept) and len(kept) >= 2

    def test_unachievable_target_returns_none(self):
        # All wrong -> no threshold clears any positive target.
        samples = [(0.9, False), (0.8, False), (0.7, False)]
        assert select_conformal_threshold(samples, target_correctness=0.5) is None

    def test_empty_samples_returns_none(self):
        assert select_conformal_threshold([], target_correctness=0.9) is None

    def test_none_confidence_filtered(self):
        samples = [(None, True), (0.9, True), (0.8, True)]
        thresh = select_conformal_threshold(samples, target_correctness=0.9)
        assert thresh is not None  # None-confidence samples filtered out

    def test_lower_target_allows_lower_threshold(self):
        # A stricter target forces a higher (more selective) threshold.
        samples = [
            (1.0, True), (0.9, True), (0.8, True), (0.7, True), (0.6, False), (0.5, False), (0.4, False)
        ]
        strict = select_conformal_threshold(samples, target_correctness=0.99)
        loose = select_conformal_threshold(samples, target_correctness=0.5)
        # Strict target -> threshold >= loose target (more abstention). Both not None here.
        assert strict is not None and loose is not None
        assert strict >= loose
