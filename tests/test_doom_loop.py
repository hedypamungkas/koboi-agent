"""Tests for koboi/harness/doom_loop.py -- Doom loop detection."""

from __future__ import annotations


from koboi.harness.doom_loop import (
    DoomLoopConfig,
    DoomLoopDetector,
    DoomLoopResult,
)


class TestDoomLoopConfig:
    def test_defaults(self):
        cfg = DoomLoopConfig()
        assert cfg.consecutive_identical_threshold == 3
        assert cfg.repeating_pattern_window == 6
        assert cfg.repeating_pattern_threshold == 2
        assert cfg.error_retry_threshold == 3
        assert cfg.enable_recovery is True
        assert cfg.adaptive_threshold is False


class TestDoomLoopResult:
    def test_defaults(self):
        r = DoomLoopResult()
        assert r.detected is False
        assert r.loop_type == ""
        assert r.iterations_wasted == 0


class TestDoomLoopDetector:
    def test_no_loop_short_history(self):
        d = DoomLoopDetector()
        d.record("read", "f.py")
        d.record("write", "f.py")
        result = d.check()
        assert result.detected is False

    def test_consecutive_identical_detected(self):
        d = DoomLoopDetector()
        for _ in range(3):
            d.record("read", "f.py")
        result = d.check()
        assert result.detected is True
        assert result.loop_type == "consecutive_identical"
        assert "read" in result.pattern_description

    def test_consecutive_identical_not_detected_different_args(self):
        d = DoomLoopDetector()
        d.record("read", "f1.py")
        d.record("read", "f2.py")
        d.record("read", "f3.py")
        result = d.check()
        assert result.detected is False

    def test_error_retry_detected(self):
        d = DoomLoopDetector()
        d.record("shell", "cmd1", is_error=True)
        d.record("shell", "cmd1", is_error=True)
        d.record("shell", "cmd1", is_error=True)
        result = d.check()
        assert result.detected is True
        assert result.loop_type == "error_retry"

    def test_error_retry_not_enough(self):
        d = DoomLoopDetector()
        d.record("shell", "cmd1", is_error=True)
        d.record("shell", "cmd1", is_error=True)
        d.record("read", "f.py", is_error=False)
        result = d.check()
        assert result.detected is False

    def test_repeating_pattern_detected(self):
        d = DoomLoopDetector()
        # A,B,A,B,A,B pattern
        for _ in range(3):
            d.record("read", "f.py")
            d.record("write", "f.py")
        result = d.check()
        assert result.detected is True
        assert result.loop_type == "repeating_pattern"

    def test_reset(self):
        d = DoomLoopDetector()
        d.record("read", "f.py")
        d.record("read", "f.py")
        d.record("read", "f.py")
        assert d.check().detected is True
        d.reset()
        assert d.check().detected is False
        assert len(d.history) == 0

    def test_history_property(self):
        d = DoomLoopDetector()
        d.record("read", "f.py")
        h = d.history
        assert len(h) == 1
        assert h[0] == ("read", "f.py")

    def test_estimate_complexity_simple(self):
        d = DoomLoopDetector()
        assert d.estimate_complexity(2, 50) == "simple"

    def test_estimate_complexity_moderate(self):
        d = DoomLoopDetector()
        assert d.estimate_complexity(5, 100) == "moderate"

    def test_estimate_complexity_complex(self):
        d = DoomLoopDetector()
        assert d.estimate_complexity(10, 500) == "complex"

    def test_effective_threshold_no_adaptive(self):
        d = DoomLoopDetector(DoomLoopConfig(adaptive_threshold=False))
        assert d.get_effective_threshold() == 3

    def test_effective_threshold_complex(self):
        d = DoomLoopDetector(
            DoomLoopConfig(
                adaptive_threshold=True,
                task_complexity_hint="complex",
            )
        )
        assert d.get_effective_threshold() > 3

    def test_effective_threshold_moderate(self):
        d = DoomLoopDetector(
            DoomLoopConfig(
                adaptive_threshold=True,
                task_complexity_hint="moderate",
            )
        )
        assert d.get_effective_threshold() >= 3

    def test_recovery_hint_present(self):
        d = DoomLoopDetector()
        for _ in range(3):
            d.record("read", "f.py")
        result = d.check()
        assert len(result.recovery_hint) > 0

    def test_custom_threshold(self):
        cfg = DoomLoopConfig(consecutive_identical_threshold=5)
        d = DoomLoopDetector(cfg)
        for _ in range(4):
            d.record("read", "f.py")
        assert d.check().detected is False
        d.record("read", "f.py")
        assert d.check().detected is True

    def test_adaptive_threshold_complex_delays_detection(self):
        cfg = DoomLoopConfig(
            consecutive_identical_threshold=3,
            adaptive_threshold=True,
            task_complexity_hint="complex",
        )
        d = DoomLoopDetector(cfg)
        # With complex hint, effective threshold should be > 3
        assert d.get_effective_threshold() > 3
        # 3 identical calls should NOT trigger (base threshold)
        for _ in range(3):
            d.record("read", "f.py")
        assert d.check().detected is False
        # Adding more should eventually trigger
        for _ in range(3):
            d.record("read", "f.py")
        assert d.check().detected is True

    def test_adaptive_threshold_simple_uses_base(self):
        cfg = DoomLoopConfig(
            consecutive_identical_threshold=3,
            adaptive_threshold=True,
            task_complexity_hint="simple",
        )
        d = DoomLoopDetector(cfg)
        assert d.get_effective_threshold() == 3
        for _ in range(3):
            d.record("read", "f.py")
        assert d.check().detected is True
