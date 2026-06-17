"""Tests for koboi.harness modules."""

from __future__ import annotations

import json
import os
import tempfile

from koboi.harness.doom_loop import DoomLoopDetector, DoomLoopConfig, DoomLoopResult
from koboi.harness.policy import PolicyEngine, PolicyRule, PolicyAction
from koboi.harness.carryover import CarryoverState
from koboi.harness.telemetry import TelemetryCollector
from koboi.harness.policy_audit import PolicyAuditLog
from koboi.types import RiskLevel


class TestDoomLoopDetector:
    def test_no_doom(self):
        detector = DoomLoopDetector()
        detector.record("tool_a", '{"x": 1}', is_error=False)
        detector.record("tool_b", '{"x": 2}', is_error=False)
        result = detector.check()
        assert result.detected is False

    def test_consecutive_identical(self):
        cfg = DoomLoopConfig(consecutive_identical_threshold=3)
        detector = DoomLoopDetector(cfg)
        for _ in range(3):
            detector.record("tool_a", '{"x": 1}', is_error=False)
        result = detector.check()
        assert result.detected is True
        detector.reset()

    def test_reset(self):
        cfg = DoomLoopConfig(consecutive_identical_threshold=2)
        detector = DoomLoopDetector(cfg)
        detector.record("t", "{}", is_error=False)
        detector.record("t", "{}", is_error=False)
        result = detector.check()
        assert result.detected is True
        detector.reset()
        detector.record("t", "{}", is_error=False)
        result = detector.check()
        assert result.detected is False


class TestPolicyEngine:
    def test_default_allow(self):
        engine = PolicyEngine()
        decision = engine.evaluate("safe_tool", "{}", RiskLevel.SAFE)
        assert decision.action == PolicyAction.ALLOW

    def test_deny_rule(self):
        engine = PolicyEngine(
            rules=[
                PolicyRule(name="deny_shell_rm", tool_pattern="run_shell", action=PolicyAction.DENY),
            ]
        )
        decision = engine.evaluate("run_shell", '{"cmd": "rm -rf /"}', RiskLevel.MODERATE)
        assert decision.action == PolicyAction.DENY

    def test_confirm_rule(self):
        engine = PolicyEngine(
            rules=[
                PolicyRule(name="confirm_delete", tool_pattern="delete_file", action=PolicyAction.CONFIRM),
            ]
        )
        decision = engine.evaluate("delete_file", '{"path": "/tmp/x"}', RiskLevel.DESTRUCTIVE)
        assert decision.action == PolicyAction.CONFIRM

    def test_hardcoded_safety(self):
        engine = PolicyEngine()
        decision = engine.evaluate("run_shell", '{"cmd": "cat /etc/shadow"}', RiskLevel.MODERATE)
        assert decision.action == PolicyAction.DENY


class TestCarryoverState:
    def test_add_goal(self):
        state = CarryoverState()
        state.add_goal("Research products")
        msg = state.to_context_message()
        assert "Research products" in msg

    def test_record_tool(self):
        state = CarryoverState()
        state.record_tool_use("web_search", '{"q": "test"}', "results", iteration=0)
        msg = state.to_context_message()
        assert "web_search" in msg

    def test_empty_state(self):
        state = CarryoverState()
        msg = state.to_context_message()
        assert msg is None or len(msg) == 0
