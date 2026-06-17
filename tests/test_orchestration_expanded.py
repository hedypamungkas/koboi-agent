"""Tests for koboi/harness/telemetry.py — TelemetryCollector expanded."""
from __future__ import annotations

import pytest

from koboi.harness.telemetry import TelemetryCollector, IterationRecord


class TestTelemetryCollectorLifecycle:
    def test_session_lifecycle(self):
        tc = TelemetryCollector(session_id="test")
        tc.session_start()
        assert tc.snapshot.start_time > 0
        tc.session_end()
        assert tc.snapshot.end_time > 0

    def test_iteration_tracking(self):
        tc = TelemetryCollector()
        tc.session_start()
        tc.iteration_start(tokens_current=100)
        tc.iteration_end(iteration=1, tool_names=["search"], tokens_after=200)
        assert tc.snapshot.total_iterations == 1
        assert len(tc.snapshot.iterations) == 1
        assert tc.snapshot.tokens_consumed_total == 100

    def test_multiple_iterations(self):
        tc = TelemetryCollector()
        tc.session_start()
        for i in range(5):
            tc.iteration_start(tokens_current=i * 100)
            tc.iteration_end(iteration=i, tool_names=["tool"], tokens_after=(i + 1) * 100)
        assert tc.snapshot.total_iterations == 5
        assert tc.snapshot.tokens_consumed_total == 500


class TestTelemetryCollectorMetrics:
    def test_record_tool_call(self):
        tc = TelemetryCollector()
        tc.record_tool_call("search")
        tc.record_tool_call("search")
        tc.record_tool_call("read")
        assert tc.snapshot.total_tool_calls == 3
        assert tc.snapshot.unique_tools_used == {"search", "read"}

    def test_record_tool_success_failure(self):
        tc = TelemetryCollector()
        tc.record_tool_success()
        tc.record_tool_success()
        tc.record_tool_failure()
        assert tc.snapshot.tools_succeeded == 2
        assert tc.snapshot.tools_failed == 1

    def test_record_compaction(self):
        tc = TelemetryCollector()
        tc.record_compaction(iteration=3, messages_before=50, messages_after=20,
                            tokens_before=5000, tokens_after=2000)
        assert len(tc.snapshot.compactions) == 1
        assert tc.snapshot.compactions[0].messages_before == 50

    def test_record_permission(self):
        tc = TelemetryCollector()
        tc.record_permission("shell", "allowed", "auto_rule")
        tc.record_permission("shell", "confirmed", "confirm_rule")
        tc.record_permission("delete", "denied", "deny_rule")
        assert len(tc.snapshot.permissions) == 3

    def test_record_doom_loop(self):
        tc = TelemetryCollector()
        tc.record_doom_loop()
        tc.record_doom_loop()
        assert tc.snapshot.doom_loops_detected == 2

    def test_record_skill_activation(self):
        tc = TelemetryCollector()
        tc.record_skill_activation("coding")
        tc.record_skill_activation("writing")
        assert tc.snapshot.skills_activated == ["coding", "writing"]

    def test_record_carryover_update(self):
        tc = TelemetryCollector()
        tc.record_carryover_update()
        tc.record_carryover_update()
        assert tc.snapshot.carryover_updates == 2


class TestTelemetryComputedMetrics:
    def test_context_efficiency_no_iterations(self):
        tc = TelemetryCollector()
        assert tc.context_efficiency() == 1.0

    def test_context_efficiency_all_productive(self):
        tc = TelemetryCollector()
        tc.session_start()
        tc.iteration_start(tokens_current=100)
        tc.iteration_end(iteration=1, tokens_after=200, was_productive=True)
        assert tc.context_efficiency() == 1.0

    def test_tool_utilization(self):
        tc = TelemetryCollector()
        tc.record_tool_call("search")
        tc.record_tool_call("search")
        tc.record_tool_call("read")
        util = tc.tool_utilization()
        assert "search" in util
        assert "read" in util

    def test_tool_utilization_no_calls(self):
        tc = TelemetryCollector()
        assert tc.tool_utilization() == {}

    def test_loop_health(self):
        tc = TelemetryCollector()
        tc.session_start()
        tc.iteration_start()
        tc.iteration_end(iteration=1, was_productive=True)
        tc.iteration_start()
        tc.iteration_end(iteration=2, was_productive=False)
        assert tc.loop_health() == 0.5

    def test_tool_success_rate(self):
        tc = TelemetryCollector()
        assert tc.tool_success_rate() == 1.0
        tc.record_tool_success()
        tc.record_tool_success()
        tc.record_tool_failure()
        assert tc.tool_success_rate() == pytest.approx(2 / 3, abs=0.01)

    def test_permission_friction(self):
        tc = TelemetryCollector()
        assert tc.permission_friction() == 1.0
        tc.record_permission("t", "allowed")
        tc.record_permission("t", "allowed")
        tc.record_permission("t", "denied")
        assert tc.permission_friction() == pytest.approx(2 / 3, abs=0.01)

    def test_compaction_fidelity(self):
        tc = TelemetryCollector()
        assert tc.compaction_fidelity() == 1.0
        tc.record_compaction(1, 50, 25, 5000, 2500)
        assert tc.compaction_fidelity() == 0.5

    def test_health_score(self):
        tc = TelemetryCollector()
        assert tc.health_score() == 100.0
        tc.session_start()
        tc.iteration_start()
        tc.iteration_end(iteration=1, was_productive=True)
        tc.record_tool_success()
        score = tc.health_score()
        assert 0 <= score <= 100

    def test_health_score_with_doom_loops(self):
        tc = TelemetryCollector()
        tc.session_start()
        for i in range(10):
            tc.iteration_start()
            tc.iteration_end(iteration=i, was_productive=True)
        tc.record_doom_loop()
        tc.record_doom_loop()
        score = tc.health_score()
        assert score < 100


class TestTelemetryReport:
    def test_report_format(self):
        tc = TelemetryCollector(session_id="sess1")
        tc.session_start()
        tc.iteration_start()
        tc.iteration_end(iteration=1, tool_names=["search"], tokens_after=100)
        tc.record_tool_call("search")
        tc.record_tool_success()
        tc.session_end()

        report = tc.report()
        assert report["session_id"] == "sess1"
        assert report["total_iterations"] == 1
        assert report["total_tool_calls"] == 1
        assert "metrics" in report
        assert "health_score" in report["metrics"]
        assert "permissions" in report

    def test_summary_format(self):
        tc = TelemetryCollector(session_id="test")
        tc.session_start()
        tc.session_end()
        summary = tc.summary()
        assert "test" in summary
        assert "Health Score" in summary


class TestTelemetryCustomWeights:
    def test_custom_health_weights(self):
        weights = {
            "loop_health": 0.5,
            "tool_success_rate": 0.1,
            "context_efficiency": 0.1,
            "compaction_fidelity": 0.1,
            "permission_friction": 0.1,
            "doom_penalty": 0.1,
        }
        tc = TelemetryCollector(health_weights=weights)
        tc.session_start()
        tc.iteration_start()
        tc.iteration_end(iteration=1, was_productive=False)
        score = tc.health_score()
        # With loop_health at 0.5 weight and 0% productivity
        assert score <= 50
