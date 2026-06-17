"""Tests for koboi/harness/carryover.py — CarryoverState and serialization."""

from __future__ import annotations

import json

import pytest

from koboi.harness.carryover import (
    CarryoverState,
    WorkLogEntry,
    _parse_counts,
    _parse_dict,
    _parse_list,
    _try_json_parse_counts,
    _try_json_parse_dict,
    _try_json_parse_list,
)


class TestWorkLogEntry:
    def test_defaults(self):
        entry = WorkLogEntry(iteration=1, action="tool_call", detail="did something")
        assert entry.success is True

    def test_failure(self):
        entry = WorkLogEntry(iteration=2, action="llm_response", detail="failed", success=False)
        assert entry.success is False


class TestCarryoverState:
    def test_initial_state(self):
        state = CarryoverState()
        assert state.user_goals == []
        assert state.completed_goals == []
        assert state.active_artifacts == {}
        assert state.verified_work == []
        assert state.work_log == []
        assert state.invoked_tools == {}
        assert state.skills_used == []

    def test_add_goal(self):
        state = CarryoverState()
        state.add_goal("Build API")
        assert "Build API" in state.user_goals
        state.add_goal("Build API")  # no duplicate
        assert state.user_goals.count("Build API") == 1

    def test_add_goal_respects_max(self):
        state = CarryoverState(max_goals=2)
        state.add_goal("a")
        state.add_goal("b")
        state.add_goal("c")
        assert len(state.user_goals) == 2
        assert state.user_goals == ["b", "c"]

    def test_complete_goal(self):
        state = CarryoverState()
        state.add_goal("Build API")
        state.complete_goal("Build API")
        assert "Build API" not in state.user_goals
        assert "Build API" in state.completed_goals

    def test_complete_goal_not_present(self):
        state = CarryoverState()
        state.complete_goal("nonexistent")  # should not crash

    def test_complete_goal_respects_max(self):
        state = CarryoverState(max_goals=2)
        state.add_goal("a")
        state.add_goal("b")
        state.complete_goal("a")
        state.complete_goal("b")
        state.complete_goal("c")  # not present, ignored
        assert len(state.completed_goals) == 2

    def test_add_artifact(self):
        state = CarryoverState()
        state.add_artifact("file.py", "Main module")
        assert state.active_artifacts["file.py"] == "Main module"

    def test_add_artifact_evicts_oldest(self):
        state = CarryoverState(max_artifacts=2)
        state.add_artifact("a", "first")
        state.add_artifact("b", "second")
        state.add_artifact("c", "third")
        assert "a" not in state.active_artifacts
        assert len(state.active_artifacts) == 2

    def test_mark_verified(self):
        state = CarryoverState()
        state.mark_verified("Tests pass")
        assert "Tests pass" in state.verified_work
        state.mark_verified("Tests pass")  # no duplicate
        assert state.verified_work.count("Tests pass") == 1

    def test_mark_verified_respects_max(self):
        state = CarryoverState(max_verified=2)
        state.mark_verified("a")
        state.mark_verified("b")
        state.mark_verified("c")
        assert len(state.verified_work) == 2

    def test_record_tool_use(self):
        state = CarryoverState()
        state.record_tool_use("search", "pattern", "found 3", iteration=1)
        assert state.invoked_tools["search"] == 1
        assert len(state.work_log) == 1
        assert state.work_log[0].action == "tool_call"

    def test_record_tool_use_truncates_log(self):
        state = CarryoverState(max_log_entries=3)
        for i in range(5):
            state.record_tool_use("tool", f"args{i}", f"result{i}", iteration=i)
        assert len(state.work_log) == 3

    def test_record_tool_use_counts(self):
        state = CarryoverState()
        state.record_tool_use("t", "a", "r")
        state.record_tool_use("t", "a", "r")
        assert state.invoked_tools["t"] == 2

    def test_record_skill(self):
        state = CarryoverState()
        state.record_skill("coding")
        assert "coding" in state.skills_used
        state.record_skill("coding")  # no duplicate
        assert state.skills_used.count("coding") == 1

    def test_to_context_message_empty(self):
        state = CarryoverState()
        assert state.to_context_message() == ""

    def test_to_context_message_with_data(self):
        state = CarryoverState()
        state.add_goal("Build API")
        state.add_artifact("main.py", "entry point")
        state.record_tool_use("search", "pattern", "found")
        state.record_skill("coding")
        state.mark_verified("Tests pass")
        msg = state.to_context_message()
        assert "<harness-carryover>" in msg
        assert "Goals:" in msg
        assert "Artifacts:" in msg
        assert "Tools used:" in msg
        assert "Skills:" in msg
        assert "Verified:" in msg

    def test_to_context_message_completed_only(self):
        state = CarryoverState()
        state.complete_goal("a")  # no goals to complete, nothing happens
        msg = state.to_context_message()
        # No goals, artifacts, tools, skills, verified -> empty
        assert msg == ""

    def test_from_context_message_roundtrip(self):
        state = CarryoverState()
        state.add_goal("Build API")
        state.add_goal("Write tests")
        state.complete_goal("Build API")
        state.add_artifact("main.py", "entry point")
        state.record_tool_use("search", "q", "found")
        state.record_skill("coding")
        state.mark_verified("Tests pass")
        msg = state.to_context_message()

        restored = CarryoverState.from_context_message(msg)
        assert restored.user_goals == ["Write tests"]
        assert restored.completed_goals == ["Build API"]
        assert restored.active_artifacts["main.py"] == "entry point"
        assert restored.invoked_tools["search"] == 1
        assert restored.skills_used == ["coding"]
        assert restored.verified_work == ["Tests pass"]

    def test_from_context_message_no_tag(self):
        state = CarryoverState.from_context_message("random text without tags")
        assert state.user_goals == []

    def test_summary(self):
        state = CarryoverState()
        state.add_goal("a")
        state.record_tool_use("t", "a", "r")
        s = state.summary()
        assert s["goals"] == 1
        assert s["tool_calls"] == 1
        assert s["unique_tools"] == 1


class TestParseHelpers:
    def test_parse_list_json(self):
        assert _try_json_parse_list('["a", "b"]') == ["a", "b"]

    def test_parse_list_invalid_json(self):
        # _try_json_parse_list falls through to _parse_list on JSON error
        result = _try_json_parse_list("not json")
        assert isinstance(result, list)

    def test_parse_list_bracket_format(self):
        assert _parse_list("[a, b, c]") == ["a", "b", "c"]

    def test_parse_list_empty(self):
        assert _parse_list("") == []
        assert _parse_list("[]") == []

    def test_parse_dict_json(self):
        assert _try_json_parse_dict('{"a": "b"}') == {"a": "b"}

    def test_parse_dict_invalid(self):
        assert _try_json_parse_dict("not json") == {}

    def test_parse_dict_colon_format(self):
        assert _parse_dict("a: b, c: d") == {"a": "b", "c": "d"}

    def test_parse_counts_json(self):
        assert _try_json_parse_counts('{"a": 3}') == {"a": 3}

    def test_parse_counts_invalid(self):
        assert _try_json_parse_counts("not json") == {}

    def test_parse_counts_colon_format(self):
        assert _parse_counts("a: 3, b: 5") == {"a": 3, "b": 5}
