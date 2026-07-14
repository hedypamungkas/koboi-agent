"""Unit tests for the e2e scenario assertion logic.

Covers ``ScenarioExecutor._check_turn`` tool-alias support: a builtin tool
(``task_create``) must be satisfiable by a functionally-equivalent MCP tool
(``add_todo``) when an alias is declared, while staying strict otherwise.
"""

from __future__ import annotations

from tests.e2e.framework.scenario import ScenarioExecutor, Turn

# Two sentinel events so the default min_events=2 gate doesn't cloud the checks.
_EVENTS = [{"type": "tool_call"}, {"type": "complete"}]


def _executor() -> ScenarioExecutor:
    # _check_turn is pure (doesn't touch the client), so a dummy executor is fine.
    return ScenarioExecutor(client=None, base_url="", api_key="")


class TestCheckTurnToolAliases:
    def test_alias_satisfies_expected_tool(self):
        ex = _executor()
        turn = Turn("msg", expect_tools=["task_create"], tool_aliases={"task_create": ["add_todo"]})
        _, ok = ex._check_turn(turn, "ok", [{"tool_name": "add_todo"}], _EVENTS)
        assert ok, "add_todo should satisfy task_create via alias"

    def test_without_alias_mcp_tool_does_not_satisfy(self):
        ex = _executor()
        turn = Turn("msg", expect_tools=["task_create"])  # no aliases
        _, ok = ex._check_turn(turn, "ok", [{"tool_name": "add_todo"}], _EVENTS)
        assert not ok, "without an alias, add_todo must NOT satisfy task_create"

    def test_canonical_tool_name_still_matches(self):
        ex = _executor()
        turn = Turn("msg", expect_tools=["task_create"], tool_aliases={"task_create": ["add_todo"]})
        _, ok = ex._check_turn(turn, "ok", [{"tool_name": "task_create"}], _EVENTS)
        assert ok

    def test_multiple_expected_tools_all_satisfied_via_aliases(self):
        ex = _executor()
        turn = Turn(
            "msg",
            expect_tools=["task_create", "task_list"],
            tool_aliases={"task_create": ["add_todo"], "task_list": ["list_todos"]},
        )
        _, ok = ex._check_turn(turn, "ok", [{"tool_name": "add_todo"}, {"tool_name": "list_todos"}], _EVENTS)
        assert ok


class TestMultiToolAliasMap:
    def test_alias_map_shape(self):
        from tests.e2e.scenarios.multi_tool import TASK_TOOL_ALIASES

        assert TASK_TOOL_ALIASES == {"task_create": ["add_todo"], "task_list": ["list_todos"]}
