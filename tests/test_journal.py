"""Tests for the step journal (P2-A) in koboi.journal + loop integration."""

from __future__ import annotations

import sqlite3

import pytest

from koboi.journal import StepJournal
from koboi.memory_sqlite import SQLiteMemory

# Shared helpers from conftest (MockClient, make_mock_response, ...).
from tests.conftest import (
    MockClient,
    make_mock_response,
    make_mock_tool_call,
    make_tool_registry,
)


def _connect(db_path) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def _steps_rows(db_path, session_id):
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT turn_index, step_index, status, is_terminal, llm_prompt_tokens "
        "FROM steps WHERE session_id=? ORDER BY id",
        (session_id,),
    ).fetchall()
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# StepJournal unit tests
# ---------------------------------------------------------------------------


class TestStepJournalUnit:
    def test_initial_turn_is_zero_for_fresh_session(self, tmp_path):
        mem = SQLiteMemory(db_path=str(tmp_path / "t.db"), session_id="S1")
        j = StepJournal(mem._ensure_conn(), mem.session_id)
        assert j.turn_index == 0

    def test_advance_turn_increments(self, tmp_path):
        mem = SQLiteMemory(db_path=str(tmp_path / "t.db"), session_id="S1")
        j = StepJournal(mem._ensure_conn(), mem.session_id)
        j.advance_turn()
        j.advance_turn()
        assert j.turn_index == 2

    def test_record_step_upsert_one_row_per_turn_step(self, tmp_path):
        mem = SQLiteMemory(db_path=str(tmp_path / "t.db"), session_id="S1")
        j = StepJournal(mem._ensure_conn(), mem.session_id)
        j.advance_turn()
        j.record_step(turn_index=j.turn_index, step_index=0, status="running")
        j.record_step(turn_index=j.turn_index, step_index=0, status="complete", is_terminal=True)
        assert len(j.list_steps()) == 1  # upsert, not append
        assert j.get_last_terminal_step()["status"] == "complete"

    def test_open_running_excludes_completed_steps(self, tmp_path):
        mem = SQLiteMemory(db_path=str(tmp_path / "t.db"), session_id="S1")
        j = StepJournal(mem._ensure_conn(), mem.session_id)
        j.advance_turn()
        j.record_step(turn_index=j.turn_index, step_index=0, status="running")
        j.record_step(turn_index=j.turn_index, step_index=0, status="complete", is_terminal=True)
        # A different step left dangling (crash).
        j.record_step(turn_index=j.turn_index, step_index=1, status="running")
        open_rows = j.list_open_running()
        assert len(open_rows) == 1
        assert open_rows[0]["step_index"] == 1

    def test_mark_interrupted_flips_running_rows(self, tmp_path):
        mem = SQLiteMemory(db_path=str(tmp_path / "t.db"), session_id="S1")
        j = StepJournal(mem._ensure_conn(), mem.session_id)
        j.advance_turn()
        j.record_step(turn_index=j.turn_index, step_index=0, status="running")
        j.mark_interrupted(j.list_open_running())
        assert len(j.list_open_running()) == 0

    def test_resumed_journal_inherits_turn_numbering(self, tmp_path):
        db = str(tmp_path / "t.db")
        mem = SQLiteMemory(db_path=db, session_id="S1")
        j = StepJournal(mem._ensure_conn(), mem.session_id)
        j.advance_turn()  # turn 1
        j.record_step(turn_index=1, step_index=0, status="complete", is_terminal=True)
        mem.close()
        # New process, same session -> inherits turn 1.
        mem2 = SQLiteMemory(db_path=db, session_id="S1")
        j2 = StepJournal(mem2._ensure_conn(), mem2.session_id)
        assert j2.turn_index == 1


# ---------------------------------------------------------------------------
# Loop integration: journal records one row per iteration
# ---------------------------------------------------------------------------


def _core(db_path, session_id, responses, tools=None, journal=None):
    from koboi.loop import AgentCore

    mem = SQLiteMemory(db_path=db_path, session_id=session_id)
    j = journal if journal is not None else StepJournal(mem._ensure_conn(), mem.session_id)
    core = AgentCore(
        client=MockClient(responses),
        memory=mem,
        tools=tools or make_tool_registry(),
        journal=j,
        max_iterations=5,
    )
    return core, mem, j


class TestJournalLoopIntegration:
    async def test_terminal_run_records_complete_step(self, tmp_path):
        db = str(tmp_path / "t.db")
        core, mem, j = _core(db, "S1", [make_mock_response(content="done")])
        await core.run("hi")
        rows = _steps_rows(db, "S1")
        statuses = [(r[2], r[3]) for r in rows]  # (status, is_terminal)
        assert ("complete", 1) in statuses
        # The terminal step carries token deltas.
        complete = [r for r in rows if r[2] == "complete"][0]
        assert complete[4] == 10  # prompt_tokens from make_mock_response usage

    async def test_tool_calls_step_then_complete(self, tmp_path):
        db = str(tmp_path / "t.db")
        responses = [
            make_mock_response(tool_calls=[make_mock_tool_call("get_weather", {"city": "X"})]),
            make_mock_response(content="final"),
        ]
        core, mem, j = _core(db, "S1", responses)
        await core.run("hi")
        statuses = [r[2] for r in _steps_rows(db, "S1")]
        assert "tool_calls" in statuses
        assert "complete" in statuses

    async def test_max_iter_records_terminal(self, tmp_path):
        db = str(tmp_path / "t.db")
        # Every iteration returns a tool call -> never completes -> max_iter.
        tc = [make_mock_tool_call("get_weather", {"city": "X"})]
        responses = [make_mock_response(tool_calls=tc) for _ in range(5)]
        core, mem, j = _core(db, "S1", responses)
        with pytest.raises(Exception):
            await core.run("hi")
        statuses = [r[2] for r in _steps_rows(db, "S1")]
        assert "max_iter" in statuses
        assert any(r[3] == 1 and r[2] == "max_iter" for r in _steps_rows(db, "S1"))

    async def test_no_journal_when_disabled(self, tmp_path):
        db = str(tmp_path / "t.db")
        # Build AgentCore with journal=None.
        from koboi.loop import AgentCore

        mem = SQLiteMemory(db_path=db, session_id="S1")
        core = AgentCore(
            client=MockClient([make_mock_response(content="done")]),
            memory=mem,
            tools=make_tool_registry(),
            journal=None,
            max_iterations=3,
        )
        await core.run("hi")
        # steps table still exists (created by SQLiteMemory._init_db) but is empty.
        assert _steps_rows(db, "S1") == []

    async def test_turn_advances_across_runs(self, tmp_path):
        db = str(tmp_path / "t.db")
        core, mem, j = _core(db, "S1", [make_mock_response(content="one")])
        await core.run("first")
        rows = _steps_rows(db, "S1")
        assert all(r[0] == 1 for r in rows)  # turn_index = 1
        # Second run on a fresh core/journal inherits turn 1, advances to 2.
        core2, mem2, j2 = _core(db, "S1", [make_mock_response(content="two")], tools=core.tools)
        await core2.run("second")
        rows2 = _steps_rows(db, "S1")
        assert any(r[0] == 2 for r in rows2)
