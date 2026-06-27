"""Tests for session resume (P2-A) in koboi.loop.AgentCore.resume + facade."""

from __future__ import annotations

import sqlite3

import pytest

from koboi.journal import StepJournal
from koboi.memory_sqlite import SQLiteMemory
from koboi.loop import AgentCore

from tests.conftest import MockClient, make_mock_response, make_tool_registry


def _has_tool_result(messages, tool_call_id):
    return any(m.get("role") == "tool" and m.get("tool_call_id") == tool_call_id for m in messages)


class TestResume:
    async def test_resume_reexecutes_missing_tool_call(self, tmp_path):
        db = str(tmp_path / "t.db")
        sid = "S1"
        # Seed a crashed session: user msg + assistant tool_calls with NO result.
        mem = SQLiteMemory(db_path=db, session_id=sid)
        mem.add_user_message("check the weather")
        mem.add_assistant_message(
            "let me check",
            tool_calls=[
                {
                    "id": "call_X",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
                }
            ],
        )
        # Record a journal step at turn 1 so resume inherits turn numbering.
        j_seed = StepJournal(mem._ensure_conn(), mem.session_id)
        j_seed.advance_turn()
        j_seed.record_step(turn_index=1, step_index=0, status="running")
        mem.close()

        # New process resumes.
        mem2 = SQLiteMemory(db_path=db, session_id=sid)
        j2 = StepJournal(mem2._ensure_conn(), mem2.session_id)
        core = AgentCore(
            client=MockClient([make_mock_response(content="all done after resume")]),
            memory=mem2,
            tools=make_tool_registry(),
            journal=j2,
            max_iterations=5,
        )
        result = await core.resume()

        msgs = mem2.get_messages()
        assert _has_tool_result(msgs, "call_X")  # the missing call was re-run
        assert result.metadata["resumed"] is True
        # Resumed turn is inherited (no advance): turn 1 was the seeded turn.
        assert result.metadata["turn_index"] == 1

    async def test_resume_with_no_dangling_tools_just_continues(self, tmp_path):
        db = str(tmp_path / "t.db")
        sid = "S2"
        mem = SQLiteMemory(db_path=db, session_id=sid)
        mem.add_user_message("hello")
        mem.add_assistant_message("hi there")  # clean ending, no tool_calls
        mem.close()

        mem2 = SQLiteMemory(db_path=db, session_id=sid)
        j2 = StepJournal(mem2._ensure_conn(), mem2.session_id)
        core = AgentCore(
            client=MockClient([make_mock_response(content="continued")]),
            memory=mem2,
            tools=make_tool_registry(),
            journal=j2,
            max_iterations=3,
        )
        result = await core.resume()
        assert result.metadata["resumed"] is True
        # The loop re-invoked the LLM and appended its response.
        assert any(
            m.get("role") == "assistant" and "continued" in (m.get("content") or "") for m in mem2.get_messages()
        )

    async def test_resume_marks_prior_running_interrupted(self, tmp_path):
        db = str(tmp_path / "t.db")
        sid = "S3"
        # Seed a stale 'running' crash-marker at a step the resume loop won't
        # touch (step 99) so the 'interrupted' transition stays observable.
        mem = SQLiteMemory(db_path=db, session_id=sid)
        mem.add_user_message("q")
        j_seed = StepJournal(mem._ensure_conn(), mem.session_id)
        j_seed.advance_turn()
        j_seed.record_step(turn_index=j_seed.turn_index, step_index=99, status="running")
        mem.close()

        mem2 = SQLiteMemory(db_path=db, session_id=sid)
        j2 = StepJournal(mem2._ensure_conn(), mem2.session_id)
        core = AgentCore(
            client=MockClient([make_mock_response(content="ok")]),
            memory=mem2,
            tools=make_tool_registry(),
            journal=j2,
            max_iterations=3,
        )
        await core.resume()
        # The stale marker was flipped to 'interrupted'; the loop only touched
        # step 0 (the resumed step), so step 99 keeps its interrupted status.
        conn = sqlite3.connect(db)
        row = conn.execute(
            "SELECT status FROM steps WHERE session_id=? AND turn_index=1 AND step_index=99",
            (sid,),
        ).fetchone()
        conn.close()
        assert row is not None and row[0] == "interrupted"
        # And nothing is left in a dangling 'running' state.
        assert j2.list_open_running() == []

    async def test_resume_does_not_add_user_message(self, tmp_path):
        db = str(tmp_path / "t.db")
        sid = "S4"
        mem = SQLiteMemory(db_path=db, session_id=sid)
        mem.add_user_message("original question")
        mem.close()
        n_before = len(SQLiteMemory.get_session_messages(db, sid))

        mem2 = SQLiteMemory(db_path=db, session_id=sid)
        j2 = StepJournal(mem2._ensure_conn(), mem2.session_id)
        core = AgentCore(
            client=MockClient([make_mock_response(content="answered")]),
            memory=mem2,
            tools=make_tool_registry(),
            journal=j2,
            max_iterations=3,
        )
        await core.resume()
        # No new user message should appear (only the LLM's response was added).
        msgs = SQLiteMemory.get_session_messages(db, sid)
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        assert len(user_msgs) == 1
        assert len(msgs) == n_before + 1


class TestResumeViaFacade:
    def test_from_config_resume_session_rehydrates(self, tmp_path):
        from koboi.config import Config

        db = str(tmp_path / "mem.db")
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            f"""
agent:
  name: resume-agent
  system_prompt: helpful
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: test
  base_url: http://localhost:8080/v1
memory:
  backend: sqlite
  db_path: {db}
  session_id: seed-session
"""
        )
        # Seed a conversation under seed-session.
        mem = SQLiteMemory(db_path=db, session_id="seed-session")
        mem.add_user_message("persisted message")
        mem.close()

        # Build an agent pointed at the same DB, resuming seed-session.
        config = Config.from_yaml(cfg_path)
        from koboi.facade import KoboiAgent

        agent = KoboiAgent._from_config(config, resume_session="seed-session")
        assert agent.core.memory.session_id == "seed-session"
        msgs = agent.core.memory.get_messages()
        assert any("persisted message" in (str(m.get("content"))) for m in msgs)
        # The journal should be wired (sqlite backend) and inherit the session.
        assert agent.core.journal is not None
        assert agent.core.journal._session_id == "seed-session"

    async def test_resume_raises_in_orchestration_mode(self, tmp_path):
        from koboi.facade import KoboiAgent
        from koboi.exceptions import AgentError

        # A facade with an orchestrator (core is None) must refuse resume.
        agent = KoboiAgent(core=None, orchestrator=object())
        with pytest.raises(AgentError):
            await agent.resume()
