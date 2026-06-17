"""Tests for SQLiteMemory static methods and uncovered paths."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from koboi.memory_sqlite import SQLiteMemory


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_memory.db")


@pytest.fixture
def populated_db(db_path):
    """Create a DB with sessions and messages for testing static methods."""
    mem = SQLiteMemory(db_path=db_path, session_id="sess1")
    mem.ensure_session_record(agent_name="test-agent", model="gpt-4o")
    mem.update_session_title("Test Session")
    mem.add_user_message("Hello")
    mem.add_assistant_message("Hi there!")
    mem.add_user_message("How are you?")
    mem.add_assistant_message("I'm fine.")
    mem.close()

    mem2 = SQLiteMemory(db_path=db_path, session_id="sess2")
    mem2.ensure_session_record(agent_name="other-agent", model="claude")
    mem2.add_user_message("Other message")
    mem2.close()
    return db_path


class TestListSessions:
    def test_list_sessions_returns_all(self, populated_db):
        sessions = SQLiteMemory.list_sessions(populated_db)
        assert len(sessions) == 2

    def test_list_sessions_has_metadata(self, populated_db):
        sessions = SQLiteMemory.list_sessions(populated_db)
        s = sessions[0]  # Most recent
        assert "session_id" in s
        assert "title" in s
        assert "message_count" in s

    def test_list_sessions_limit(self, populated_db):
        sessions = SQLiteMemory.list_sessions(populated_db, limit=1)
        assert len(sessions) == 1

    def test_list_sessions_empty(self, db_path):
        SQLiteMemory(db_path=db_path)  # Creates schema
        sessions = SQLiteMemory.list_sessions(db_path)
        assert len(sessions) == 0


class TestDeleteSession:
    def test_delete_session(self, populated_db):
        result = SQLiteMemory.delete_session(populated_db, "sess1")
        assert result is True
        sessions = SQLiteMemory.list_sessions(populated_db)
        assert len(sessions) == 1

    def test_delete_nonexistent(self, populated_db):
        result = SQLiteMemory.delete_session(populated_db, "nonexistent")
        # No error, just returns whether any rows were affected


class TestGetSessionMessages:
    def test_get_messages(self, populated_db):
        messages = SQLiteMemory.get_session_messages(populated_db, "sess1")
        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    def test_get_messages_with_tool_calls(self, db_path):
        mem = SQLiteMemory(db_path=db_path, session_id="s1")
        mem.add_assistant_message("calling tool", tool_calls=[{"id": "tc1", "function": {"name": "calc"}}])
        mem.add_tool_result("tc1", "42")
        mem.close()
        messages = SQLiteMemory.get_session_messages(db_path, "s1")
        assert any(m.get("tool_calls") for m in messages)
        assert any(m.get("tool_call_id") == "tc1" for m in messages)

    def test_get_messages_empty(self, db_path):
        SQLiteMemory(db_path=db_path)
        messages = SQLiteMemory.get_session_messages(db_path, "nonexistent")
        assert messages == []


class TestForkSession:
    def test_fork_session(self, populated_db):
        new_id = SQLiteMemory.fork_session(populated_db, "sess1", "forked_sess")
        assert new_id == "forked_sess"
        messages = SQLiteMemory.get_session_messages(populated_db, "forked_sess")
        original = SQLiteMemory.get_session_messages(populated_db, "sess1")
        assert len(messages) == len(original)

    def test_fork_creates_session_record(self, populated_db):
        SQLiteMemory.fork_session(populated_db, "sess1", "forked")
        sessions = SQLiteMemory.list_sessions(populated_db)
        forked = [s for s in sessions if s["session_id"] == "forked"]
        assert len(forked) == 1


class TestEnsureSessionRecord:
    def test_creates_new(self, db_path):
        mem = SQLiteMemory(db_path=db_path, session_id="s1")
        mem.ensure_session_record(agent_name="agent", model="gpt-4o")
        sessions = SQLiteMemory.list_sessions(db_path)
        assert len(sessions) == 1
        mem.close()

    def test_upsert_existing(self, db_path):
        mem = SQLiteMemory(db_path=db_path, session_id="s1")
        mem.ensure_session_record(agent_name="agent1", model="gpt-4o")
        mem.add_user_message("hello")
        mem.ensure_session_record(agent_name="agent1", model="gpt-4o")
        sessions = SQLiteMemory.list_sessions(db_path)
        assert len(sessions) == 1
        mem.close()


class TestUpdateSessionTitle:
    def test_updates_title(self, db_path):
        mem = SQLiteMemory(db_path=db_path, session_id="s1")
        mem.ensure_session_record()
        mem.update_session_title("My Session")
        sessions = SQLiteMemory.list_sessions(db_path)
        assert sessions[0]["title"] == "My Session"
        mem.close()


class TestMultimodalContent:
    def test_stores_list_content(self, db_path):
        mem = SQLiteMemory(db_path=db_path, session_id="s1")
        content = [{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {"url": "data:..."}}]
        mem.add_user_message(content)
        mem.close()
        messages = SQLiteMemory.get_session_messages(db_path, "s1")
        # The content should be a JSON string or the deserialized list
        assert messages[0]["role"] == "user"

    def test_loads_json_content(self, db_path):
        mem = SQLiteMemory(db_path=db_path, session_id="s1")
        mem.add_user_message([{"type": "text", "text": "hello"}])
        mem.close()
        mem2 = SQLiteMemory(db_path=db_path, session_id="s1")
        msgs = mem2.get_messages()
        assert isinstance(msgs[0]["content"], list)
        mem2.close()


class TestContextMessage:
    def test_add_context_message(self, db_path):
        mem = SQLiteMemory(db_path=db_path, session_id="s1")
        mem.add_context_message("RAG context", label="rag")
        mem.close()
        messages = SQLiteMemory.get_session_messages(db_path, "s1")
        assert messages[0]["role"] == "system"
