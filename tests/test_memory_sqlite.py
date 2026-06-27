"""Tests for koboi.memory_sqlite module."""

from __future__ import annotations

import sqlite3
import threading


from koboi.memory_sqlite import SQLiteMemory


class TestSQLiteMemory:
    def test_db_init_creates_table_with_correct_schema(self, tmp_path):
        """Test that DB initialization creates messages table with correct schema."""
        db_path = tmp_path / "test.db"
        mem = SQLiteMemory(db_path=str(db_path), session_id="test_session")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
        result = cursor.fetchone()
        assert result is not None

        # Check schema
        cursor.execute("PRAGMA table_info(messages)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        assert columns["id"] == "INTEGER"
        assert columns["session_id"] == "TEXT"
        assert columns["role"] == "TEXT"
        assert columns["content"] == "TEXT"
        assert columns["tool_calls_json"] == "TEXT"
        assert columns["tool_call_id"] == "TEXT"
        assert columns["label"] == "TEXT"
        assert columns["created_at"] == "REAL"

        # Check index exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_messages_session'")
        result = cursor.fetchone()
        assert result is not None

        conn.close()
        mem.close()

    def test_add_user_message_persists_and_loads_back(self, tmp_path):
        """Test that user messages are persisted and can be loaded back."""
        db_path = tmp_path / "test.db"
        session_id = "test_session"

        # Create memory and add message
        mem1 = SQLiteMemory(db_path=str(db_path), session_id=session_id)
        mem1.add_user_message("Hello, world!")
        mem1.close()

        # Create new instance with same session_id
        mem2 = SQLiteMemory(db_path=str(db_path), session_id=session_id)
        messages = mem2.get_messages()

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, world!"
        mem2.close()

    def test_add_assistant_message_without_tool_calls(self, tmp_path):
        """Test assistant messages without tool calls are persisted correctly."""
        db_path = tmp_path / "test.db"
        session_id = "test_session"

        mem1 = SQLiteMemory(db_path=str(db_path), session_id=session_id)
        mem1.add_assistant_message("Response text")
        mem1.close()

        mem2 = SQLiteMemory(db_path=str(db_path), session_id=session_id)
        messages = mem2.get_messages()

        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "Response text"
        assert "tool_calls" not in messages[0]
        mem2.close()

    def test_add_assistant_message_with_tool_calls(self, tmp_path):
        """Test assistant messages with tool calls are persisted correctly."""
        db_path = tmp_path / "test.db"
        session_id = "test_session"

        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "arguments": '{"arg": "value"}',
                },
            }
        ]

        mem1 = SQLiteMemory(db_path=str(db_path), session_id=session_id)
        mem1.add_assistant_message("I'll use a tool", tool_calls=tool_calls)
        mem1.close()

        mem2 = SQLiteMemory(db_path=str(db_path), session_id=session_id)
        messages = mem2.get_messages()

        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "I'll use a tool"
        assert messages[0]["tool_calls"] == tool_calls
        mem2.close()

    def test_add_tool_result_persists_tool_call_id(self, tmp_path):
        """Test that tool results are persisted with correct tool_call_id."""
        db_path = tmp_path / "test.db"
        session_id = "test_session"

        mem1 = SQLiteMemory(db_path=str(db_path), session_id=session_id)
        mem1.add_tool_result("call_123", "Tool result content")
        mem1.close()

        mem2 = SQLiteMemory(db_path=str(db_path), session_id=session_id)
        messages = mem2.get_messages()

        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert messages[0]["tool_call_id"] == "call_123"
        assert messages[0]["content"] == "Tool result content"
        mem2.close()

    def test_add_context_message_persists_with_label(self, tmp_path):
        """Test that context messages are persisted with label."""
        db_path = tmp_path / "test.db"
        session_id = "test_session"

        mem1 = SQLiteMemory(db_path=str(db_path), session_id=session_id)
        mem1.add_context_message("Context info", label="test_label")
        mem1.close()

        mem2 = SQLiteMemory(db_path=str(db_path), session_id=session_id)
        messages = mem2.get_messages()

        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Context info"
        mem2.close()

    def test_load_from_db_restores_all_message_types(self, tmp_path):
        """Test that _load_from_db restores all message types correctly."""
        db_path = tmp_path / "test.db"
        session_id = "test_session"

        tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "tool", "arguments": "{}"}}]

        mem1 = SQLiteMemory(db_path=str(db_path), session_id=session_id)
        mem1.add_user_message("User message")
        mem1.add_assistant_message("Assistant response", tool_calls=tool_calls)
        mem1.add_tool_result("call_1", "Tool output")
        mem1.add_context_message("System context", label="ctx")
        mem1.close()

        mem2 = SQLiteMemory(db_path=str(db_path), session_id=session_id)
        messages = mem2.get_messages()

        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["tool_calls"] == tool_calls
        assert messages[2]["role"] == "tool"
        assert messages[3]["role"] == "system"
        mem2.close()

    def test_clear_removes_session_messages_from_db(self, tmp_path):
        """Test that clear() removes messages from the database."""
        db_path = tmp_path / "test.db"
        session_id = "test_session"

        mem = SQLiteMemory(db_path=str(db_path), session_id=session_id)
        mem.add_user_message("Message 1")
        mem.add_user_message("Message 2")

        # Verify in DB
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
        assert cursor.fetchone()[0] == 2
        conn.close()

        # Clear and verify
        mem.clear()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
        assert cursor.fetchone()[0] == 0
        conn.close()

        mem.close()

    def test_close_cleans_up_connection(self, tmp_path):
        """Test that close() properly closes the database connection."""
        db_path = tmp_path / "test.db"
        mem = SQLiteMemory(db_path=str(db_path), session_id="test_session")
        mem.add_user_message("Test")

        # Connection should be open
        assert mem._conn is not None

        mem.close()

        # Connection should be closed
        assert mem._conn is None

    def test_session_isolation_different_sessions_dont_interfere(self, tmp_path):
        """Test that different session_ids don't interfere with each other."""
        db_path = tmp_path / "test.db"

        mem1 = SQLiteMemory(db_path=str(db_path), session_id="session_1")
        mem1.add_user_message("Session 1 message")
        mem1.close()

        mem2 = SQLiteMemory(db_path=str(db_path), session_id="session_2")
        mem2.add_user_message("Session 2 message")
        mem2.close()

        # Load session 1
        mem1_reload = SQLiteMemory(db_path=str(db_path), session_id="session_1")
        messages1 = mem1_reload.get_messages()
        assert len(messages1) == 1
        assert messages1[0]["content"] == "Session 1 message"
        mem1_reload.close()

        # Load session 2
        mem2_reload = SQLiteMemory(db_path=str(db_path), session_id="session_2")
        messages2 = mem2_reload.get_messages()
        assert len(messages2) == 1
        assert messages2[0]["content"] == "Session 2 message"
        mem2_reload.close()

    def test_system_prompt_is_not_persisted_to_db(self, tmp_path):
        """Test that system prompt is not stored in database."""
        db_path = tmp_path / "test.db"
        session_id = "test_session"

        mem1 = SQLiteMemory(db_path=str(db_path), session_id=session_id, system_prompt="System prompt")
        mem1.add_user_message("User message")
        mem1.close()

        # Check DB directly
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
        assert cursor.fetchone()[0] == 1  # Only user message, not system prompt
        conn.close()

        # But it should still be in get_messages
        mem2 = SQLiteMemory(db_path=str(db_path), session_id=session_id, system_prompt="System prompt")
        messages = mem2.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System prompt"
        mem2.close()

    def test_wal_mode_enabled(self, tmp_path):
        """Test that WAL mode is enabled for better concurrency."""
        db_path = tmp_path / "test.db"
        mem = SQLiteMemory(db_path=str(db_path), session_id="test_session")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        assert mode == "wal"
        conn.close()

        mem.close()

    def test_pragma_busy_timeout_set(self, tmp_path):
        """Test that busy_timeout is set to handle concurrent access."""
        db_path = tmp_path / "test.db"
        mem = SQLiteMemory(db_path=str(db_path), session_id="test_session")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA busy_timeout")
        timeout = cursor.fetchone()[0]
        assert timeout == 5000
        conn.close()

        mem.close()

    def test_custom_session_id(self, tmp_path):
        """Test that custom session_id is used correctly."""
        db_path = tmp_path / "test.db"
        custom_session = "my_custom_session_123"

        mem = SQLiteMemory(db_path=str(db_path), session_id=custom_session)
        assert mem._session_id == custom_session

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT session_id FROM messages LIMIT 1")
        session_id = cursor.fetchone()
        assert session_id is None  # No messages yet
        conn.close()

        mem.add_user_message("Test")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT session_id FROM messages LIMIT 1")
        session_id = cursor.fetchone()
        assert session_id[0] == custom_session
        conn.close()

        mem.close()

    def test_auto_generated_session_id(self, tmp_path):
        """Test that session_id is auto-generated if not provided."""
        db_path = tmp_path / "test.db"

        mem1 = SQLiteMemory(db_path=str(db_path))
        assert mem1._session_id is not None
        assert len(mem1._session_id) == 32  # UUID hex length

        mem2 = SQLiteMemory(db_path=str(db_path))
        assert mem2._session_id != mem1._session_id

        mem1.close()
        mem2.close()

    def test_len_returns_correct_count(self, tmp_path):
        """Test that __len__ returns correct message count (excluding system prompt)."""
        db_path = tmp_path / "test.db"
        mem = SQLiteMemory(db_path=str(db_path), session_id="test", system_prompt="System")

        assert len(mem) == 0
        mem.add_user_message("User")
        assert len(mem) == 1
        mem.add_assistant_message("Assistant")
        assert len(mem) == 2
        mem.add_tool_result("call_1", "Result")
        assert len(mem) == 3

        mem.close()

    def test_get_messages_includes_system_prompt(self, tmp_path):
        """Test that get_messages includes system prompt first."""
        db_path = tmp_path / "test.db"
        sys_prompt = "You are a helpful assistant."

        mem = SQLiteMemory(db_path=str(db_path), session_id="test", system_prompt=sys_prompt)
        mem.add_user_message("Hello")

        messages = mem.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == sys_prompt
        assert messages[1]["role"] == "user"

        mem.close()

    def test_empty_message_content_handling(self, tmp_path):
        """Test handling of None or empty content in various message types."""
        db_path = tmp_path / "test.db"

        mem = SQLiteMemory(db_path=str(db_path), session_id="test")
        mem.add_assistant_message(None)  # Content is None
        mem.add_assistant_message("", tool_calls=[{"id": "call_1"}])  # Empty content with tool calls

        messages = mem.get_messages()
        assert len(messages) == 2
        # When content is None, no content key is added
        assert "content" not in messages[0] or messages[0].get("content") == ""
        # With tool_calls, content is set to empty string
        assert messages[1].get("content") == ""
        assert "tool_calls" in messages[1]

        mem.close()

    def test_multiple_operations_same_instance(self, tmp_path):
        """Test that multiple operations on same instance work correctly."""
        db_path = tmp_path / "test.db"

        mem = SQLiteMemory(db_path=str(db_path), session_id="test")
        mem.add_user_message("Q1")
        mem.add_assistant_message("A1")
        mem.add_user_message("Q2")
        mem.add_assistant_message("A2")

        messages = mem.get_messages()
        assert len(messages) == 4
        assert messages[0]["content"] == "Q1"
        assert messages[1]["content"] == "A1"
        assert messages[2]["content"] == "Q2"
        assert messages[3]["content"] == "A2"

        mem.close()

    def test_persistence_across_reopen(self, tmp_path):
        """Test full persistence across closing and reopening."""
        db_path = tmp_path / "test.db"
        session_id = "persist_test"

        # First instance: create and populate
        mem1 = SQLiteMemory(db_path=str(db_path), session_id=session_id)
        mem1.add_user_message("User message 1")
        mem1.add_assistant_message("Assistant response 1")
        mem1.add_tool_result("call_1", "Tool result 1")
        mem1.close()

        # Second instance: verify data loaded
        mem2 = SQLiteMemory(db_path=str(db_path), session_id=session_id)
        messages = mem2.get_messages()
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "User message 1"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Assistant response 1"
        assert messages[2]["role"] == "tool"
        assert messages[2]["tool_call_id"] == "call_1"

        # Add more data
        mem2.add_user_message("User message 2")
        mem2.close()

        # Third instance: verify all data including new message
        mem3 = SQLiteMemory(db_path=str(db_path), session_id=session_id)
        messages = mem3.get_messages()
        assert len(messages) == 4
        assert messages[3]["content"] == "User message 2"
        mem3.close()

    def test_cross_thread_access_no_error(self, tmp_path):
        """Test that SQLiteMemory can be used from a different thread than it was created on."""
        db_path = tmp_path / "test.db"
        mem = SQLiteMemory(db_path=str(db_path), session_id="test")
        mem.add_user_message("Main thread message")

        errors: list[Exception] = []

        def worker():
            try:
                mem.add_user_message("Background thread message")
                mem.add_assistant_message("Background response")
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert errors == [], f"Cross-thread access raised: {errors}"
        messages = mem.get_messages()
        assert len(messages) == 3
        assert messages[1]["content"] == "Background thread message"
        assert messages[2]["content"] == "Background response"
        mem.close()
