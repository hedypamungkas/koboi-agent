"""koboi/memory_sqlite.py -- SQLite-backed conversation memory with async persistence."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from koboi.memory import ConversationMemory

if TYPE_CHECKING:
    from koboi.logger import AgentLogger


class SQLiteMemory(ConversationMemory):
    """ConversationMemory backed by SQLite. Persists sessions across restarts.

    Uses synchronous sqlite3 for writes (sub-millisecond for single rows).
    The in-memory list is the primary read source; SQLite is for durability.
    """

    def __init__(
        self,
        db_path: str = "koboi_memory.db",
        session_id: str | None = None,
        logger: AgentLogger | None = None,
        system_prompt: str | None = None,
    ):
        super().__init__(logger=logger, system_prompt=system_prompt)
        self._db_path = db_path
        self._session_id = session_id or uuid4().hex
        self._conn: sqlite3.Connection | None = None
        self._init_db()
        self._load_from_db()

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
        return self._conn

    @staticmethod
    def _open_conn(db_path: str) -> sqlite3.Connection:
        """Open a short-lived connection with WAL and busy_timeout."""
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_db(self) -> None:
        conn = self._ensure_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_calls_json TEXT,
                tool_call_id TEXT,
                label TEXT,
                created_at REAL DEFAULT (julianday('now'))
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                created_at REAL DEFAULT (julianday('now')),
                updated_at REAL DEFAULT (julianday('now')),
                message_count INTEGER DEFAULT 0,
                model TEXT,
                agent_name TEXT,
                tags TEXT
            )
        """)
        conn.commit()

    def _load_from_db(self) -> None:
        conn = self._ensure_conn()
        rows = conn.execute(
            "SELECT role, content, tool_calls_json, tool_call_id FROM messages WHERE session_id = ? ORDER BY id",
            (self._session_id,),
        ).fetchall()
        for role, content, tool_calls_json, tool_call_id in rows:
            if role == "user":
                # Deserialize JSON list content (multimodal messages)
                if content and content.startswith("["):
                    try:
                        user_content = json.loads(content)
                    except (json.JSONDecodeError, ValueError):
                        user_content = content or ""
                else:
                    user_content = content or ""
                self._messages.append({"role": "user", "content": user_content})
            elif role == "assistant":
                msg: dict = {"role": "assistant"}
                if content:
                    msg["content"] = content
                if tool_calls_json:
                    msg["tool_calls"] = json.loads(tool_calls_json)
                    msg["content"] = content or ""
                self._messages.append(msg)
            elif role == "tool":
                self._messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id or "",
                        "content": content or "",
                    }
                )
            elif role == "system":
                self._messages.append({"role": "system", "content": content or ""})

    def _persist(self, role: str, content: str | list | None, **kwargs) -> None:
        conn = self._ensure_conn()
        # Serialize list content (multimodal) as JSON for SQLite storage
        stored_content = json.dumps(content) if isinstance(content, list) else content
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls_json, tool_call_id, label) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                self._session_id,
                role,
                stored_content,
                json.dumps(kwargs.get("tool_calls")) if kwargs.get("tool_calls") else None,
                kwargs.get("tool_call_id"),
                kwargs.get("label", ""),
            ),
        )
        conn.execute(
            "UPDATE sessions SET updated_at = julianday('now'), "
            "message_count = (SELECT COUNT(*) FROM messages WHERE session_id = ?) "
            "WHERE session_id = ?",
            (self._session_id, self._session_id),
        )
        conn.commit()

    def add_user_message(self, content: str | list) -> None:
        super().add_user_message(content)
        self._persist("user", content)

    def add_assistant_message(self, content: str | None, tool_calls: list[dict] | None = None) -> None:
        super().add_assistant_message(content, tool_calls)
        self._persist("assistant", content, tool_calls=tool_calls)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        super().add_tool_result(tool_call_id, content)
        self._persist("tool", content, tool_call_id=tool_call_id)

    def add_context_message(self, content: str, label: str = "") -> None:
        super().add_context_message(content, label)
        self._persist("system", content, label=label)

    def clear(self) -> None:
        conn = self._ensure_conn()
        conn.execute("DELETE FROM messages WHERE session_id = ?", (self._session_id,))
        conn.commit()
        super().clear()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def session_id(self) -> str:
        """Current session ID."""
        return self._session_id

    @property
    def db_path(self) -> str:
        """Path to the SQLite database file."""
        return self._db_path

    def switch_session(self, session_id: str) -> None:
        """Switch to a different session, reloading messages from DB."""
        self._session_id = session_id
        self._load_from_db()

    def fork_and_switch(self, new_session_id: str | None = None) -> str:
        """Fork the current session and switch to the new one.

        Returns the new session ID.
        """
        from uuid import uuid4

        new_id = new_session_id or uuid4().hex
        self.fork_session(self._db_path, self._session_id, new_id)
        self._session_id = new_id
        self._load_from_db()
        return new_id

    # -- Session metadata API (Phase 5) -----------------------------------------

    def ensure_session_record(self, agent_name: str = "", model: str = "") -> None:
        """Create or update the sessions metadata row."""
        conn = self._ensure_conn()
        conn.execute(
            "INSERT INTO sessions (session_id, agent_name, model, updated_at) "
            "VALUES (?, ?, ?, julianday('now')) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "updated_at = julianday('now'), "
            "message_count = (SELECT COUNT(*) FROM messages WHERE session_id = ?)",
            (self._session_id, agent_name, model, self._session_id),
        )
        conn.commit()

    def update_session_title(self, title: str) -> None:
        """Set a human-readable title for the session."""
        conn = self._ensure_conn()
        conn.execute(
            "UPDATE sessions SET title = ? WHERE session_id = ?",
            (title, self._session_id),
        )
        conn.commit()

    @staticmethod
    def list_sessions(db_path: str, limit: int = 50) -> list[dict]:
        """List all sessions with metadata, most recent first."""
        conn = SQLiteMemory._open_conn(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT s.session_id, s.title, s.created_at, s.updated_at, "
            "s.message_count, s.model, s.agent_name, s.tags, "
            "(SELECT content FROM messages WHERE session_id = s.session_id "
            "AND role = 'user' ORDER BY id LIMIT 1) as first_message "
            "FROM sessions s ORDER BY s.updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    @staticmethod
    def delete_session(db_path: str, session_id: str) -> bool:
        """Delete a session and all its messages."""
        conn = SQLiteMemory._open_conn(db_path)
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        deleted = conn.total_changes > 0
        conn.close()
        return deleted

    @staticmethod
    def get_session_messages(db_path: str, session_id: str) -> list[dict]:
        """Load messages for any session (not just the current one)."""
        conn = SQLiteMemory._open_conn(db_path)
        rows = conn.execute(
            "SELECT role, content, tool_calls_json, tool_call_id FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        conn.close()
        messages = []
        for role, content, tool_calls_json, tool_call_id in rows:
            msg: dict = {"role": role, "content": content or ""}
            if tool_calls_json:
                msg["tool_calls"] = json.loads(tool_calls_json)
            if tool_call_id:
                msg["tool_call_id"] = tool_call_id
            messages.append(msg)
        return messages

    @staticmethod
    def fork_session(
        db_path: str,
        source_session_id: str,
        new_session_id: str,
    ) -> str:
        """Fork a session: copy all messages into a new session."""
        conn = SQLiteMemory._open_conn(db_path)
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls_json, tool_call_id, label) "
            "SELECT ?, role, content, tool_calls_json, tool_call_id, label "
            "FROM messages WHERE session_id = ?",
            (new_session_id, source_session_id),
        )
        conn.execute(
            "INSERT INTO sessions (session_id, title, message_count) "
            "SELECT ?, 'Fork of ' || COALESCE(title, ?), "
            "(SELECT COUNT(*) FROM messages WHERE session_id = ?) "
            "FROM sessions WHERE session_id = ?",
            (new_session_id, source_session_id, new_session_id, source_session_id),
        )
        conn.commit()
        conn.close()
        return new_session_id
