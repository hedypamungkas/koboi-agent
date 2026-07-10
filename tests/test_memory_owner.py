"""Issue #2: owner/tenant column on the memory schema (additive migration)."""

from __future__ import annotations

import sqlite3

from koboi.memory_sqlite import SQLiteMemory

# A faithful pre-#2 schema (messages/sessions WITHOUT the owner column).
_OLD_MESSAGES = """
    CREATE TABLE messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT,
        tool_calls_json TEXT,
        tool_call_id TEXT,
        label TEXT,
        created_at REAL DEFAULT (julianday('now'))
    )
"""
_OLD_SESSIONS = """
    CREATE TABLE sessions (
        session_id TEXT PRIMARY KEY,
        title TEXT,
        created_at REAL DEFAULT (julianday('now')),
        updated_at REAL DEFAULT (julianday('now')),
        message_count INTEGER DEFAULT 0,
        model TEXT,
        agent_name TEXT,
        tags TEXT
    )
"""


def _make_old_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(_OLD_MESSAGES)
    conn.execute(_OLD_SESSIONS)
    conn.execute("INSERT INTO messages (session_id, role, content) VALUES ('S', 'user', 'old')")
    conn.execute("INSERT INTO sessions (session_id) VALUES ('S')")
    conn.commit()
    conn.close()


class TestOwnerColumn:
    def test_migration_adds_owner_to_existing_db(self, tmp_path):
        db = str(tmp_path / "old.db")
        _make_old_db(db)
        # Opening with an owner triggers the additive migration (no crash).
        mem = SQLiteMemory(db_path=db, session_id="S", owner="alice")
        cols = {r[1] for r in mem._ensure_conn().execute("PRAGMA table_info(messages)").fetchall()}
        sess_cols = {r[1] for r in mem._ensure_conn().execute("PRAGMA table_info(sessions)").fetchall()}
        assert "owner" in cols
        assert "owner" in sess_cols
        # old row keeps NULL owner; new rows are stamped.
        mem.add_user_message("new")
        row = mem._ensure_conn().execute(
            "SELECT owner FROM messages WHERE content='new'"
        ).fetchone()
        assert row[0] == "alice"

    def test_migration_is_idempotent(self, tmp_path):
        db = str(tmp_path / "idem.db")
        _make_old_db(db)
        conn = sqlite3.connect(db)
        SQLiteMemory._migrate_add_owner(conn)
        SQLiteMemory._migrate_add_owner(conn)  # second run must not error
        conn.close()

    def test_owner_stamped_on_insert_and_session(self, tmp_path):
        db = str(tmp_path / "own.db")
        mem = SQLiteMemory(db_path=db, session_id="S1", owner="bob")
        mem.add_user_message("hi")
        mem.ensure_session_record(agent_name="a", model="m")
        mrow = mem._ensure_conn().execute(
            "SELECT owner FROM messages WHERE session_id='S1'"
        ).fetchone()
        srow = mem._ensure_conn().execute(
            "SELECT owner FROM sessions WHERE session_id='S1'"
        ).fetchone()
        assert mrow[0] == "bob"
        assert srow[0] == "bob"
        assert mem.owner == "bob"

    def test_default_owner_is_none(self, tmp_path):
        mem = SQLiteMemory(db_path=str(tmp_path / "n.db"), session_id="S")
        mem.add_user_message("hi")
        row = mem._ensure_conn().execute(
            "SELECT owner FROM messages WHERE session_id='S'"
        ).fetchone()
        assert row[0] is None
        assert mem.owner is None

    def test_list_sessions_owner_filter(self, tmp_path):
        db = str(tmp_path / "ls.db")
        a = SQLiteMemory(db_path=db, session_id="A", owner="alice")
        a.ensure_session_record(agent_name="ag", model="m")
        b = SQLiteMemory(db_path=db, session_id="B", owner="bob")
        b.ensure_session_record(agent_name="ag", model="m")
        all_sessions = SQLiteMemory.list_sessions(db)
        assert len(all_sessions) == 2
        alice_only = SQLiteMemory.list_sessions(db, owner="alice")
        assert len(alice_only) == 1
        assert alice_only[0]["session_id"] == "A"
        assert alice_only[0]["owner"] == "alice"

    def test_fork_copies_owner(self, tmp_path):
        # I1: fork_session must carry the owner column through (was dropped -> NULL).
        db = str(tmp_path / "fork.db")
        src = SQLiteMemory(db_path=db, session_id="SRC", owner="alice")
        src.add_user_message("hello")
        src.ensure_session_record(agent_name="a", model="m")
        src.close()
        SQLiteMemory.fork_session(db, "SRC", "DST")
        forked = SQLiteMemory.list_sessions(db, owner="alice")
        assert any(r["session_id"] == "DST" for r in forked)  # owner preserved
        conn = sqlite3.connect(db)
        msg_owner = conn.execute("SELECT owner FROM messages WHERE session_id='DST'").fetchone()
        sess_owner = conn.execute("SELECT owner FROM sessions WHERE session_id='DST'").fetchone()
        conn.close()
        assert msg_owner[0] == "alice"
        assert sess_owner[0] == "alice"

    def test_list_and_delete_self_heal_legacy_db(self, tmp_path):
        # C3: list_sessions/delete_session must not crash on a pre-#23 DB that
        # lacks the owner column + the session_meta table. _make_old_db builds a
        # faithful pre-owner schema (full messages/sessions columns, no owner,
        # no session_meta/steps/tasks).
        db = str(tmp_path / "legacy.db")
        _make_old_db(db)
        rows = SQLiteMemory.list_sessions(db)  # self-heals owner col + tables
        assert any(r["session_id"] == "S" for r in rows)
        assert SQLiteMemory.delete_session(db, "S") is True  # self-heals session_meta
        assert SQLiteMemory.list_sessions(db) == []
