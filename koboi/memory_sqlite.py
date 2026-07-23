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


def ensure_steps_table(conn: sqlite3.Connection) -> None:
    """Create the ``steps`` table (P2-A step journal) with graph-durability columns.

    Idempotent: ``CREATE TABLE IF NOT EXISTS`` defines the columns for new DBs;
    guarded ``ALTER TABLE ADD COLUMN`` adds ``graph_run_id``/``node_id`` to DBs whose
    ``steps`` table predates #3. Shared by :class:`SQLiteMemory` (per-iteration rows)
    and :class:`DagScheduler` (graph-plan rows) so graph-level durability data lands
    in one queryable place (consumed by the Phase-3 graph-cursor resume).
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            turn_index INTEGER NOT NULL,
            step_index INTEGER NOT NULL,
            status TEXT NOT NULL,
            llm_prompt_tokens INTEGER,
            llm_completion_tokens INTEGER,
            tool_call_count INTEGER DEFAULT 0,
            tool_calls_json TEXT,
            is_terminal INTEGER DEFAULT 0,
            error TEXT,
            graph_run_id TEXT,
            node_id TEXT,
            trace_id TEXT,
            checkpoint_sha TEXT,
            created_at REAL DEFAULT (julianday('now'))
        )
    """)
    # Additive columns for pre-existing 'steps' tables (pre-#3); no-op if present.
    for _stmt in (
        "ALTER TABLE steps ADD COLUMN graph_run_id TEXT",
        "ALTER TABLE steps ADD COLUMN node_id TEXT",
        "ALTER TABLE steps ADD COLUMN trace_id TEXT",  # P4: W3C trace-id (cross-instance correlation)
        "ALTER TABLE steps ADD COLUMN checkpoint_sha TEXT",  # Wave 2: shadow-repo checkpoint per step
    ):
        try:
            conn.execute(_stmt)
        except sqlite3.OperationalError:
            pass
    conn.execute("CREATE INDEX IF NOT EXISTS idx_steps_session ON steps(session_id, turn_index, step_index)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_steps_graph ON steps(graph_run_id, node_id)")


def ensure_tasks_table(conn: sqlite3.Connection) -> None:
    """Create the session-scoped ``tasks`` table (#6) for durable task state.

    Lets TaskManager state survive ``--resume``: created on first use, additive
    (CREATE IF NOT EXISTS). Columns mirror the in-memory Task dataclass.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            subject TEXT,
            description TEXT,
            status TEXT,
            blocked_by_json TEXT,
            task_order INTEGER,
            created_at REAL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id)")


def ensure_research_context_table(conn: sqlite3.Connection) -> None:
    """Create the ``research_context`` table (W2) for durable deep-research state.

    One row per ``graph_run_id`` holding the journaled ``ResearchContext`` JSON, so a
    ``--resume`` rehydrates sub-questions / SourceStore / coverage_map / budget and
    continues at the recorded depth (no re-bill across completed depth rounds). Mirrors
    ``ensure_tasks_table`` (CREATE IF NOT EXISTS + index).
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS research_context (
            graph_run_id TEXT PRIMARY KEY,
            context_json TEXT NOT NULL,
            updated_at REAL,
            session_id TEXT,
            depth INTEGER
        )
        """
    )
    # Additive column for pre-existing DBs created before session scoping (so
    # GET /v1/sessions/{id} can map a session to its deep-research context).
    _migrate_research_session_id(conn)
    # Additive column for report-wins precedence (a multi-step report survives a later
    # trivial direct-answer in the same session). Backfilled from context_json.
    _migrate_research_depth(conn)


def _migrate_research_session_id(conn: sqlite3.Connection) -> None:
    """Add the ``session_id`` column to an older ``research_context`` table if missing."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(research_context)").fetchall()}
    if "session_id" not in cols:
        conn.execute("ALTER TABLE research_context ADD COLUMN session_id TEXT")
        conn.commit()


def _migrate_research_depth(conn: sqlite3.Connection) -> None:
    """Add the ``depth`` column to an older ``research_context`` table + backfill from context_json.

    ``depth`` is denormalized from ``ResearchContext.depth`` (0 = direct-answer, >=1 = multi-step)
    so ``load_research_context_for_session`` can rank a richer report above a later trivial answer
    without parsing JSON in SQL.
    """
    cols = {r[1] for r in conn.execute("PRAGMA table_info(research_context)").fetchall()}
    if "depth" not in cols:
        conn.execute("ALTER TABLE research_context ADD COLUMN depth INTEGER")
        for graph_run_id, context_json in conn.execute(
            "SELECT graph_run_id, context_json FROM research_context"
        ).fetchall():
            try:
                depth = int(json.loads(context_json).get("depth", 0))
            except (ValueError, TypeError):
                depth = 0
            conn.execute("UPDATE research_context SET depth=? WHERE graph_run_id=?", (depth, graph_run_id))
        conn.commit()


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
        retention_cap: int | None = None,
        owner: str | None = None,
    ):
        super().__init__(logger=logger, system_prompt=system_prompt)
        self._db_path = db_path
        self._session_id = session_id or uuid4().hex
        self._conn: sqlite3.Connection | None = None
        # Issue #4b: optional cap on stored message rows (oldest pruned). None =
        # unbounded (default). Set via memory.retention.max_messages.
        self._retention_cap = retention_cap
        # Issue #2: optional tenant/owner tag on stored rows (schema prep for
        # multi-tenancy; real isolation lands with externalized state). None =
        # no tagging (today's behavior).
        self._owner = owner
        self._init_db()
        self._load_from_db()
        self._apply_retention()

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
                owner TEXT,
                created_at REAL DEFAULT (julianday('now'))
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                owner TEXT,
                created_at REAL DEFAULT (julianday('now')),
                updated_at REAL DEFAULT (julianday('now')),
                message_count INTEGER DEFAULT 0,
                model TEXT,
                agent_name TEXT,
                tags TEXT
            )
        """)
        # P2-A: step journal (one row per loop iteration). #3 adds graph-durability
        # columns (graph_run_id, node_id); shared with DagScheduler graph-plan writes.
        ensure_steps_table(conn)
        # Issue #4a: generic per-session key/value store for cross-restart state
        # (e.g. the sliding_window summary). Additive -- safe on existing DBs.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_meta (
                session_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                PRIMARY KEY (session_id, key)
            )
        """
        )
        # Issue #2: add `owner` column to pre-existing DBs (additive; new DBs get
        # it from the CREATE above). NULL default = today's global behavior.
        self._migrate_add_owner(conn)

    @staticmethod
    def _migrate_add_owner(conn: sqlite3.Connection) -> None:
        msg_cols = {r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()}
        if "owner" not in msg_cols:
            conn.execute("ALTER TABLE messages ADD COLUMN owner TEXT")
        sess_cols = {r[1] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()}
        if "owner" not in sess_cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN owner TEXT")
        conn.commit()

    @staticmethod
    def _ensure_schema_on(conn: sqlite3.Connection) -> None:
        """Ensure the full schema exists on a raw/short-lived connection.

        The static helpers (list/delete/fork/get) open via ``_open_conn``, which
        skips ``_init_db``. They must self-heal the schema (steps/tasks/session_meta
        tables + the owner column) before referencing it, or they crash on DBs
        created before these features existed (every pre-#23 deployment).
        """
        ensure_steps_table(conn)
        ensure_tasks_table(conn)
        ensure_research_context_table(conn)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS session_meta ("
            "session_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT, "
            "PRIMARY KEY (session_id, key))"
        )
        SQLiteMemory._migrate_add_owner(conn)

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
            "INSERT INTO messages (session_id, role, content, tool_calls_json, tool_call_id, label, owner) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                self._session_id,
                role,
                stored_content,
                json.dumps(kwargs.get("tool_calls")) if kwargs.get("tool_calls") else None,
                kwargs.get("tool_call_id"),
                kwargs.get("label", ""),
                self._owner,
            ),
        )
        conn.execute(
            "UPDATE sessions SET updated_at = julianday('now'), "
            "message_count = (SELECT COUNT(*) FROM messages WHERE session_id = ?) "
            "WHERE session_id = ?",
            (self._session_id, self._session_id),
        )
        conn.commit()
        self._apply_retention()

    def _apply_retention(self) -> None:
        """Issue #4b: prune oldest message rows beyond the retention cap.

        Keeps the in-memory list and the DB rows in lockstep (both ordered by
        insertion). No-op when no cap is set.
        """
        cap = self._retention_cap
        if not cap or cap <= 0:
            return
        conn = self._ensure_conn()
        row = conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (self._session_id,)).fetchone()
        count = int(row[0]) if row and row[0] is not None else 0
        if count <= cap:
            return
        excess = count - cap
        conn.execute(
            "DELETE FROM messages WHERE id IN (SELECT id FROM messages WHERE session_id = ? ORDER BY id LIMIT ?)",
            (self._session_id, excess),
        )
        conn.commit()
        # Drop the same count from the in-memory head (matches DB insertion order).
        if 0 < excess <= len(self._messages):
            del self._messages[:excess]

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

    # -- Per-session metadata (issue #4a) -------------------------------------

    def get_meta(self, key: str) -> str | None:
        """Read a per-session metadata value, or None if unset."""
        conn = self._ensure_conn()
        row = conn.execute(
            "SELECT value FROM session_meta WHERE session_id = ? AND key = ?",
            (self._session_id, key),
        ).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        """Write (upsert) a per-session metadata value."""
        conn = self._ensure_conn()
        conn.execute(
            "INSERT INTO session_meta (session_id, key, value) VALUES (?, ?, ?) "
            "ON CONFLICT(session_id, key) DO UPDATE SET value = excluded.value",
            (self._session_id, key, value),
        )
        conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def session_id(self) -> str:
        """Current session ID."""
        return self._session_id

    @property
    def owner(self) -> str | None:
        """Tenant/owner tag stamped on this session's rows (None = untagged)."""
        return self._owner

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
        # Issue #2: stamp owner on INSERT only (a resumed session keeps its
        # original owner; the ON CONFLICT branch does not overwrite it).
        conn.execute(
            "INSERT INTO sessions (session_id, agent_name, model, owner, updated_at) "
            "VALUES (?, ?, ?, ?, julianday('now')) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "updated_at = julianday('now'), "
            "message_count = (SELECT COUNT(*) FROM messages WHERE session_id = ?)",
            (self._session_id, agent_name, model, self._owner, self._session_id),
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
    def list_sessions(db_path: str, limit: int = 50, owner: str | None = None) -> list[dict]:
        """List sessions with metadata, most recent first. Optional owner filter (issue #2)."""
        conn = SQLiteMemory._open_conn(db_path)
        SQLiteMemory._ensure_schema_on(conn)
        conn.row_factory = sqlite3.Row
        select = (
            "SELECT s.session_id, s.title, s.created_at, s.updated_at, "
            "s.message_count, s.model, s.agent_name, s.tags, s.owner, "
            "(SELECT content FROM messages WHERE session_id = s.session_id "
            "AND role = 'user' ORDER BY id LIMIT 1) as first_message "
            "FROM sessions s"
        )
        if owner is not None:
            rows = conn.execute(
                select + " WHERE s.owner = ? ORDER BY s.updated_at DESC LIMIT ?",
                (owner, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                select + " ORDER BY s.updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    @staticmethod
    def delete_session(db_path: str, session_id: str) -> bool:
        """Delete a session and all its rows (messages, steps, session_meta, sessions, tasks)."""
        conn = SQLiteMemory._open_conn(db_path)
        try:
            SQLiteMemory._ensure_schema_on(conn)
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM steps WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM session_meta WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM tasks WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            deleted = conn.total_changes > 0
        finally:
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
        SQLiteMemory._ensure_schema_on(conn)
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls_json, tool_call_id, label, owner) "
            "SELECT ?, role, content, tool_calls_json, tool_call_id, label, owner "
            "FROM messages WHERE session_id = ?",
            (new_session_id, source_session_id),
        )
        conn.execute(
            "INSERT INTO sessions (session_id, title, message_count, owner) "
            "SELECT ?, 'Fork of ' || COALESCE(title, ?), "
            "(SELECT COUNT(*) FROM messages WHERE session_id = ?), owner "
            "FROM sessions WHERE session_id = ?",
            (new_session_id, source_session_id, new_session_id, source_session_id),
        )
        conn.commit()
        conn.close()
        return new_session_id
