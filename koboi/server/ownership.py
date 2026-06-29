"""koboi/server/ownership -- session_id → owner SQLite sidecar (M3).

Sessions persist in SQLite (default backend); ownership must persist too. Uses
the same ``db_path`` as the memory backend (additive table, ``CREATE TABLE IF
NOT EXISTS``). Falls back to ``:memory:`` when the memory backend is non-SQLite.
"""

from __future__ import annotations

import sqlite3
import time


class OwnershipStore:
    """SQLite-backed session ownership (``session_id → owner``)."""

    def __init__(self, db_path: str = "koboi_memory.db") -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        try:
            self._ensure_schema()
        except Exception:
            self._conn.close()
            raise

    def _ensure_schema(self) -> None:
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS session_owners ("
            "  session_id TEXT PRIMARY KEY,"
            "  owner TEXT NOT NULL,"
            "  created_at REAL NOT NULL"
            ")"
        )
        self._conn.commit()

    def set_owner(self, session_id: str, owner: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO session_owners (session_id, owner, created_at) VALUES (?, ?, ?)",
            (session_id, owner, time.time()),
        )
        self._conn.commit()

    def get_owner(self, session_id: str) -> str | None:
        row = self._conn.execute("SELECT owner FROM session_owners WHERE session_id = ?", (session_id,)).fetchone()
        return row["owner"] if row else None

    def is_owner(self, session_id: str, owner: str) -> bool:
        return self.get_owner(session_id) == owner

    def delete(self, session_id: str) -> None:
        self._conn.execute("DELETE FROM session_owners WHERE session_id = ?", (session_id,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
