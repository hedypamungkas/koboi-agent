"""koboi/server/workflow_store -- SQLite-backed workflow store (owner-scoped).

A near-verbatim clone of :class:`~koboi.server.ownership.OwnershipStore` /
:class:`~koboi.server.jobs.JobStore`: a ``workflows`` table keyed by
``(owner, name)`` holding the bundle YAML. ``owner`` is the auth-middleware
``api_key_id``. Imported lazily by ``app.py`` (only when the ``[api]`` extra is
present).
"""

from __future__ import annotations

import sqlite3
import time


class WorkflowStore:
    """SQLite-backed workflow bundles (``(owner, name) -> bundle_yaml``)."""

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
            "CREATE TABLE IF NOT EXISTS workflows ("
            "  name TEXT NOT NULL,"
            "  owner TEXT NOT NULL,"
            "  bundle_yaml TEXT NOT NULL,"
            "  description TEXT,"
            "  created_at REAL NOT NULL,"
            "  updated_at REAL NOT NULL,"
            "  PRIMARY KEY (owner, name)"
            ")"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_workflows_owner ON workflows(owner)")
        # v2: captured-bundle response-cache sidecar (owner + name scoped).
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS workflows_cache ("
            "  owner TEXT NOT NULL, name TEXT NOT NULL, cache_key TEXT NOT NULL,"
            "  payload_json TEXT NOT NULL, model TEXT, created_at REAL NOT NULL,"
            "  PRIMARY KEY (owner, name, cache_key)"
            ")"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_workflows_cache_owner ON workflows_cache(owner, name)")
        self._conn.commit()

    def put(self, name: str, owner: str, bundle_yaml: str, description: str | None = None) -> None:
        now = time.time()
        # Upsert: keep created_at on overwrite, refresh the bundle + updated_at.
        self._conn.execute(
            "INSERT INTO workflows (name, owner, bundle_yaml, description, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(owner, name) DO UPDATE SET "
            "bundle_yaml=excluded.bundle_yaml, description=excluded.description, "
            "updated_at=excluded.updated_at",
            (name, owner, bundle_yaml, description, now, now),
        )
        self._conn.commit()

    def put_with_sidecar(
        self,
        name: str,
        owner: str,
        bundle_yaml: str,
        description: str | None = None,
        entries: list[tuple[str, dict]] | None = None,
    ) -> None:
        """Upsert a bundle + freeze cache entries into the ``workflows_cache`` sidecar."""
        self.put(name, owner, bundle_yaml, description=description)
        if entries:
            from koboi.workflows.cache_sidecar import SqliteCacheSidecar

            SqliteCacheSidecar(self._conn, owner, name).write(entries)

    def get_sidecar(self, owner: str, name: str):
        """Return a SqliteCacheSidecar, or None when the workflow has no cached entries."""
        from koboi.workflows.cache_sidecar import SqliteCacheSidecar

        sidecar = SqliteCacheSidecar(self._conn, owner, name)
        return sidecar if len(sidecar) > 0 else None

    def get(self, name: str, owner: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM workflows WHERE name = ? AND owner = ?", (name, owner)).fetchone()
        return dict(row) if row else None

    def list_by_owner(self, owner: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT name, description, created_at, updated_at FROM workflows WHERE owner = ? ORDER BY name",
            (owner,),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete(self, name: str, owner: str) -> bool:
        cur = self._conn.execute("DELETE FROM workflows WHERE name = ? AND owner = ?", (name, owner))
        self._conn.execute(
            "DELETE FROM workflows_cache WHERE name = ? AND owner = ?", (name, owner)
        )  # v2: cascade the sidecar
        self._conn.commit()
        return cur.rowcount > 0

    def ping(self) -> bool:
        try:
            self._conn.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    def close(self) -> None:
        self._conn.close()
