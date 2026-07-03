"""Audit trail for recording important events for compliance and debugging."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from koboi.types import AuditEntry, RiskLevel

if TYPE_CHECKING:
    from koboi.logger import AgentLogger


class AuditTrail:
    """Record all important events for compliance and debugging."""

    def __init__(self, logger: AgentLogger | None = None):
        self.logger = logger
        self._entries: list[AuditEntry] = []

    def record(self, entry: AuditEntry) -> None:
        self._entries.append(entry)
        if self.logger:
            self.logger.log(
                f"[AUDIT] {entry.event_type}"
                f"{' tool=' + entry.tool_name if entry.tool_name else ''}"
                f"{f' risk={entry.risk_level}' if entry.risk_level else ''}"
                f" | {entry.details}"
            )

    def get_entries(
        self,
        event_type: str | None = None,
        tool_name: str | None = None,
    ) -> list[AuditEntry]:
        entries = self._entries
        if event_type:
            entries = [e for e in entries if e.event_type == event_type]
        if tool_name:
            entries = [e for e in entries if e.tool_name == tool_name]
        return entries

    def summary(self) -> dict:
        total = len(self._entries)
        by_type: dict[str, int] = defaultdict(int)
        by_tool: dict[str, int] = defaultdict(int)
        blocked = 0
        for e in self._entries:
            by_type[e.event_type] += 1
            if e.tool_name:
                by_tool[e.tool_name] += 1
            if (
                "denied" in e.details.lower()
                or "block" in (e.result or "").lower()
                or "passed=false" in e.details.lower()
            ):
                blocked += 1
        return {
            "total_events": total,
            "blocked": blocked,
            "by_type": dict(by_type),
            "by_tool": dict(by_tool),
        }


class SQLiteAuditTrail(AuditTrail):
    """AuditTrail backed by SQLite for persistence across restarts."""

    _memory_cap: int = 100

    def __init__(
        self,
        db_path: str = "koboi_audit.db",
        logger: AgentLogger | None = None,
    ):
        super().__init__(logger=logger)
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
        return self._conn

    def _init_db(self) -> None:
        conn = self._ensure_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                tool_name TEXT,
                arguments TEXT,
                result TEXT,
                risk_level TEXT,
                details TEXT DEFAULT ''
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_event ON audit_entries(event_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_tool ON audit_entries(tool_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_entries(timestamp)")
        conn.commit()

    def record(self, entry: AuditEntry) -> None:
        super().record(entry)
        if len(self._entries) > self._memory_cap:
            self._entries = self._entries[-self._memory_cap :]
        conn = self._ensure_conn()
        risk = entry.risk_level.value if isinstance(entry.risk_level, RiskLevel) else entry.risk_level
        conn.execute(
            "INSERT INTO audit_entries "
            "(timestamp, event_type, tool_name, arguments, result, risk_level, details) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                entry.timestamp,
                entry.event_type,
                entry.tool_name,
                entry.arguments,
                entry.result,
                risk,
                entry.details,
            ),
        )
        conn.commit()

    def query_db(
        self,
        event_type: str | None = None,
        tool_name: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query persisted audit entries from SQLite."""
        conn = self._ensure_conn()
        clauses: list[str] = []
        params: list = []
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if tool_name:
            clauses.append("tool_name = ?")
            params.append(tool_name)
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        where = " AND ".join(clauses) if clauses else "1=1"
        query = (
            f"SELECT timestamp, event_type, tool_name, arguments, "
            f"result, risk_level, details "
            f"FROM audit_entries WHERE {where} "
            f"ORDER BY timestamp DESC LIMIT ?"
        )
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [
            {
                "timestamp": r[0],
                "event_type": r[1],
                "tool_name": r[2],
                "arguments": r[3],
                "result": r[4],
                "risk_level": r[5],
                "details": r[6],
            }
            for r in rows
        ]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
