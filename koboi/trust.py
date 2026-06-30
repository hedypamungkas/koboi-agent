"""koboi/trust.py -- SQLite-backed trust database for graduated permissions.

Learns from user approval decisions to reduce permission fatigue.
When a user says "always allow" for a tool type, future calls auto-approve.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Protocol, runtime_checkable

from koboi.types import RiskLevel

#: H5: default TTL (7 days) for new "always allow" rules -- trust rules are no
#: longer permanent. ``record_decision`` applies this when ``ttl_seconds`` is
#: None. Existing rows (``expires_at IS NULL``) are left as-is on upgrade.
DEFAULT_TRUST_TTL_SECONDS: float = 604800.0


def _args_hash(arguments: str) -> str | None:
    """sha256 of ``arguments``; None (wildcard) when arguments is empty.

    A NULL args_hash matches any arguments (back-compat); a non-NULL hash matches
    only the exact recorded arguments -- so "always allow write_file(/tmp/x)" does
    not auto-approve "write_file(/etc/passwd)".
    """
    if not arguments:
        return None
    return hashlib.sha256(arguments.encode("utf-8")).hexdigest()


@dataclass
class TrustDecision:
    """Result of a trust lookup."""

    auto_approve: bool
    matched_rule: str | None = None
    reason: str = ""


@dataclass
class TrustRule:
    """A stored trust rule."""

    id: int
    tool_pattern: str
    risk_level: str
    decision: str  # "allow" or "deny"
    created_at: float
    expires_at: float | None
    context: str


@runtime_checkable
class TrustStore(Protocol):
    """Minimal read/write surface the tool-execution pipeline consumes.

    ``TrustDatabase`` is the default implementation. Defining this Protocol now
    (M0) means a future multi-tenant store (per-tenant scoping, Redis/Postgres
    backing) can be swapped in at the facade wiring point without editing
    ``loop_pipeline.py`` or its tests.
    """

    def should_auto_approve(self, tool_name: str, risk_level: RiskLevel, arguments: str = "") -> TrustDecision: ...

    def record_decision(
        self,
        tool_name: str,
        risk_level: RiskLevel,
        decision: str,
        always: bool = False,
        ttl_seconds: float | None = None,
        arguments: str = "",
    ) -> None: ...


class TrustDatabase:
    """SQLite-backed trust store for graduated permissions.

    Schema:
        trust_rules:
            id INTEGER PRIMARY KEY AUTOINCREMENT
            tool_pattern TEXT NOT NULL  -- glob pattern
            risk_level TEXT NOT NULL    -- "safe", "moderate", "destructive"
            decision TEXT NOT NULL      -- "allow" or "deny"
            created_at REAL NOT NULL
            expires_at REAL             -- NULL = never expires
            context TEXT                -- JSON metadata
    """

    def __init__(self, db_path: str = "koboi_trust.db"):
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS trust_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_pattern TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                decision TEXT NOT NULL CHECK (decision IN ('allow', 'deny')),
                created_at REAL NOT NULL,
                expires_at REAL,
                context TEXT,
                args_hash TEXT
            )
        """)
        # H5: additive migration for pre-existing DBs (argument-hash scoping).
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(trust_rules)").fetchall()}
        if "args_hash" not in cols:
            self._conn.execute("ALTER TABLE trust_rules ADD COLUMN args_hash TEXT")
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trust_pattern
            ON trust_rules(tool_pattern)
        """)
        self._conn.commit()

    def should_auto_approve(self, tool_name: str, risk_level: RiskLevel, arguments: str = "") -> TrustDecision:
        """Check if a tool call should be auto-approved based on learned trust.

        Returns TrustDecision with auto_approve=True if a matching "allow" rule
        exists and hasn't expired. ``arguments`` scopes the match by sha256 hash
        (H5): a rule recorded with specific args matches only those args; a rule
        recorded without args (NULL args_hash) is a wildcard.
        """
        now = time.time()
        want_hash = _args_hash(arguments)
        rows = self._conn.execute(
            "SELECT * FROM trust_rules WHERE decision = 'allow' ORDER BY created_at DESC"
        ).fetchall()

        for row in rows:
            # Skip expired rules
            if row["expires_at"] is not None and row["expires_at"] < now:
                continue
            # Check pattern match
            if not fnmatch(tool_name, row["tool_pattern"]):
                continue
            # H5: scope by argument hash. NULL args_hash = wildcard (back-compat);
            # else the rule matches only its exact recorded arguments.
            rule_hash = row["args_hash"]
            if rule_hash is not None and rule_hash != want_hash:
                continue
            # Check risk level compatibility — allow rule covers equal or lower risk
            rule_risk = row["risk_level"]
            if self._risk_leq(risk_level.value, rule_risk):
                return TrustDecision(
                    auto_approve=True,
                    matched_rule=row["tool_pattern"],
                    reason=f"Auto-approved by trust rule: {row['tool_pattern']} ({rule_risk})",
                )

        # Check for deny rules (kept global -- conservative)
        for row in self._conn.execute(
            "SELECT * FROM trust_rules WHERE decision = 'deny' ORDER BY created_at DESC"
        ).fetchall():
            if row["expires_at"] is not None and row["expires_at"] < now:
                continue
            if fnmatch(tool_name, row["tool_pattern"]):
                return TrustDecision(
                    auto_approve=False,
                    matched_rule=row["tool_pattern"],
                    reason=f"Auto-denied by trust rule: {row['tool_pattern']}",
                )

        return TrustDecision(auto_approve=False, reason="No matching trust rule")

    def record_decision(
        self,
        tool_name: str,
        risk_level: RiskLevel,
        decision: str,
        always: bool = False,
        ttl_seconds: float | None = None,
        arguments: str = "",
    ) -> None:
        """Record a user's approval decision.

        Args:
            tool_name: The tool that was approved/denied.
            risk_level: The risk level of the tool call.
            decision: "allow" or "deny".
            always: If True, create a persistent rule. If False, record for
                    statistics only (no future auto-approval).
            ttl_seconds: Optional TTL for the rule. None = DEFAULT_TRUST_TTL_SECONDS
                    (H5: rules are no longer permanent).
            arguments: Optional argument string; hashed to scope the rule (H5).
        """
        if not always:
            return  # One-shot decisions don't create rules

        # H5: no permanent rules -- default TTL when none given.
        if ttl_seconds is None:
            ttl_seconds = DEFAULT_TRUST_TTL_SECONDS
        expires_at = time.time() + ttl_seconds

        # Use the tool name as a glob pattern (exact match)
        self._conn.execute(
            "INSERT INTO trust_rules (tool_pattern, risk_level, decision, created_at, expires_at, context, args_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                tool_name,
                risk_level.value,
                decision,
                time.time(),
                expires_at,
                json.dumps({"always": always}),
                _args_hash(arguments),
            ),
        )
        self._conn.commit()

    def clear_rules(self, tool_pattern: str | None = None) -> int:
        """Clear trust rules. Returns number of rules removed."""
        if tool_pattern:
            cursor = self._conn.execute("DELETE FROM trust_rules WHERE tool_pattern = ?", (tool_pattern,))
        else:
            cursor = self._conn.execute("DELETE FROM trust_rules")
        self._conn.commit()
        return cursor.rowcount

    def get_rules(self) -> list[TrustRule]:
        """List all active (non-expired) trust rules."""
        now = time.time()
        rows = self._conn.execute(
            "SELECT * FROM trust_rules WHERE expires_at IS NULL OR expires_at > ? ORDER BY created_at DESC",
            (now,),
        ).fetchall()
        return [
            TrustRule(
                id=row["id"],
                tool_pattern=row["tool_pattern"],
                risk_level=row["risk_level"],
                decision=row["decision"],
                created_at=row["created_at"],
                expires_at=row["expires_at"],
                context=row["context"] or "",
            )
            for row in rows
        ]

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _risk_leq(risk_a: str, risk_b: str) -> bool:
        """Check if risk_a <= risk_b in severity order."""
        order = {"safe": 0, "moderate": 1, "destructive": 2}
        return order.get(risk_a, 0) <= order.get(risk_b, 0)
