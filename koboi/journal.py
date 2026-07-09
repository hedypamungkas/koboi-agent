"""koboi/journal -- SQLite-backed step journal for durability + resume (P2-A).

Records one row per loop iteration (1 LLM call + its tool calls) so a crashed
or redeployed agent can be resumed via ``koboi run --resume <session>``. The
journal borrows a :class:`~koboi.memory_sqlite.SQLiteMemory` connection so step
rows live in the same WAL-mode ``koboi_memory.db`` as the conversation.

Resume semantics (rehydrate + continue): the conversation is the source of
truth; the journal marks step status ('running' -> 'interrupted' on resume) and
captures per-step token deltas for observability. Each ``(turn, step)`` has
exactly one row (upsert): the 'running' marker inserts, and the step outcome
updates that same row. Re-running an interrupted step updates its row in place.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import TYPE_CHECKING

from koboi.redact import redact_tool_arguments

if TYPE_CHECKING:
    from koboi.types import ToolCall

_logger = logging.getLogger(__name__)


class StepJournal:
    """Per-session step journal. SQLite-only.

    Constructed by the facade from a SQLiteMemory's connection + session_id and
    passed to AgentCore as the optional ``journal`` collaborator.
    """

    def __init__(self, conn: sqlite3.Connection, session_id: str, record_tool_calls: bool = True):
        self._conn = conn
        self._session_id = session_id
        self._record_tool_calls = record_tool_calls
        self._turn_index = self._compute_initial_turn()

    # -- turn accounting ---------------------------------------------------

    def _compute_initial_turn(self) -> int:
        """Highest turn already recorded for this session (0 if fresh).

        On resume this lets the next run() continue the turn numbering instead
        of restarting at 1.
        """
        row = self._conn.execute(
            "SELECT COALESCE(MAX(turn_index), 0) FROM steps WHERE session_id = ?",
            (self._session_id,),
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    @property
    def turn_index(self) -> int:
        return self._turn_index

    def advance_turn(self) -> None:
        self._turn_index += 1

    # -- write paths -------------------------------------------------------

    def record_step(
        self,
        *,
        turn_index: int,
        step_index: int,
        status: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        tool_calls: list[ToolCall] | None = None,
        is_terminal: bool = False,
        error: str | None = None,
    ) -> int:
        """Record one step row (upsert by session/turn/step) and commit.

        Exactly one row per (turn, step): the 'running' marker INSERTs, and the
        step outcome ('complete'/'tool_calls'/'max_iter'/...) UPDATEs that same
        row. Committing eagerly makes the 'running' crash-marker durable right
        away (before the LLM call) so a mid-step crash leaves recoverable
        evidence -- a 'running' row with no outcome is an interrupted step.
        Returns the row id.
        """
        tool_calls_json = None
        tool_call_count = 0
        if tool_calls and self._record_tool_calls:
            # Redact secrets in tool arguments before durable storage so the
            # step journal never persists leaked credentials. Round-trippable:
            # redact_tool_arguments returns a JSON string of the same shape.
            tool_calls_json = json.dumps(
                [{"name": tc.name, "arguments": redact_tool_arguments(tc.arguments)} for tc in tool_calls]
            )
            tool_call_count = len(tool_calls)
        elif tool_calls:
            tool_call_count = len(tool_calls)

        existing = self._conn.execute(
            "SELECT id FROM steps WHERE session_id = ? AND turn_index = ? AND step_index = ?",
            (self._session_id, turn_index, step_index),
        ).fetchone()
        if existing:
            row_id = int(existing[0])
            self._conn.execute(
                "UPDATE steps SET status = ?, llm_prompt_tokens = ?, llm_completion_tokens = ?, "
                "tool_call_count = ?, tool_calls_json = ?, is_terminal = ?, error = ?, "
                "created_at = julianday('now') WHERE id = ?",
                (
                    status,
                    prompt_tokens,
                    completion_tokens,
                    tool_call_count,
                    tool_calls_json,
                    1 if is_terminal else 0,
                    error,
                    row_id,
                ),
            )
        else:
            cur = self._conn.execute(
                "INSERT INTO steps "
                "(session_id, turn_index, step_index, status, llm_prompt_tokens, "
                " llm_completion_tokens, tool_call_count, tool_calls_json, is_terminal, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self._session_id,
                    turn_index,
                    step_index,
                    status,
                    prompt_tokens,
                    completion_tokens,
                    tool_call_count,
                    tool_calls_json,
                    1 if is_terminal else 0,
                    error,
                ),
            )
            row_id = int(cur.lastrowid)
        self._conn.commit()
        return row_id

    def mark_interrupted(self, rows: list[dict]) -> None:
        """Flip prior 'running' rows to 'interrupted' (called once on resume)."""
        if not rows:
            return
        ids = [r["id"] for r in rows if r.get("id") is not None]
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        # Safe: only '?' placeholders are interpolated; ids are bound as params
        # (standard variable-length IN-clause pattern, no user data in SQL).
        self._conn.execute(  # nosec B608 - parameterized placeholders, not data
            f"UPDATE steps SET status = 'interrupted' WHERE id IN ({placeholders})",
            ids,
        )
        self._conn.commit()
        _logger.info("Resume: marked %d interrupted step(s) for session %s", len(ids), self._session_id)

    # -- read paths --------------------------------------------------------

    def list_open_running(self) -> list[dict]:
        """'running' rows for this session (crash markers awaiting resume)."""
        rows = self._conn.execute(
            "SELECT id, turn_index, step_index, status, created_at FROM steps "
            "WHERE session_id = ? AND status = 'running' ORDER BY id",
            (self._session_id,),
        ).fetchall()
        cols = ["id", "turn_index", "step_index", "status", "created_at"]
        return [dict(zip(cols, r, strict=False)) for r in rows]

    def get_last_terminal_step(self) -> dict | None:
        """The most recent terminal step (complete/max_iter/error), or None."""
        rows = self._conn.execute(
            "SELECT id, turn_index, step_index, status, is_terminal, error, created_at "
            "FROM steps WHERE session_id = ? AND is_terminal = 1 ORDER BY id DESC LIMIT 1",
            (self._session_id,),
        ).fetchall()
        if not rows:
            return None
        cols = ["id", "turn_index", "step_index", "status", "is_terminal", "error", "created_at"]
        return dict(zip(cols, rows[0], strict=False))

    def list_steps(self, turn_index: int | None = None) -> list[dict]:
        """List steps for this session (optionally one turn), oldest first."""
        if turn_index is None:
            rows = self._conn.execute(
                "SELECT id, turn_index, step_index, status, llm_prompt_tokens, "
                "llm_completion_tokens, tool_call_count, is_terminal, error, created_at "
                "FROM steps WHERE session_id = ? ORDER BY id",
                (self._session_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, turn_index, step_index, status, llm_prompt_tokens, "
                "llm_completion_tokens, tool_call_count, is_terminal, error, created_at "
                "FROM steps WHERE session_id = ? AND turn_index = ? ORDER BY id",
                (self._session_id, turn_index),
            ).fetchall()
        cols = [
            "id",
            "turn_index",
            "step_index",
            "status",
            "llm_prompt_tokens",
            "llm_completion_tokens",
            "tool_call_count",
            "is_terminal",
            "error",
            "created_at",
        ]
        return [dict(zip(cols, r, strict=False)) for r in rows]
