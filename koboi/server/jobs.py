"""koboi/server/jobs -- autonomous background job runner (M4).

JobStore: SQLite ``jobs`` table (durable records). JobRegistry: in-memory
(task + event buffer + status). run_job: executes an agent with
AutonomousApprovalHandler, drains events to the buffer, updates status on
completion/failure/timeout/cancel. Resume-on-startup: requeue pending, mark
running-as-failed (simplified; full journal resume deferred to M5).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import re
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import httpx

if TYPE_CHECKING:
    from koboi.hooks.langfuse_hook import LangfuseTracingHook
    from koboi.server.pool import AgentPool

_logger = logging.getLogger(__name__)

#: Terminal statuses (no further state transitions).
TERMINAL = frozenset({"completed", "failed", "timed_out", "cancelled"})


class DuplicateIdempotencyKey(Exception):
    """M1: raised by JobStore.insert when a concurrent same-key insert won the race.

    Carries the existing (canonical) job_id so the caller can return it instead
    of creating a duplicate (double side-effect).
    """

    def __init__(self, existing_job_id: str) -> None:
        super().__init__(f"duplicate idempotency_key -> {existing_job_id}")
        self.existing_job_id = existing_job_id


# M2: redact common secret-value shapes from persisted error strings so a
# failure message never durable-stores leaked credentials.
_SECRET_VALUE_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),  # OpenAI-style keys
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key IDs
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+"),  # bearer tokens
    re.compile(r"(?i)(api[_-]?key|token|password|passwd|secret)[=:]\s*\S+"),
)


def _redact_error(text: str, limit: int = 500) -> str:
    """Mask common secret-value shapes and truncate a persisted error string (M2)."""
    redacted = text
    for pat in _SECRET_VALUE_PATTERNS:
        redacted = pat.sub("***REDACTED***", redacted)
    return redacted[:limit]


# ---------------------------------------------------------------------------
# JobStore — SQLite durable records
# ---------------------------------------------------------------------------


class JobStore:
    """SQLite-backed job records (``jobs`` table)."""

    def __init__(self, db_path: str = "koboi_memory.db") -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        try:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS jobs ("
                "  job_id TEXT PRIMARY KEY,"
                "  session_id TEXT NOT NULL,"
                "  owner TEXT NOT NULL,"
                "  status TEXT NOT NULL,"
                "  message TEXT,"
                "  result_json TEXT,"
                "  error TEXT,"
                "  error_class TEXT,"
                "  retriable INTEGER DEFAULT 0,"
                "  idempotency_key TEXT,"
                "  created_at REAL NOT NULL,"
                "  updated_at REAL NOT NULL"
                ")"
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_owner ON jobs(owner)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_idem ON jobs(idempotency_key)")
            # M1: unique partial index closes the find→insert TOCTOU window.
            # WHERE NOT NULL so the many NULL idempotency_key rows never conflict.
            # Wrapped: a legacy DB with pre-race duplicate keys degrades to
            # app-level dedup (warn) instead of bricking startup.
            try:
                self._conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_idem_unique "
                    "ON jobs(idempotency_key) WHERE idempotency_key IS NOT NULL"
                )
            except sqlite3.IntegrityError:
                _logger.warning(
                    "Could not create unique idempotency index (legacy duplicate keys present); "
                    "idempotency dedup degraded to app-level only."
                )
            self._migrate_add_columns()
            self._conn.commit()
        except Exception:
            self._conn.close()
            raise

    def _migrate_add_columns(self) -> None:
        """Idempotent ALTER TABLE for new columns on pre-existing M4 databases."""
        existing = {r["name"] for r in self._conn.execute("PRAGMA table_info(jobs)").fetchall()}
        if "error_class" not in existing:
            self._conn.execute("ALTER TABLE jobs ADD COLUMN error_class TEXT")
        if "retriable" not in existing:
            self._conn.execute("ALTER TABLE jobs ADD COLUMN retriable INTEGER DEFAULT 0")
        # G2: per-request mode + iteration cap, persisted so resume re-applies them.
        if "mode" not in existing:
            self._conn.execute("ALTER TABLE jobs ADD COLUMN mode TEXT")
        if "max_iterations" not in existing:
            self._conn.execute("ALTER TABLE jobs ADD COLUMN max_iterations INTEGER")

    def insert(
        self,
        job_id: str,
        session_id: str,
        owner: str,
        message: str,
        idempotency_key: str | None = None,
        mode: str | None = None,
        max_iterations: int | None = None,
    ) -> None:
        now = time.time()
        try:
            self._conn.execute(
                "INSERT INTO jobs (job_id, session_id, owner, status, message, "
                "idempotency_key, mode, max_iterations, created_at, updated_at) "
                "VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)",
                (job_id, session_id, owner, message, idempotency_key, mode, max_iterations, now, now),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as err:
            # M1: a concurrent same-key insert won the race. Roll back the failed
            # statement so the shared connection is reusable, then surface the
            # canonical job. (Without rollback the next SELECT raises
            # "Recursive use of cursors" on the shared connection.)
            self._conn.rollback()
            existing = self.find_by_idempotency_key(idempotency_key) if idempotency_key else None
            if existing:
                raise DuplicateIdempotencyKey(existing["job_id"]) from err
            raise

    def get(self, job_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def update_status(
        self,
        job_id: str,
        status: str,
        result_json: str | None = None,
        error: str | None = None,
        error_class: str | None = None,
        retriable: bool = False,
    ) -> None:
        self._conn.execute(
            "UPDATE jobs SET status = ?, result_json = ?, error = ?, error_class = ?, "
            "retriable = ?, updated_at = ? WHERE job_id = ?",
            (status, result_json, error, error_class, 1 if retriable else 0, time.time(), job_id),
        )
        self._conn.commit()

    def list_by_owner(self, owner: str, status: str | None = None) -> list[dict]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE owner = ? AND status = ? ORDER BY created_at DESC",
                (owner, status),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE owner = ? ORDER BY created_at DESC", (owner,)
            ).fetchall()
        return [dict(r) for r in rows]

    def find_by_idempotency_key(self, key: str, window_seconds: float = 86400) -> dict | None:
        cutoff = time.time() - window_seconds
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE idempotency_key = ? AND created_at >= ? ORDER BY created_at DESC LIMIT 1",
            (key, cutoff),
        ).fetchone()
        return dict(row) if row else None

    def list_by_status(self, status: str) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM jobs WHERE status = ? ORDER BY created_at", (status,)).fetchall()
        return [dict(r) for r in rows]

    def reap_terminal_older_than(self, cutoff: float) -> list[str]:
        """Delete terminal jobs updated before ``cutoff``. Returns reaped job_ids (G5c-a)."""
        rows = self._conn.execute(
            "SELECT job_id FROM jobs WHERE status IN (?, ?, ?, ?) AND updated_at < ?",
            ("completed", "failed", "timed_out", "cancelled", cutoff),
        ).fetchall()
        reaped = [r["job_id"] for r in rows]
        for jid in reaped:
            self._conn.execute("DELETE FROM jobs WHERE job_id = ?", (jid,))
        if reaped:
            self._conn.commit()
        return reaped

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# JobRegistry — in-memory task + event buffer + status
# ---------------------------------------------------------------------------


@dataclass
class JobRecord:
    """In-memory tracking for a single job."""

    job_id: str
    session_id: str
    owner: str
    status: str = "pending"
    events: list = field(default_factory=list)
    task: asyncio.Task | None = None
    # Lazy: ``asyncio.Event()`` binds to the current event loop at creation time,
    # and on Python 3.9 raises ``RuntimeError`` if no loop is set. Creating it
    # eagerly in the dataclass default_factory coupled JobRecord construction to
    # having a live loop — which broke sync callers (e.g. unit tests) when the
    # loop had been cleared. Deferred to first access (always in async context).
    _terminal: asyncio.Event | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def terminal(self) -> asyncio.Event:
        if self._terminal is None:
            self._terminal = asyncio.Event()
        return self._terminal


class JobRegistry:
    """In-memory job registry with capped event buffer."""

    def __init__(self, max_events: int = 500) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._pending: deque[str] = deque()
        self._max_events = max_events

    def register(self, job_id: str, session_id: str, owner: str) -> JobRecord:
        record = JobRecord(job_id=job_id, session_id=session_id, owner=owner)
        self._jobs[job_id] = record
        return record

    def get(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    def list_by_owner(self, owner: str) -> list[JobRecord]:
        return [r for r in self._jobs.values() if r.owner == owner]

    def append_event(self, job_id: str, event: Any) -> None:
        record = self._jobs.get(job_id)
        if record:
            record.events.append(event)
            if len(record.events) > self._max_events:
                record.events = record.events[-self._max_events :]

    def set_running(self, job_id: str, task: asyncio.Task) -> None:
        record = self._jobs.get(job_id)
        if record:
            record.status = "running"
            record.task = task

    def set_terminal(self, job_id: str, status: str) -> None:
        record = self._jobs.get(job_id)
        if record:
            record.status = status
            record.terminal.set()

    async def cancel(self, job_id: str) -> bool:
        record = self._jobs.get(job_id)
        if record and record.task and not record.task.done():
            record.task.cancel()
            return True
        return False

    @property
    def active_count(self) -> int:
        return sum(1 for r in self._jobs.values() if r.status == "running")

    def active_count_for_owner(self, owner: str) -> int:
        """Running jobs for one owner (per-tenant concurrency basis; G5a)."""
        return sum(1 for r in self._jobs.values() if r.owner == owner and r.status == "running")

    def peek_admit(self, max_concurrent: int, queue_depth: int) -> str:
        """Non-mutating admission decision for a new job: ``run`` | ``queue`` | ``reject``."""
        if self.active_count < max_concurrent:
            return "run"
        if len(self._pending) < queue_depth:
            return "queue"
        return "reject"

    def enqueue_pending(self, job_id: str) -> None:
        self._pending.append(job_id)

    def pop_pending(self) -> str | None:
        return self._pending.popleft() if self._pending else None

    def remove_pending(self, job_id: str) -> bool:
        try:
            self._pending.remove(job_id)
            return True
        except ValueError:
            return False

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def forget(self, job_ids) -> None:
        """Drop records (used by the job TTL reaper). Pending entries are cleaned defensively."""
        ids = set(job_ids)
        for jid in ids:
            self._jobs.pop(jid, None)
        self._pending = deque(j for j in self._pending if j not in ids)

    def cancel_all(self) -> int:
        """Cancel all active job tasks (used by graceful shutdown). Returns count cancelled."""
        count = 0
        for record in self._jobs.values():
            if record.task and not record.task.done():
                record.task.cancel()
                count += 1
        return count


# ---------------------------------------------------------------------------
# run_job + resume_on_startup
# ---------------------------------------------------------------------------


def new_job_id() -> str:
    return f"job_{uuid4().hex[:24]}"


# Outbound webhook delivery for terminal job statuses. Fire-and-forget tasks are
# tracked here so CPython doesn't GC them mid-POST (same pattern as CommandHook).
_WEBHOOK_TASKS: set[asyncio.Task] = set()
_WEBHOOK_DEFAULT_TIMEOUT = 10.0


def _webhook_payload(store: JobStore, job_id: str, status: str) -> dict | None:
    """Build the JSON payload for a terminal job status from the stored row."""
    row = store.get(job_id)
    if row is None:
        return None
    result: Any = None
    raw = row.get("result_json")
    if raw:
        try:
            result = json.loads(raw)
        except (TypeError, ValueError):
            result = raw
    return {
        "job_id": job_id,
        "session_id": row.get("session_id"),
        "owner": row.get("owner"),
        "status": status,
        "event": f"job.{status}",
        "result": result,
        "error": row.get("error"),
        "error_class": row.get("error_class"),
        "retriable": bool(row.get("retriable")),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


async def _post_webhook(url: str, body: bytes, headers: dict, timeout: float) -> None:
    """POST ``body`` to ``url``; 2 attempts on 5xx / network error; fail-safe (logs only)."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            last = None
            for _attempt in range(2):
                try:
                    resp = await client.post(url, content=body, headers=headers)
                    if resp.status_code < 500:
                        return
                    last = f"HTTP {resp.status_code}"
                except (httpx.TimeoutException, httpx.ConnectError) as exc:
                    last = str(exc)
            if last:
                _logger.warning("job webhook %s failed after retry: %s", url, last)
    except Exception as exc:  # noqa: BLE001 -- never let delivery break the job flow
        _logger.warning("job webhook %s error: %s", url, exc)


async def _deliver_webhooks(webhooks: list[dict], store: JobStore, job_id: str, status: str) -> None:
    """POST the job payload to every webhook whose events match ``status``."""
    if not webhooks:
        return
    payload = _webhook_payload(store, job_id, status)
    if payload is None:
        return
    body = json.dumps(payload).encode()
    for wh in webhooks:
        events = wh.get("events") or []
        if events and status not in events:
            continue
        url = wh.get("url")
        if not url:
            continue
        headers = {"Content-Type": "application/json"}
        secret = wh.get("secret")
        if secret:
            signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
            headers["X-Koboi-Signature"] = f"sha256={signature}"
        timeout = wh.get("timeout") or _WEBHOOK_DEFAULT_TIMEOUT
        await _post_webhook(url, body, headers, float(timeout))


def _emit_job_webhooks(webhooks: list[dict] | None, store: JobStore, job_id: str, status: str) -> None:
    """Schedule outbound webhook delivery for a terminal status (fire-and-forget).

    Called after ``set_terminal`` so the job-queue admission (``_on_job_done``) and
    next-job scheduling are not delayed by HTTP. Holds task refs to avoid GC.
    """
    if not webhooks:
        return
    task = asyncio.create_task(_deliver_webhooks(webhooks, store, job_id, status))
    _WEBHOOK_TASKS.add(task)
    task.add_done_callback(_WEBHOOK_TASKS.discard)


async def run_job(
    job_id: str,
    pool: AgentPool,
    registry: JobRegistry,
    store: JobStore,
    message: str,
    timeout: float = 1800,
    mode: str | None = None,
    max_iterations: int | None = None,
    resume: bool = False,
    webhooks: list[dict] | None = None,
) -> None:
    """Execute a job: create agent, install AutonomousApprovalHandler, run, drain events.

    G2: ``mode``/``max_iterations`` are persisted on the jobs row and re-applied
    on resume. ``mode`` is validated + yolo-rejected at submit, so it is trusted here.
    ``resume=True`` (#5) rehydrates-and-continues an interrupted job via
    ``AgentCore.resume()`` instead of re-running ``run_stream(message)``.
    """

    record = registry.get(job_id)
    if record is None:
        return

    try:
        final_content = await asyncio.wait_for(
            _execute_job(job_id, pool, registry, store, message, mode, max_iterations, resume=resume),
            timeout=timeout,
        )
        result_json = json.dumps({"content": final_content}) if final_content else None
        store.update_status(job_id, "completed", result_json=result_json)
        registry.set_terminal(job_id, "completed")
        _emit_job_webhooks(webhooks, store, job_id, "completed")
    except asyncio.CancelledError:
        store.update_status(job_id, "cancelled")
        registry.set_terminal(job_id, "cancelled")
        _emit_job_webhooks(webhooks, store, job_id, "cancelled")
        raise
    except asyncio.TimeoutError:
        store.update_status(
            job_id, "timed_out", error="Job exceeded timeout", error_class="TimeoutError", retriable=True
        )
        registry.set_terminal(job_id, "timed_out")
        _emit_job_webhooks(webhooks, store, job_id, "timed_out")
    except Exception as exc:
        # M2: log type only (no traceback/locals) + mask/truncate the persisted
        # error so a failure never durable-stores the user prompt or leaked creds.
        _logger.error("Job %s failed: %s", job_id, type(exc).__name__)
        store.update_status(
            job_id,
            "failed",
            error=_redact_error(str(exc)),
            error_class=type(exc).__name__,
            retriable=False,
        )
        registry.set_terminal(job_id, "failed")
        _emit_job_webhooks(webhooks, store, job_id, "failed")


async def _execute_job(
    job_id: str,
    pool: AgentPool,
    registry: JobRegistry,
    store: JobStore,
    message: str,
    mode: str | None = None,
    max_iterations: int | None = None,
    resume: bool = False,
) -> str | None:
    """Inner execution: agent setup + run_stream → event buffer.

    Returns the final content (from ``CompleteEvent``) for ``result_json``
    persistence so completed jobs survive restart.
    """
    from koboi.events import CompleteEvent

    record = registry.get(job_id)
    agent = await pool.get_or_create(record.session_id)

    # C3: autonomous jobs must run contained. 'passthrough' has no fs/network
    # isolation, so refuse it -- raise before running; run_job marks the job failed.
    sb = agent._core.tools.get_dep("sandbox")
    if getattr(sb, "name", "passthrough") == "passthrough":
        raise PermissionError(
            "Autonomous jobs require sandbox.backend='restricted'; 'passthrough' is refused. "
            "Configure the 'sandbox:' section before enabling jobs."
        )

    store.update_status(job_id, "running")
    # 16.21: enrich Langfuse trace with job context.
    if agent._core and agent._core.hooks:
        lf_hook = agent._core.hooks.find_hook(lambda h: type(h).__name__ == "LangfuseTracingHook")
        if lf_hook:
            # find_hook returns the base Hook type; cast to the duck-typed langfuse
            # hook (looked up by class name) to satisfy mypy's attr-defined check.
            cast("LangfuseTracingHook", lf_hook).set_serving_metadata(
                mode="autonomous", job_id=job_id, owner=record.owner
            )
    final_content: str | None = None
    async with pool.session_lock(record.session_id):
        prior_handler = agent._core.approval_handler
        had_pipeline = hasattr(agent._core, "_tool_pipeline")
        prior_pipeline = getattr(agent._core, "_tool_pipeline", None)
        # G2: save mode + iteration cap to restore after the run; the pooled agent
        # is reused across jobs/sessions. mode persists on the jobs row so a
        # resumed job re-stamps the same mode.
        prior_mode = agent._core.mode_manager.current_mode
        prior_max_iter = agent._core.max_iterations
        try:
            if had_pipeline:
                del agent._core._tool_pipeline
            from koboi.guardrails.approval import AutonomousApprovalHandler
            from koboi.modes import AgentMode

            agent._core.approval_handler = AutonomousApprovalHandler(
                trust_db=agent.trust_db,
                audit_trail=agent._core.audit_trail,
                # Autonomous jobs run contained in a restricted sandbox (C3
                # refuses passthrough above), so in-workdir file writes are safe
                # to auto-approve. Without this, every write_file/delete_file is
                # denied and file-producing jobs can't run (e.g. job_multi_write_grep).
                auto_approve_tools={"write_file", "delete_file"},
            )
            if mode is not None:
                agent._core.mode_manager.switch_mode(AgentMode(mode))
            if max_iterations is not None:
                agent._core.max_iterations = max_iterations
            if resume:
                # #5: rehydrate-and-continue the interrupted loop. The agent's memory
                # + journal were already rehydrated by pool.get_or_create (memory.
                # session_id), so resume() continues from the interrupted step.
                result = await agent.resume()
                final_content = result.content
            else:
                async for event in agent.run_stream(message):
                    registry.append_event(job_id, event)
                    if isinstance(event, CompleteEvent):
                        final_content = event.content
        finally:
            agent._core.approval_handler = prior_handler
            if had_pipeline:
                agent._core._tool_pipeline = prior_pipeline
            agent._core.mode_manager.switch_mode(prior_mode)
            agent._core.max_iterations = prior_max_iter
    return final_content


async def resume_on_startup(
    store: JobStore,
    pool: AgentPool,
    registry: JobRegistry,
    timeout: float,
    webhooks: list[dict] | None = None,
) -> int:
    """Resume interrupted jobs + requeue pending ones (#5: rehydrate-and-continue).

    Running jobs (killed mid-flight by a redeploy) are rehydrated-and-continued via
    ``AgentCore.resume()`` (``run_job(resume=True)``) rather than marked failed: the
    agent's memory + journal rehydrate via ``pool.get_or_create(session_id)``, so the
    interrupted loop continues from its last durable step. A resume failure falls
    through to ``run_job``'s exception handler (mark failed). Returns the count of
    resumed + requeued jobs.

    Webhook emission for resumed/requeued jobs happens inside ``run_job`` itself
    (all terminal branches call ``_emit_job_webhooks``) -- no separate emit needed
    here for the resume path.
    """
    count = 0

    # #5: rehydrate-and-continue running jobs (was: mark running-as-failed).
    for job in store.list_by_status("running"):
        registry.register(job["job_id"], job["session_id"], job["owner"])
        task = asyncio.create_task(
            run_job(
                job["job_id"],
                pool,
                registry,
                store,
                job["message"],
                timeout,
                job.get("mode"),
                job.get("max_iterations"),
                resume=True,
                webhooks=webhooks,
            )
        )
        registry.set_running(job["job_id"], task)
        count += 1
        _logger.info("Resuming interrupted job %s on startup", job["job_id"])

    # Requeue pending jobs (fresh run).
    for job in store.list_by_status("pending"):
        registry.register(job["job_id"], job["session_id"], job["owner"])
        task = asyncio.create_task(
            run_job(
                job["job_id"],
                pool,
                registry,
                store,
                job["message"],
                timeout,
                job.get("mode"),
                job.get("max_iterations"),
                webhooks=webhooks,
            )
        )
        registry.set_running(job["job_id"], task)
        count += 1
        _logger.info("Requeued job %s on startup", job["job_id"])

    return count
