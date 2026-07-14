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
TERMINAL = frozenset({"completed", "failed", "timed_out", "cancelled", "awaiting_human"})


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
        if "workflow_ref" not in existing:
            self._conn.execute("ALTER TABLE jobs ADD COLUMN workflow_ref TEXT")
        # v2: cache-mode workflow jobs (replay_mode) + their per-job cache_dir (capture).
        if "replay_mode" not in existing:
            self._conn.execute("ALTER TABLE jobs ADD COLUMN replay_mode TEXT")
        if "cache_dir" not in existing:
            self._conn.execute("ALTER TABLE jobs ADD COLUMN cache_dir TEXT")

    def set_cache_dir(self, job_id: str, cache_dir: str) -> None:
        """Record a cache-mode workflow job's per-job cache_dir (for capture)."""
        self._conn.execute("UPDATE jobs SET cache_dir = ? WHERE job_id = ?", (cache_dir, job_id))
        self._conn.commit()

    def insert(
        self,
        job_id: str,
        session_id: str,
        owner: str,
        message: str,
        idempotency_key: str | None = None,
        mode: str | None = None,
        max_iterations: int | None = None,
        workflow_ref: str | None = None,
        replay_mode: str | None = None,
    ) -> None:
        now = time.time()
        try:
            self._conn.execute(
                "INSERT INTO jobs (job_id, session_id, owner, status, message, "
                "idempotency_key, mode, max_iterations, workflow_ref, replay_mode, created_at, updated_at) "
                "VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    session_id,
                    owner,
                    message,
                    idempotency_key,
                    mode,
                    max_iterations,
                    workflow_ref,
                    replay_mode,
                    now,
                    now,
                ),
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

    def get_events(self, job_id: str) -> list[Any]:
        """Return the capped event list for a job (issue #1 EventBuffer surface).

        Route handlers read events through this method (not ``record.events``
        directly) so a future Redis EventBuffer can swap in transparently.
        """
        record = self._jobs.get(job_id)
        return list(record.events) if record else []

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


def _on_webhook_task_done(task: asyncio.Task) -> None:
    """Discard the strong ref + surface any exception that escaped ``_deliver_webhooks``.

    ``_post_webhook`` catches its own delivery errors (network/5xx), but a bug in
    ``_deliver_webhooks`` itself (e.g. a non-numeric ``timeout`` config value, or a
    non-str ``secret``) would otherwise escape as an unretrieved task exception --
    logged only by asyncio's default handler at GC time, bypassing this module's
    logger entirely. Mirrors ``CommandHook._on_bg_done``.
    """
    _WEBHOOK_TASKS.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        _logger.error("job webhook delivery failed: %s", exc, exc_info=True)


async def drain_webhook_tasks(timeout: float = 5.0) -> None:
    """Await in-flight fire-and-forget webhook deliveries (bounded).

    Without this, a graceful shutdown (``_shutdown`` -> ``job_store.close()``) can
    race an in-flight webhook POST for the very last job to finish -- the delivery
    that most needs to land is the one most likely to be silently cut off. Called
    from the server's ``_shutdown`` before closing the job store.
    """
    if not _WEBHOOK_TASKS:
        return
    try:
        await asyncio.wait_for(asyncio.gather(*_WEBHOOK_TASKS, return_exceptions=True), timeout=timeout)
    except asyncio.TimeoutError:
        _logger.warning("job webhook drain exceeded %.1fs; %d task(s) abandoned", timeout, len(_WEBHOOK_TASKS))


def _emit_job_webhooks(webhooks: list[dict] | None, store: JobStore, job_id: str, status: str) -> None:
    """Schedule outbound webhook delivery for a terminal status (fire-and-forget).

    Called after ``set_terminal`` so the job-queue admission (``_on_job_done``) and
    next-job scheduling are not delayed by HTTP. Holds task refs to avoid GC.
    """
    if not webhooks:
        return
    task = asyncio.create_task(_deliver_webhooks(webhooks, store, job_id, status))
    _WEBHOOK_TASKS.add(task)
    task.add_done_callback(_on_webhook_task_done)


def _emit_handover_webhook(
    webhooks: list[dict] | None,
    session_id: str,
    handover_id: str,
    reason: str,
    summary: str,
) -> None:
    """B5: fire-and-forget -- notify the host CS platform of a CHAT-path handover.

    Unlike ``_emit_job_webhooks`` (terminal job status), this fires mid-conversation
    when a ``HandoverEvent`` is emitted on ``/chat/stream`` (B1/B1.5). Reuses the
    jobs ``_post_webhook`` (2-retry, fail-safe) + HMAC signing + ``_WEBHOOK_TASKS``.
    Payload: ``{event: "handover.requested", session_id, handover_id, reason, summary}``.
    Job-path handovers already fire ``job.awaiting_human`` via ``_emit_job_webhooks``.
    """
    if not webhooks:
        return
    payload = {
        "event": "handover.requested",
        "session_id": session_id,
        "handover_id": handover_id,
        "reason": reason,
        "summary": summary,
    }
    body = json.dumps(payload).encode()
    for wh in webhooks:
        url = wh.get("url")
        if not url:
            continue
        headers = {"Content-Type": "application/json"}
        secret = wh.get("secret")
        if secret:
            signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
            headers["X-Koboi-Signature"] = f"sha256={signature}"
        timeout = wh.get("timeout") or _WEBHOOK_DEFAULT_TIMEOUT
        task = asyncio.create_task(_post_webhook(url, body, headers, float(timeout)))
        _WEBHOOK_TASKS.add(task)
        task.add_done_callback(_on_webhook_task_done)


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
    workflow_ref: str | None = None,
    workflow_store: Any | None = None,
    replay_mode: str | None = None,
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

    from koboi.exceptions import AgentHandoverError  # noqa: PLC0415 (lazy; jobs.py keeps koboi imports function-local)

    try:
        final_content = await asyncio.wait_for(
            _execute_job(
                job_id,
                pool,
                registry,
                store,
                message,
                mode,
                max_iterations,
                resume=resume,
                workflow_ref=workflow_ref,
                workflow_store=workflow_store,
                replay_mode=replay_mode,
            ),
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
    except AgentHandoverError as he:
        # B1: the agent yielded via transfer_to_human -> awaiting_human (NOT failed).
        # Not reapable (not in the reaper's status IN tuple) -- it awaits human action.
        store.update_status(
            job_id,
            "awaiting_human",
            result_json=json.dumps({"reason": he.reason, "summary": he.summary}),
        )
        registry.set_terminal(job_id, "awaiting_human")
        _emit_job_webhooks(webhooks, store, job_id, "awaiting_human")
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


async def _execute_workflow_job(
    job_id: str,
    registry: JobRegistry,
    store: JobStore,
    message: str,
    workflow_ref: str,
    workflow_store: Any | None,
    owner: str,
    resume: bool = False,
    replay_mode: str | None = None,
) -> str | None:
    """Run a stored workflow bundle as an autonomous job.

    Builds a fresh :class:`~koboi.facade.KoboiAgent` from the bundle YAML (NOT the
    pooled server-level agent) and runs it via ``run_stream`` (works for both
    single-agent and orchestrator-backed bundles). Enforces the C3 floor:
    ``sandbox.backend='restricted'`` is required.

    v1 limitations: the agent is rebuilt on every run (no pooled session memory),
    so ``resume`` re-runs from scratch; per-sub-agent ``AutonomousApprovalHandler``
    is not installed on orchestrator-backed bundles (the restricted sandbox is the
    safety floor).
    """
    from koboi.events import CompleteEvent

    if workflow_store is None:
        raise ValueError("workflow jobs require a workflow_store")
    wf = workflow_store.get(workflow_ref, owner)
    if wf is None:
        raise ValueError(f"unknown workflow {workflow_ref!r} for this owner")
    from koboi.config import Config
    from koboi.facade import KoboiAgent

    bundle_yaml = wf["bundle_yaml"]
    cfg = Config.from_string(bundle_yaml)
    if cfg.get("sandbox", "backend", default="passthrough") != "restricted":
        raise PermissionError(
            "Autonomous workflow jobs require sandbox.backend='restricted'; 'passthrough' is refused."
        )
    # v2 cache mode: mint a per-job cache_dir; hydrate a captured sidecar if the
    # workflow has one (offline replay). prepare_captured_bundle points the bundle
    # at the per-job dir so CachedClient hits.
    effective_yaml = bundle_yaml
    _effective_mode = replay_mode or cfg.get("replay", "mode", default="live")
    if _effective_mode in ("cache", "replay"):
        cache_dir = f".koboi/cache/jobs/{job_id}"
        _get_sidecar = getattr(workflow_store, "get_sidecar", None)
        if _get_sidecar is not None:
            _sc = _get_sidecar(owner, workflow_ref)
            if _sc is not None:
                from koboi.llm.cache import ResponseCache

                _entries = [(e.key, e.payload) for e in _sc.read()]
                ResponseCache(cache_dir).load_entries(_entries)
                _logger.info("hydrated %d cached response(s) for job %s", len(_entries), job_id)
            elif _effective_mode == "replay":
                _logger.warning("replay mode but no sidecar for workflow %r -- will raise on miss", workflow_ref)
        elif _effective_mode == "replay":
            _logger.warning("workflow_store has no get_sidecar -- hydration skipped for job %s", job_id)
        from koboi.workflows import prepare_captured_bundle

        effective_yaml = prepare_captured_bundle(bundle_yaml, cache_dir=cache_dir, mode=_effective_mode)
        store.set_cache_dir(job_id, cache_dir)
    agent = KoboiAgent.from_config_string(effective_yaml)
    # Install the same deny-by-default handler as a regular autonomous job so a
    # single-agent workflow job is NOT more permissive (C3 + Trust-DB gate).
    # Orchestrator-backed bundles (agent._core is None) keep the restricted
    # sandbox as their floor; per-sub-agent handler install is deferred (v1).
    if agent._core is not None:
        from koboi.guardrails.approval import AutonomousApprovalHandler

        agent._core.approval_handler = AutonomousApprovalHandler(
            trust_db=agent.trust_db,
            audit_trail=agent._core.audit_trail,
            auto_approve_tools={"write_file", "delete_file"},
        )
    store.update_status(job_id, "running")
    final_content: str | None = None
    try:
        async for event in agent.run_stream(message):
            registry.append_event(job_id, event)
            if isinstance(event, CompleteEvent):
                final_content = event.content
    finally:
        await agent.close()
    return final_content


async def _execute_plain_cache_job(
    job_id: str,
    pool: AgentPool,
    registry: JobRegistry,
    store: JobStore,
    message: str,
    record: Any,
    mode: str | None = None,
    max_iterations: int | None = None,
    replay_mode: str = "cache",
) -> str | None:
    """Run a PLAIN (non-workflow_ref) job in cache/replay mode with an isolated
    per-job cache_dir (the pooled agent's shared client can't isolate a run).

    Builds a fresh ``KoboiAgent`` from the server config (``with_replay`` + per-job
    ``cache_dir``), enforces the C3 restricted-sandbox floor + the deny-by-default
    ``AutonomousApprovalHandler``, and records the cache_dir so the run can be
    captured with cache. v3 #4-a.
    """
    from koboi.events import CompleteEvent
    from koboi.facade import KoboiAgent
    from koboi.guardrails.approval import AutonomousApprovalHandler
    from koboi.modes import AgentMode

    cache_dir = f".koboi/cache/jobs/{job_id}"
    cfg = pool._config.with_replay(replay_mode=replay_mode, cache_dir=cache_dir)
    # Per-job isolation (mirror AgentPool._build_agent): stamp the session_id + a
    # per-session workdir so concurrent plain cache jobs don't share memory/fs state.
    import os

    cfg._data.setdefault("memory", {})["session_id"] = record.session_id
    workdir = pool.workdir_for(record.session_id)
    cfg._data.setdefault("sandbox", {})["workdir"] = workdir
    os.makedirs(workdir, exist_ok=True)
    agent = KoboiAgent._from_config(cfg)
    if agent._core is None:
        raise PermissionError(
            "plain cache/replay jobs require a single-agent server config "
            "(orchestration-mode servers are unsupported for plain cache jobs)"
        )
    sb = agent._core.tools.get_dep("sandbox")
    if getattr(sb, "name", "passthrough") == "passthrough":
        raise PermissionError("Autonomous jobs require sandbox.backend='restricted'; 'passthrough' is refused.")
    agent._core.approval_handler = AutonomousApprovalHandler(
        trust_db=agent.trust_db,
        audit_trail=agent._core.audit_trail,
        auto_approve_tools={"write_file", "delete_file"},
    )
    if mode is not None:
        agent._core.mode_manager.switch_mode(AgentMode(mode))
    if max_iterations is not None:
        agent._core.max_iterations = max_iterations
    store.set_cache_dir(job_id, cache_dir)
    store.update_status(job_id, "running")
    final_content: str | None = None
    try:
        async for event in agent.run_stream(message):
            registry.append_event(job_id, event)
            if isinstance(event, CompleteEvent):
                final_content = event.content
    finally:
        await agent.close()
    return final_content


async def _execute_job(
    job_id: str,
    pool: AgentPool,
    registry: JobRegistry,
    store: JobStore,
    message: str,
    mode: str | None = None,
    max_iterations: int | None = None,
    resume: bool = False,
    workflow_ref: str | None = None,
    workflow_store: Any | None = None,
    replay_mode: str | None = None,
) -> str | None:
    """Inner execution: agent setup + run_stream → event buffer.

    Returns the final content (from ``CompleteEvent``) for ``result_json``
    persistence so completed jobs survive restart. When ``workflow_ref`` is set,
    execution is delegated to :func:`_execute_workflow_job` (builds the agent from
    the stored bundle instead of the pooled server-level agent).
    """
    record = registry.get(job_id)
    if workflow_ref:
        return await _execute_workflow_job(
            job_id,
            registry,
            store,
            message,
            workflow_ref,
            workflow_store,
            record.owner,
            resume=resume,
            replay_mode=replay_mode,
        )
    if replay_mode in ("cache", "replay"):
        # v3 #4-a: a plain (non-workflow_ref) cache/replay job builds a fresh
        # per-job agent (the pooled agent shares one client and can't isolate
        # this run's cache). Restricted sandbox + AutonomousApprovalHandler apply.
        return await _execute_plain_cache_job(
            job_id, pool, registry, store, message, record, mode, max_iterations, replay_mode
        )

    from koboi.events import CompleteEvent, OrchestrationCompleteEvent

    agent = await pool.get_or_create(record.session_id)

    # W5.1 B1-jobs middle path: orchestrated configs (core=None) run with a config-level
    # sandbox check (not job-level AutonomousApprovalHandler). deep_research nodes carry
    # their own sandbox/approval from factory build -- the AutonomousApprovalHandler only
    # auto-approves write_file/delete_file (unused by deep_research nodes), so skipping it
    # is safe. Single-agent configs (core is not None) fall through to the full job path below.
    if agent._core is None:
        backend = (
            agent._config.get("sandbox", "backend", default="passthrough")
            if hasattr(agent, "_config") and agent._config
            else "passthrough"
        )
        if backend == "passthrough":
            raise PermissionError(
                "Autonomous jobs require sandbox.backend='restricted'; 'passthrough' is refused. "
                "Configure the 'sandbox:' section before enabling jobs."
            )
        store.update_status(job_id, "running")
        orchestrated_content: str | None = None
        async with pool.session_lock(record.session_id):
            if resume:
                result = await agent.resume()
                orchestrated_content = result.content
            else:
                async for event in agent.run_stream(message):
                    registry.append_event(job_id, event)
                    if isinstance(event, CompleteEvent):
                        orchestrated_content = event.content
                    elif isinstance(event, OrchestrationCompleteEvent):
                        # deep_research/dynamic/dag emit OrchestrationCompleteEvent
                        # (NOT CompleteEvent); the cited report is in final_answer.
                        orchestrated_content = event.final_answer
        return orchestrated_content

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
    workflow_store: Any | None = None,
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
        if job.get("workflow_ref") or job.get("replay_mode") in ("cache", "replay"):
            # workflow_ref + plain cache/replay jobs build a fresh agent per run and
            # cannot rehydrate an interrupted loop, so resuming would duplicate side
            # effects. Mark failed (retriable) so the operator re-submits.
            store.update_status(
                job["job_id"],
                "failed",
                error="cache/replay + workflow jobs cannot be resumed; re-submit the job",
                error_class="WorkflowResumeUnsupported",
                retriable=True,
            )
            registry.register(job["job_id"], job["session_id"], job["owner"])
            registry.set_terminal(job["job_id"], "failed")
            _logger.warning("Workflow job %s cannot be resumed in v1; marked failed", job["job_id"])
            count += 1
            continue
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
                workflow_ref=job.get("workflow_ref"),
                workflow_store=workflow_store,
                replay_mode=job.get("replay_mode"),
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
                workflow_ref=job.get("workflow_ref"),
                workflow_store=workflow_store,
                replay_mode=job.get("replay_mode"),
            )
        )
        registry.set_running(job["job_id"], task)
        count += 1
        _logger.info("Requeued job %s on startup", job["job_id"])

    return count
