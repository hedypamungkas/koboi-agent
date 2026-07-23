"""koboi/harness/background_shell -- start/poll/kill long-running shell processes.

Wave 4. The sandbox's ``run()`` is built entirely on ``Popen(...).communicate()``
(run-to-completion) -- there is no "start and return a handle" primitive anywhere
else in the codebase. ``BackgroundShellManager`` adds one, deliberately additive
(``run_shell``/``BaseSandbox.run()`` are untouched): spawn via
``asyncio.create_subprocess_shell``, drain stdout/stderr continuously (never
``communicate()``, which would block until exit and risk a full-pipe deadlock
with nothing reading it), and track live jobs in an in-memory registry mirroring
``koboi.media.backend.MediaBackend._jobs`` -- NOT durable across a process
restart; a crash orphans the OS process (a future wave could persist PID+pgid to
SQLite, mirroring ``JobStore``, to reap orphans on restart).

Security note: once a job is started, it runs entirely outside the per-call
approval/policy/audit pipeline for its whole lifetime -- ``check_command_blocked``
only evaluates the *launch* command, not anything the process does afterward.
``max_lifetime_seconds`` is the mitigating cap. See koboi/tools/builtin/background_shell.py.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import uuid
from dataclasses import dataclass, field
from typing import Literal

_logger = logging.getLogger(__name__)

_GRACE_SECONDS = 5.0

# Closed state set for BackgroundShellJob.status -- a Literal (not a bare str)
# makes a typo'd transition (e.g. "exit" vs "exited") a static error instead of
# silently breaking admission accounting (_running_count) and kill idempotency.
BgShellStatus = Literal["running", "exited", "killed", "timeout"]


@dataclass
class BackgroundShellJob:
    job_id: str
    command: str
    cwd: str
    pid: int
    status: BgShellStatus = "running"
    returncode: int | None = None
    output: list[str] = field(default_factory=list)
    output_chars: int = 0
    # Set by _terminate() BEFORE signaling, so the single status write (in
    # _drain(), the coroutine that actually awaits proc.wait()) records the
    # INTENDED outcome ("killed"/"timeout") instead of racing _terminate's own
    # concurrent proc.wait() to plain "exited".
    pending_status: BgShellStatus | None = None


class BackgroundShellManager:
    """In-memory registry of live background shell processes.

    One instance is shared by the ``submit_background_shell``/``check_background_shell``/
    ``kill_background_shell`` tools (mirrors ``submit_media_job``/``check_media_job``'s
    shared-backend split, ``koboi/tools/builtin/media.py``).
    """

    def __init__(
        self,
        *,
        max_concurrent: int = 4,
        output_buffer_chars: int = 20000,
        default_max_lifetime: float = 1800.0,
        grace_seconds: float = _GRACE_SECONDS,
    ) -> None:
        self._jobs: dict[str, BackgroundShellJob] = {}
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        self._max_concurrent = max_concurrent
        self._output_buffer_chars = output_buffer_chars
        self._default_max_lifetime = default_max_lifetime
        self._grace_seconds = grace_seconds
        # Reader/watchdog tasks must not be GC'd mid-flight (mirrors
        # server/jobs.py's module-level _WEBHOOK_TASKS tracking pattern).
        self._bg_tasks: set[asyncio.Task] = set()

    def _running_count(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status == "running")

    async def start(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        max_lifetime_seconds: float | None = None,
    ) -> BackgroundShellJob:
        from koboi.harness.policy import check_command_blocked

        blocked = check_command_blocked(command)
        if blocked:
            raise ValueError(blocked)
        if max_lifetime_seconds is not None and max_lifetime_seconds <= 0:
            # 0/negative scheduled a watchdog that fired instantly, killing the
            # process at launch (mislabeled "timeout"). Require a positive cap.
            raise ValueError("max_lifetime_seconds must be positive")
        if self._running_count() >= self._max_concurrent:
            raise ValueError(f"max_concurrent background shell jobs reached ({self._max_concurrent})")

        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd or None,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,  # own process group -> killpg never touches the agent's own group
        )
        job_id = f"bgsh_{uuid.uuid4().hex[:24]}"
        job = BackgroundShellJob(job_id=job_id, command=command, cwd=cwd or ".", pid=proc.pid)
        self._jobs[job_id] = job
        self._procs[job_id] = proc

        self._spawn_tracked(self._drain(job_id, proc))
        lifetime = max_lifetime_seconds if max_lifetime_seconds is not None else self._default_max_lifetime
        self._spawn_tracked(self._watchdog(job_id, lifetime))
        return job

    def _spawn_tracked(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _drain(self, job_id: str, proc: asyncio.subprocess.Process) -> None:
        """Continuously read stdout/stderr into a bounded ring buffer.

        Must run concurrently with the process, never after ``proc.wait()`` --
        an unread full pipe buffer would deadlock the child.
        """
        job = self._jobs[job_id]

        async def _read_stream(stream: asyncio.StreamReader | None) -> None:
            if stream is None:
                return
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode(errors="replace")
                job.output.append(text)
                job.output_chars += len(text)
                while job.output_chars > self._output_buffer_chars and job.output:
                    dropped = job.output.pop(0)
                    job.output_chars -= len(dropped)

        await asyncio.gather(_read_stream(proc.stdout), _read_stream(proc.stderr), return_exceptions=True)
        returncode = await proc.wait()
        # The single status write for this job -- avoids a race against
        # _terminate()'s own concurrent proc.wait() (see BackgroundShellJob.pending_status).
        job.status = job.pending_status or "exited"
        job.returncode = returncode

    async def _watchdog(self, job_id: str, max_lifetime: float) -> None:
        await asyncio.sleep(max_lifetime)
        job = self._jobs.get(job_id)
        if job is None or job.status != "running":
            return
        _logger.warning("Background shell job %s exceeded max_lifetime_seconds=%s -- killing", job_id, max_lifetime)
        await self._terminate(job_id, force=False, mark_status="timeout")

    async def poll(self, job_id: str) -> BackgroundShellJob | None:
        return self._jobs.get(job_id)

    def tail(self, job_id: str, lines: int = 50) -> str:
        job = self._jobs.get(job_id)
        if job is None:
            return ""
        return "".join(job.output[-lines:]) if lines > 0 else "".join(job.output)

    async def kill(self, job_id: str, *, force: bool = False) -> BackgroundShellJob | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.status != "running":
            return job  # idempotent: killing an already-terminal job is a no-op
        await self._terminate(job_id, force=force, mark_status="killed")
        return self._jobs[job_id]

    async def _terminate(self, job_id: str, *, force: bool, mark_status: BgShellStatus) -> None:
        job = self._jobs.get(job_id)
        proc = self._procs.get(job_id)
        if job is None or proc is None or job.status != "running":
            return
        # Record intent BEFORE any signal/await: both this coroutine and _drain()
        # await the same proc.wait() concurrently, so whichever finishes first must
        # write the SAME target value -- pending_status makes that deterministic
        # instead of a race between "killed"/"timeout" and a generic "exited".
        job.pending_status = mark_status
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            job.status = mark_status
            return
        try:
            if not force:
                os.killpg(pgid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=self._grace_seconds)
                except asyncio.TimeoutError:
                    pass
            if proc.returncode is None:
                os.killpg(pgid, signal.SIGKILL)
                await proc.wait()
        except ProcessLookupError:
            pass
        job.status = mark_status
        job.returncode = proc.returncode

    async def kill_all(self) -> None:
        """Force-kill every live job. Called on shutdown/close (see facade.py)."""
        for job_id, job in list(self._jobs.items()):
            if job.status == "running":
                await self.kill(job_id, force=True)
