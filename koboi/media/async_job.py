"""koboi/media/async_job.py -- shared async-job poll loop for video + music (W1).

Gateway async generation (Surplus ``/v1/video|music/generations``) returns a job on submit
(``queued``) and resolves over minutes via a status state machine
``queued -> submitted -> running -> succeeded | failed | canceled | expired``. This module
implements the blocking facade once: ``submit -> poll until terminal -> fetch artifact``.

Providers implement the three transport calls (``submit``/``poll``/``fetch``); the poll-loop
bookkeeping (intervals, deadline, terminal handling, fail-soft) lives here and is shared by
``BaseVideoProvider`` and ``BaseMusicProvider``.

Fail-soft: transport errors and ``max_wait`` exhaustion surface as ``failed`` ``MediaResult``
(carrying the ``job_id`` in ``raw`` so a caller can resume via the provider's advanced API),
never as raised exceptions.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from koboi.media.types import MediaRequest, MediaResult

_TERMINAL_OK = frozenset({"succeeded"})
_TERMINAL_BAD = frozenset({"failed", "canceled", "expired"})
_TERMINAL = _TERMINAL_OK | _TERMINAL_BAD


@dataclass
class MediaJob:
    """An in-flight async generation job. ``result`` is populated by ``poll`` on success."""

    job_id: str
    kind: str  # "video" | "music"
    status: str = "queued"
    poll_url: str | None = None
    cancel_url: str | None = None
    job_token: str | None = None
    estimated_cost_usd: Decimal | None = None
    max_cost_usd: Decimal | None = None
    progress: float | None = None
    result: MediaResult | None = None
    raw: dict[str, Any] = field(default_factory=dict)


# Provider transport callables. poll may populate job.result with metadata (content_type,
# duration_seconds, width/height, cost_usd, billing_unit) on terminal success; fetch returns the
# artifact bytes which run_async_job then attaches to that result.
SubmitFn = Callable[[MediaRequest], Awaitable[MediaJob]]
PollFn = Callable[[MediaJob], Awaitable[MediaJob]]
FetchFn = Callable[[MediaJob], Awaitable[bytes]]


async def run_async_job(
    submit: SubmitFn,
    poll: PollFn,
    fetch: FetchFn,
    req: MediaRequest,
    kind: str,
    *,
    poll_interval: float = 5.0,
    max_wait: float = 1500.0,
) -> MediaResult:
    """Submit -> poll until terminal -> fetch artifact. Blocking facade for async providers.

    On ``max_wait`` exhaustion returns a ``failed`` result carrying the ``job_id`` (in ``raw``) so
    a caller can resume via the provider's advanced poll API rather than hanging.
    """
    request_id = req.idempotency_key or ""

    try:
        job = await submit(req)
    except Exception as e:  # noqa: BLE001 - boundary: any submit failure becomes a failed result
        return _failed(request_id, kind, f"submit failed: {e}")

    deadline = time.monotonic() + max_wait
    while job.status not in _TERMINAL:
        if time.monotonic() >= deadline:
            return MediaResult(
                request_id=request_id,
                modality=kind,
                status="failed",
                rejection_reason=f"job {job.job_id} still running after {max_wait:.0f}s",
                raw={"job_id": job.job_id, "kind": kind, "last_status": job.status},
            )
        await asyncio.sleep(poll_interval)
        try:
            job = await poll(job)
        except Exception as e:  # noqa: BLE001 - boundary: any poll failure becomes a failed result
            return _failed(request_id, kind, f"poll failed: {e}", job_id=job.job_id)

    if job.status in _TERMINAL_BAD:
        status = "rejected" if job.status == "canceled" else "failed"
        return MediaResult(
            request_id=request_id,
            modality=kind,
            status=status,
            rejection_reason=f"job {job.status}",
            raw={**job.raw, "job_id": job.job_id},
        )

    # succeeded -> attach fetched bytes to the metadata-rich result the provider built in poll.
    result = job.result if job.result is not None else MediaResult(request_id=request_id, modality=kind, raw=job.raw)
    try:
        result.data = await fetch(job)
    except Exception as e:  # noqa: BLE001 - boundary: any fetch failure becomes a failed result
        return _failed(request_id, kind, f"artifact fetch failed: {e}", job_id=job.job_id)
    result.request_id = request_id or result.request_id
    result.modality = kind
    result.status = "ok"
    return result


def _failed(request_id: str, kind: str, reason: str, *, job_id: str | None = None) -> MediaResult:
    raw: dict[str, Any] = {"kind": kind}
    if job_id:
        raw["job_id"] = job_id
    return MediaResult(request_id=request_id, modality=kind, status="failed", rejection_reason=reason, raw=raw)
