"""koboi/server/approvals -- HITL approval coordinator + registry (M2).

``ApprovalCoordinator`` (per-run) bridges ``AsyncCallbackApprovalHandler`` to the
SSE stream: when a tool needs approval, the coordinator pushes a
``PendingApprovalEvent`` onto the run's ``asyncio.Queue`` (so the SSE drain loop
emits it WHILE ``run_stream`` is suspended), then awaits an ``asyncio.Future``
resolved by ``POST /v1/sessions/:id/approve``.

``ApprovalRegistry`` is the server-level ``session_id -> coordinator`` map so the
approve route can find the active coordinator. One coordinator per session at a
time (the per-session lock serializes runs).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from koboi.events import PendingApprovalEvent
from koboi.guardrails.approval_types import ApprovalRequest, ApprovalResponse

if TYPE_CHECKING:
    pass


class ApprovalCoordinator:
    """Per-run approval coordinator (one per ``/chat/stream`` invocation).

    The ``AsyncCallbackApprovalHandler``'s ``callback`` is ``coordinator.request``.
    It pushes a ``PendingApprovalEvent`` onto the shared queue, then awaits a
    ``Future`` that ``POST /approve`` resolves.
    """

    def __init__(self, queue: asyncio.Queue, timeout: float = 120.0) -> None:
        self._queue = queue
        self._timeout = timeout
        self._futures: dict[str, asyncio.Future[ApprovalResponse]] = {}

    async def request(self, req: ApprovalRequest) -> ApprovalResponse:
        """Handler callback: emit PendingApprovalEvent + await the decision."""
        event = PendingApprovalEvent(
            approval_id=req.approval_id,
            tool_name=req.tool_name,
            arguments=req.arguments,
            risk_level=req.risk_level.value,
            reason=req.reason or "risk-based approval",
            timeout_seconds=self._timeout,
        )
        self._queue.put_nowait(event)  # sync; unbounded queue never blocks
        future: asyncio.Future[ApprovalResponse] = asyncio.get_event_loop().create_future()
        self._futures[req.approval_id] = future
        return await future  # handler's wait_for enforces the timeout

    def resolve(self, approval_id: str, response: ApprovalResponse) -> bool:
        """Resolve a pending approval. Returns False if not found or already done."""
        future = self._futures.get(approval_id)
        if future is None or future.done():
            return False
        future.set_result(response)
        return True

    def cancel_all(self) -> None:
        """Cancel all pending approvals (on client disconnect / run end)."""
        for future in self._futures.values():
            if not future.done():
                future.cancel()
        self._futures.clear()


class ApprovalRegistry:
    """Server-level map: ``session_id`` -> active ``ApprovalCoordinator``."""

    def __init__(self) -> None:
        self._coordinators: dict[str, ApprovalCoordinator] = {}

    def register(self, session_id: str, coordinator: ApprovalCoordinator) -> None:
        self._coordinators[session_id] = coordinator

    def get(self, session_id: str) -> ApprovalCoordinator | None:
        return self._coordinators.get(session_id)

    def unregister(self, session_id: str) -> None:
        coord = self._coordinators.pop(session_id, None)
        if coord is not None:
            coord.cancel_all()
