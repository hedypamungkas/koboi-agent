"""koboi/guardrails/approval_types.py -- contract types for approval resolution.

Decoupled from Textual/HTTP so the same shapes are shared by:

* ``AsyncCallbackApprovalHandler`` (M0) -- the non-blocking approval handler used
  by the future REST/SSE server,
* the tool-execution pipeline's unified ``_resolve_approval`` path (M0), and
* pipeline unit tests.

Nothing here imports Textual, FastAPI, or asyncio -- only ``koboi.types`` -- so
the contract is cheap to import and stable across milestones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from koboi.types import RiskLevel

#: Why an approval resolution produced its outcome. Closed string set so audit
#: trails and metrics can group consistently.
OutcomeReason = Literal[
    "approved",  # a handler was asked and the user approved
    "denied",  # a handler was asked and the user denied (or trust deny rule)
    "skipped_via_trust",  # trust auto-allow rule matched; no prompt
    "no_handler",  # nothing to ask -> proceed (policy confirm is inert)
    "inert",  # handler present but this call did not require a new prompt
]


@dataclass
class ApprovalRequest:
    """Pipeline -> handler (and M2 server -> client) request shape.

    ``approval_id`` correlates the request with the future ``PendingApprovalEvent``
    emitted on the SSE stream (M2); M0 leaves it populated by the handler.
    """

    tool_name: str
    arguments: str
    risk_level: RiskLevel
    reason: str = ""
    approval_id: str = ""


@dataclass
class ApprovalResponse:
    """Handler -> pipeline response shape."""

    approved: bool
    always_allow: bool = False


class ApprovalCallback(Protocol):
    """Async contract ``AsyncCallbackApprovalHandler`` consumes.

    Decoupled from the legacy ``ApprovalHandler.should_approve -> bool`` ABC so
    the server can carry ``approval_id`` / ``scope`` through one typed object.
    Implementations may be a plain ``async def`` or any callable matching this.
    """

    async def __call__(self, request: ApprovalRequest) -> ApprovalResponse:  # pragma: no cover - protocol
        ...


@dataclass
class ApprovalOutcome:
    """Single return type from the pipeline's unified approval resolution.

    Carries enough state to (a) decide proceed/abort, (b) know whether a prompt
    already happened (so step-3 risk approval and step-4b policy confirmation
    cannot double-prompt the user), and (c) record a consistent audit entry.
    """

    proceed: bool
    reason: OutcomeReason
    prompted: bool = False
    audit_details: str = ""
    trust_rule: str | None = None
