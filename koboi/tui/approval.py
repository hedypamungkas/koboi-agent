"""koboi/guardrails/tui_approval.py -- Non-blocking approval handler for Textual TUI.

Replaces CLIApprovalHandler's stdin-based prompt with a Textual message-based
flow. The agent worker pauses via asyncio.Future while the user responds in a
PermissionDialog modal overlay.
"""
from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from textual.app import App
from textual.message import Message

from koboi.guardrails.approval import ApprovalHandler
from koboi.types import RiskLevel, AuditEntry

if TYPE_CHECKING:
    from koboi.guardrails.audit import AuditTrail
    from koboi.trust import TrustDatabase


# -- Textual Messages for permission flow --


class PermissionRequest(Message):
    """Posted to the app when a tool needs approval."""

    def __init__(
        self,
        tool_name: str,
        arguments: str,
        risk_level: str,
        future: asyncio.Future,
    ) -> None:
        super().__init__()
        self.tool_name = tool_name
        self.arguments = arguments
        self.risk_level = risk_level
        self.future = future


class PermissionResponse(Message):
    """Posted by the permission dialog when the user decides."""

    def __init__(
        self,
        approved: bool,
        always_allow: bool = False,
    ) -> None:
        super().__init__()
        self.approved = approved
        self.always_allow = always_allow


# -- Approval Handler --


class TUIApprovalHandler(ApprovalHandler):
    """ApprovalHandler that uses Textual's message system for non-blocking approval.

    The agent worker awaits an asyncio.Future while the user responds in a
    PermissionDialog overlay. This is non-blocking for the Textual event loop.
    """

    def __init__(
        self,
        app: App,
        trust_db: TrustDatabase | None = None,
        audit_trail: AuditTrail | None = None,
    ):
        self._app = app
        self._trust_db = trust_db
        self.audit_trail = audit_trail
        self._pending_future: asyncio.Future | None = None

    async def should_approve(
        self, tool_name: str, arguments: str, risk_level: RiskLevel
    ) -> bool:
        # 1. Check trust DB for auto-approval
        if self._trust_db:
            trust_decision = self._trust_db.should_auto_approve(tool_name, risk_level)
            if trust_decision.auto_approve:
                self._audit(tool_name, arguments, risk_level, True, trust_decision.reason)
                return True

        # 2. Create a future for the agent worker to await
        future: asyncio.Future[PermissionResponse] = asyncio.get_event_loop().create_future()
        self._pending_future = future

        # 3. Post permission request to the app (will show dialog)
        self._app.post_message(PermissionRequest(
            tool_name=tool_name,
            arguments=arguments,
            risk_level=risk_level.value,
            future=future,
        ))

        # 4. Await user response (non-blocking for Textual event loop)
        try:
            response = await future
        except asyncio.CancelledError:
            response = PermissionResponse(approved=False)

        self._pending_future = None

        # 5. Record in trust DB if "always allow"
        if response.always_allow and self._trust_db:
            self._trust_db.record_decision(
                tool_name=tool_name,
                risk_level=risk_level,
                decision="allow" if response.approved else "deny",
                always=True,
            )

        # 6. Audit
        self._audit(
            tool_name, arguments, risk_level, response.approved,
            "always_allow" if response.always_allow else "one_shot",
        )

        return response.approved

    def resolve_pending(self, response: PermissionResponse) -> None:
        """Called by the TUI when the user responds to the permission dialog."""
        if self._pending_future and not self._pending_future.done():
            self._pending_future.set_result(response)

    def cancel_pending(self) -> None:
        """Cancel any pending approval (e.g., on session end)."""
        if self._pending_future and not self._pending_future.done():
            self._pending_future.cancel()

    def _audit(
        self,
        tool_name: str,
        arguments: str,
        risk_level: RiskLevel,
        approved: bool,
        details: str,
    ) -> None:
        if self.audit_trail:
            self.audit_trail.record(AuditEntry(
                timestamp=time.time(),
                event_type="tool_approved" if approved else "tool_denied",
                tool_name=tool_name,
                arguments=arguments[:500],
                result="approved" if approved else "denied",
                risk_level=risk_level.value,
                details=f"TUI approval: {details}",
            ))
