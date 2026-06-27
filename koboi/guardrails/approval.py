"""Human approval for destructive tools.

In CLI, approval is a y/n prompt. For programmatic usage,
can be skipped with auto_approve=True or overridden with custom callback.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable
from typing import Callable
from typing import TYPE_CHECKING

from koboi.types import AuditEntry, RiskLevel

if TYPE_CHECKING:
    from koboi.guardrails.audit import AuditTrail

_rich_available = False
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm

    _rich_available = True
except ImportError:
    pass

_console = Console() if _rich_available else None


def _risk_color(level: RiskLevel) -> str:
    return {"safe": "green", "moderate": "yellow", "destructive": "red"}.get(level.value, "white")


class ApprovalHandler:
    """Base class — deny destructive ops by default, allow others."""

    def should_approve(self, tool_name: str, arguments: str, risk_level: RiskLevel) -> bool | Awaitable[bool]:
        # May be sync (bool) or async (Awaitable[bool]); the execution pipeline
        # awaits when iscoroutinefunction(should_approve) is True.
        return risk_level != RiskLevel.DESTRUCTIVE


class CLIApprovalHandler(ApprovalHandler):
    """Prompt user in terminal before executing destructive tool."""

    def __init__(
        self,
        require_for: set[str] | None = None,
        audit_trail: AuditTrail | None = None,
    ):
        self.require_for = require_for or {"destructive"}
        self.audit_trail = audit_trail

    def should_approve(self, tool_name: str, arguments: str, risk_level: RiskLevel) -> bool:
        if risk_level.value not in self.require_for:
            return True

        if _rich_available and _console is not None:
            color = _risk_color(risk_level)
            _console.print(
                Panel(
                    f"[bold]Tool:[/bold] {tool_name}\n"
                    f"[bold]Risk:[/bold] [{color}]{risk_level.value}[/{color}]\n"
                    f"[bold]Args:[/bold] {arguments[:200]}",
                    title="[bold yellow]Approval Required[/bold yellow]",
                    border_style="yellow",
                )
            )
            try:
                approved = Confirm.ask("Approve execution?", default=False)
            except (EOFError, KeyboardInterrupt):
                approved = False
        else:
            print(f"\nApproval Required\n  Tool: {tool_name}\n  Risk: {risk_level.value}\n  Args: {arguments[:200]}")
            try:
                answer = input("Approve execution? [y/N]: ").strip().lower()
                approved = answer in ("y", "yes")
            except (EOFError, KeyboardInterrupt):
                approved = False

        if self.audit_trail:
            event = "tool_approved" if approved else "tool_denied"
            self.audit_trail.record(
                AuditEntry(
                    timestamp=time.time(),
                    event_type=event,
                    tool_name=tool_name,
                    arguments=arguments[:500],
                    result="approved" if approved else "denied",
                    risk_level=risk_level.value,
                    details=f"Human approval via CLI: {'yes' if approved else 'no'}",
                )
            )

        return approved


class CallbackApprovalHandler(ApprovalHandler):
    """Use custom callback for approval (for web/API integration)."""

    def __init__(
        self,
        callback: Callable[[str, str, str], bool],
        audit_trail: AuditTrail | None = None,
    ):
        self.callback = callback
        self.audit_trail = audit_trail

    def should_approve(self, tool_name: str, arguments: str, risk_level: RiskLevel) -> bool:
        approved = self.callback(tool_name, arguments, risk_level.value)

        if self.audit_trail:
            event = "tool_approved" if approved else "tool_denied"
            self.audit_trail.record(
                AuditEntry(
                    timestamp=time.time(),
                    event_type=event,
                    tool_name=tool_name,
                    arguments=arguments[:500],
                    result="approved" if approved else "denied",
                    risk_level=risk_level.value,
                    details=f"Callback approval: {'yes' if approved else 'no'}",
                )
            )

        return approved
