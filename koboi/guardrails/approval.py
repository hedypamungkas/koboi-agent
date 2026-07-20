"""Human approval for destructive tools.

In CLI, approval is a y/n prompt. For programmatic usage,
can be skipped with auto_approve=True or overridden with custom callback.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable
from collections.abc import Callable
from typing import TYPE_CHECKING
from uuid import uuid4

from koboi.guardrails.approval_types import ApprovalCallback, ApprovalRequest, ApprovalResponse
from koboi.types import AuditEntry, RiskLevel

if TYPE_CHECKING:
    from koboi.guardrails.audit import AuditTrail
    from koboi.trust import TrustDatabase

_logger = logging.getLogger(__name__)

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

    #: Set by subclasses that support auditing (CLI/Callback/Async/TUI). Stays
    #: ``None`` on the bare base / in handlers with no trail attached.
    audit_trail: AuditTrail | None = None

    def should_approve(self, tool_name: str, arguments: str, risk_level: RiskLevel) -> bool | Awaitable[bool]:
        # May be sync (bool) or async (Awaitable[bool]); the execution pipeline
        # awaits when iscoroutinefunction(should_approve) is True.
        return risk_level != RiskLevel.DESTRUCTIVE

    def _audit(
        self,
        tool_name: str,
        arguments: str,
        risk_level: RiskLevel,
        approved: bool,
        detail: str,
        source: str,
    ) -> None:
        """Record an approval decision to the attached audit trail (no-op if none).

        ``source`` is the handler-specific prefix (e.g. "Human approval via CLI")
        so the resulting ``details`` reads ``"{source}: {detail}"`` and each
        subclass preserves its historical audit wording.
        """
        if self.audit_trail is not None:
            self.audit_trail.record(
                AuditEntry(
                    timestamp=time.time(),
                    event_type="tool_approved" if approved else "tool_denied",
                    tool_name=tool_name,
                    arguments=arguments[:500],
                    result="approved" if approved else "denied",
                    risk_level=risk_level.value,
                    details=f"{source}: {detail}",
                )
            )


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

        self._audit(
            tool_name,
            arguments,
            risk_level,
            approved,
            "yes" if approved else "no",
            source="Human approval via CLI",
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

        self._audit(
            tool_name,
            arguments,
            risk_level,
            approved,
            "yes" if approved else "no",
            source="Callback approval",
        )

        return approved


class AsyncCallbackApprovalHandler(ApprovalHandler):
    """Non-blocking approval backed by an async ``callback`` (REST/SSE friendly).

    Modeled on ``TUIApprovalHandler`` (``koboi/tui/approval.py``) but replaces
    Textual's message bus with a caller-supplied async ``callback`` that receives
    an :class:`ApprovalRequest` and resolves an :class:`ApprovalResponse`.

    The tool-execution pipeline already ``await``s async ``should_approve``
    implementations (``loop_pipeline.py``), so this handler slots in unchanged.

    Use cases: M0 unit tests; M2 REST/SSE server (the callback enqueues a
    pending approval, emits a ``PendingApprovalEvent`` on the SSE stream, and
    awaits an ``asyncio.Future`` resolved by ``POST /approve``).
    """

    def __init__(
        self,
        callback: ApprovalCallback,
        trust_db: TrustDatabase | None = None,
        audit_trail: AuditTrail | None = None,
        timeout: float = 120.0,
        auto_approve_safe: bool = True,
    ) -> None:
        self._callback = callback
        self._trust_db = trust_db
        self.audit_trail = audit_trail
        self._timeout = timeout
        #: SAFE tools (read-only / side-effect-free) auto-run without HITL;
        #: only MODERATE/DESTRUCTIVE prompt the human. Matches the base-class
        #: intent ("deny destructive, allow others") and keeps HITL meaningful.
        self._auto_approve_safe = auto_approve_safe

    async def should_approve(self, tool_name: str, arguments: str, risk_level: RiskLevel) -> bool:
        # 0. SAFE tools (read-only / side-effect-free) auto-approve — no HITL
        #    prompt for calculator/memory/search/reads. Only MODERATE/DESTRUCTIVE
        #    reach the human. Auto-allow is auditable so the trail stays complete.
        if self._auto_approve_safe and risk_level == RiskLevel.SAFE:
            self._audit(tool_name, arguments, risk_level, True, "auto-approve (safe)", source="Async callback approval")
            return True
        # 1. Trust DB fast-path (auto-allow). Auto-deny is left to the pipeline's
        #    own trust consultation; here we only short-circuit on an allow rule.
        if self._trust_db:
            trust_decision = self._trust_db.should_auto_approve(tool_name, risk_level, arguments)
            if trust_decision.auto_approve:
                self._audit(
                    tool_name, arguments, risk_level, True, trust_decision.reason, source="Async callback approval"
                )
                return True

        # 2. Delegate the actual prompt to the caller-supplied async callback;
        #    enforce a timeout (deny on timeout) and fail-closed on error.
        request = ApprovalRequest(
            tool_name=tool_name,
            arguments=arguments,
            risk_level=risk_level,
            reason="risk-based approval",
            approval_id=f"ap_{uuid4().hex[:24]}",
        )
        try:
            response: ApprovalResponse = await asyncio.wait_for(self._callback(request), timeout=self._timeout)
        except asyncio.TimeoutError:
            response = ApprovalResponse(approved=False)
        except Exception as exc:  # fail-closed: never silently proceed
            _logger.warning("Async approval callback error for %s: %s", tool_name, exc)
            response = ApprovalResponse(approved=False)

        # 3. Persist "always" decisions so future calls auto-approve.
        if response.always_allow and self._trust_db:
            self._trust_db.record_decision(
                tool_name=tool_name,
                risk_level=risk_level,
                decision="allow" if response.approved else "deny",
                always=True,
                arguments=arguments,
            )

        # 4. Audit.
        self._audit(
            tool_name,
            arguments,
            risk_level,
            response.approved,
            "always_allow" if response.always_allow else "one_shot",
            source="Async callback approval",
        )
        return response.approved


class AutonomousApprovalHandler(ApprovalHandler):
    """Autonomous-mode handler (M4): safe/moderate auto-approve; destructive → Trust DB or deny.

    No human interaction (no pause, no Future). Destructive tools without a
    Trust DB allow-rule are denied by default — anti prompt-injection safeguard
    for jobs that run without human review.

    ``auto_approve_tools`` is a job-scoped allowlist of tool names to
    auto-approve regardless of risk (e.g. in-workdir ``write_file``/``delete_file``
    for autonomous jobs). Containment is enforced by the restricted sandbox,
    which rejects out-of-workdir paths at execution time (after approval), so
    this only lifts the gate -- it does NOT bypass filesystem containment. We
    use this instead of seeding a Trust-DB rule because the trust DB is shared
    across all pooled agents (one ``db_path``), so a persistent seeded rule
    would leak auto-approve to chat sessions.

    ``shell_allowlist`` (Wave 2, ``jobs.shell_allowlist``) is the run_shell
    analog: a job-scoped list of COMMAND glob patterns (``pytest*``,
    ``git commit*``) auto-approved without a Trust-DB wildcard -- so an
    unattended coding job can run its build/test/git commands while every
    non-matching command still falls through to the trust-db-or-deny flow.
    The hardcoded policy deny-list is re-checked first and always wins.
    """

    def __init__(
        self,
        trust_db: TrustDatabase | None = None,
        audit_trail: AuditTrail | None = None,
        auto_approve_tools: set[str] | None = None,
        shell_allowlist: list[str] | None = None,
    ) -> None:
        self._trust_db = trust_db
        self.audit_trail = audit_trail
        self._auto_approve_tools = set(auto_approve_tools or ())
        self._shell_allowlist = list(shell_allowlist or ())

    def _shell_allowlist_decision(self, arguments: str) -> tuple[bool, str] | None:
        """Match a run_shell command against the job allowlist.

        Returns (approved, audit_reason) when the allowlist decides, or None to
        fall through to the trust-db/deny flow (no pattern matched).
        """
        from fnmatch import fnmatch

        from koboi.harness.policy import _extract_command, check_command_blocked

        command = _extract_command(arguments)
        if not command:
            return None
        blocked = check_command_blocked(command)
        if blocked:
            # Hardcoded safety wins even over a matching pattern.
            return False, f"denied (job shell allowlist: {blocked})"
        for pattern in self._shell_allowlist:
            if fnmatch(command, pattern):
                return True, f"auto-approve (job shell allowlist: {pattern})"
        return None

    def should_approve(self, tool_name: str, arguments: str, risk_level: RiskLevel) -> bool:
        # Job-scoped allowlist first: auto-approve (e.g. in-workdir writes for
        # autonomous jobs). The restricted sandbox already confines these to the
        # workdir, so lifting the approval gate is safe.
        if tool_name in self._auto_approve_tools:
            self._audit(
                tool_name,
                arguments,
                risk_level,
                True,
                "auto-approve (job allowlist)",
                source="Autonomous",
            )
            return True
        # Wave 2: job-scoped shell COMMAND allowlist (glob on the command string).
        if tool_name == "run_shell" and self._shell_allowlist:
            decision = self._shell_allowlist_decision(arguments)
            if decision is not None:
                approved, reason = decision
                self._audit(tool_name, arguments, risk_level, approved, reason, source="Autonomous")
                return approved
        # Safe / moderate: allow (base behavior).
        if risk_level != RiskLevel.DESTRUCTIVE:
            return True
        # Destructive: check Trust DB; deny if no rule.
        if self._trust_db:
            trust_decision = self._trust_db.should_auto_approve(tool_name, risk_level, arguments)
            if trust_decision.auto_approve:
                self._audit(tool_name, arguments, risk_level, True, trust_decision.reason, source="Autonomous")
                return True
        self._audit(
            tool_name,
            arguments,
            risk_level,
            False,
            "denied (autonomous: no trust rule for destructive tool)",
            source="Autonomous",
        )
        return False
