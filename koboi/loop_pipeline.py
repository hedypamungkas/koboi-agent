"""koboi/loop_pipeline.py -- ToolExecutionPipeline: shared tool execution logic.

Encapsulates the tool execution flow used by both AgentCore.run()
and AgentCore.run_stream(), eliminating ~50 lines of duplication.

Approval resolution (M0) is unified through ``_resolve_approval`` so the
Trust DB fast-path (auto-allow / deny rule), the risk-based approval handler
(step 3), and the ``policy_needs_confirmation`` confirmation (step 4c) share
one code path, one audit point, and a single "already prompted" guard that
prevents double-prompting the user.
"""

from __future__ import annotations

import asyncio
import logging as _logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from collections.abc import Callable

from koboi.guardrails.approval_types import ApprovalOutcome
from koboi.modes import AgentMode
from koboi.types import RiskLevel, ToolCall

if TYPE_CHECKING:
    from koboi.guardrails.rate_limiter import RateLimiter
    from koboi.guardrails.approval import ApprovalHandler
    from koboi.tools.registry import ToolRegistry
    from koboi.memory import ConversationMemory
    from koboi.hooks.chain import HookChain
    from koboi.logger import AgentLogger
    from koboi.modes import ModeManager
    from koboi.trust import TrustStore

_log = _logging.getLogger("koboi.pipeline")

# Callback type for yielding events during streaming
EventCallback = Callable[[str, dict], None]


@dataclass
class ToolPipelineResult:
    """Result of processing a single tool call through the pipeline."""

    tool_call_id: str
    tool_name: str
    result: str
    skipped: bool = False
    skip_reason: str = ""


class ToolExecutionPipeline:
    """Encapsulates the tool execution flow: rate limit -> risk -> hooks (policy/mode) -> approval -> execute -> record.

    Policy-abort and mode-block run BEFORE approval so an approved tool is never
    later mode-blocked (wasted prompt) and a trust-DB allow cannot bypass
    chat/plan mode-blocking. Used by both run() and run_stream().
    """

    def __init__(
        self,
        tools: ToolRegistry,
        memory: ConversationMemory,
        rate_limiter: RateLimiter | None = None,
        approval_handler: ApprovalHandler | None = None,
        hook_chain: HookChain | None = None,
        logger: AgentLogger | None = None,
        verbose: bool = False,
        audit_fn: Callable[..., None] | None = None,
        mode_manager: ModeManager | None = None,
        trust_db: TrustStore | None = None,
    ):
        self.tools = tools
        self.memory = memory
        self.rate_limiter = rate_limiter
        self.approval_handler = approval_handler
        self.hooks = hook_chain
        self.logger = logger
        self.verbose = verbose
        self._audit = audit_fn or (lambda *a, **kw: None)
        self.mode_manager = mode_manager
        self.trust_db = trust_db

    def _log(self, msg: str) -> None:
        if self.verbose:
            _log.debug(msg)

    def _deny_or_skip(
        self,
        tc: ToolCall,
        message: str,
        skip_reason: str,
        on_event: EventCallback | None = None,
    ) -> ToolPipelineResult:
        """Common deny/skip tail: persist the tool result + emit event + build result.

        ``message`` is used for both conversation memory and the streamed result,
        so there is one canonical string per deny path.
        """
        self.memory.add_tool_result(tc.id, message)
        if on_event:
            on_event(
                "tool_result",
                {"tool_name": tc.name, "tool_call_id": tc.id, "result": message},
            )
        return ToolPipelineResult(
            tool_call_id=tc.id,
            tool_name=tc.name,
            result=message,
            skipped=True,
            skip_reason=skip_reason,
        )

    def _deny_tool(
        self,
        tc: ToolCall,
        risk: RiskLevel,
        skip_reason: str,
        details: str,
        on_event: EventCallback | None = None,
    ) -> ToolPipelineResult:
        """Unified tool-denial: log + audit(``tool_denied``) + deny/skip tail.

        Shared by the risk-based denial (step 3) and the policy-confirm denial
        (step 4c) so memory/result wording stays identical across both.
        """
        self._log(f"Tool denied ({skip_reason}): {tc.name}")
        self._audit(
            "tool_denied",
            tool_name=tc.name,
            arguments=tc.arguments[:200],
            risk_level=risk.value,
            details=details,
        )
        return self._deny_or_skip(tc, "Error: Tool execution denied by user", skip_reason, on_event)

    async def _resolve_approval(
        self,
        tc: ToolCall,
        risk: RiskLevel,
        policy_reason: str | None = None,
        already_prompted: bool = False,
    ) -> ApprovalOutcome:
        """Unified approval resolution: trust_db -> handler -> outcome.

        Order of precedence:
          1. Trust DB auto-allow  -> proceed, no prompt (``skipped_via_trust``).
          2. Trust DB deny rule   -> deny, no prompt (``denied``).
          3. No approval handler  -> proceed, inert (``no_handler``). This makes
             ``policy_needs_confirmation`` a no-op for default configs (Q1).
          4. ``already_prompted`` -> proceed, inert (``inert``) -- guards against
             double-prompting when called again after step 3 already asked.
          5. Otherwise            -> await the handler (sync or async), audit via
             the handler, return ``approved``/``denied`` with ``prompted=True``.

        Handler errors are fail-closed (deny).
        """
        # 1-2. Trust DB fast-path.
        if self.trust_db is not None:
            trust_decision = self.trust_db.should_auto_approve(tc.name, risk, tc.arguments)
            if trust_decision.auto_approve:
                self._audit(
                    "tool_approved",
                    tool_name=tc.name,
                    arguments=tc.arguments[:200],
                    risk_level=risk.value,
                    details=trust_decision.reason,
                )
                return ApprovalOutcome(
                    proceed=True,
                    reason="skipped_via_trust",
                    trust_rule=trust_decision.matched_rule,
                    audit_details=trust_decision.reason,
                )
            if trust_decision.matched_rule is not None:
                # An explicit deny rule matched -- hard deny.
                return ApprovalOutcome(
                    proceed=False,
                    reason="denied",
                    trust_rule=trust_decision.matched_rule,
                    audit_details=trust_decision.reason,
                )

        # 3. No handler configured -> inert proceed.
        if self.approval_handler is None:
            return ApprovalOutcome(proceed=True, reason="no_handler")

        # 4. Already prompted earlier in this tool call -> inert proceed.
        if already_prompted:
            return ApprovalOutcome(proceed=True, reason="inert")

        # 5. Ask the handler.
        reason = policy_reason or "risk-based approval"
        try:
            if asyncio.iscoroutinefunction(self.approval_handler.should_approve):
                approved = await self.approval_handler.should_approve(tc.name, tc.arguments, risk)
            else:
                approved = self.approval_handler.should_approve(tc.name, tc.arguments, risk)
        except Exception as exc:
            # Fail-closed: a broken handler must never silently let a tool run.
            self._log(f"Approval handler error for {tc.name}: {exc}")
            return ApprovalOutcome(proceed=False, reason="denied", audit_details=f"handler error: {exc}")

        return ApprovalOutcome(
            proceed=approved,
            reason="approved" if approved else "denied",
            prompted=True,
            audit_details=reason,
        )

    async def execute_tool_call(
        self,
        tc: ToolCall,
        iteration: int,
        on_event: EventCallback | None = None,
    ) -> ToolPipelineResult:
        """Execute a single tool call through the full pipeline.

        Args:
            tc: The tool call to execute.
            iteration: Current loop iteration number.
            on_event: Optional callback(event_type, data) for streaming events.

        Returns:
            ToolPipelineResult with the tool's output or skip reason.
        """
        self._log(f"tool: {tc.name}({tc.arguments[:100]})")  # 16.17: truncate args in logs

        is_yolo = self.mode_manager is not None and self.mode_manager.current_mode == AgentMode.YOLO

        # 1. Rate limiter check (skipped in YOLO mode)
        if not is_yolo and self.rate_limiter:
            rl_result = self.rate_limiter.check(tc.name)
            if not rl_result.passed:
                self._log(f"Rate limited: {rl_result.reason}")
                self._audit("rate_limit", tool_name=tc.name, details=rl_result.reason)
                return self._deny_or_skip(tc, f"Error: {rl_result.reason}", "rate_limit", on_event)
            # Record immediately after check passes so subsequent checks
            # see the correct count (prevents off-by-one burst over-limit).
            self.rate_limiter.record(tc.name)

        # 2. Risk level check
        risk = self.tools.get_risk_level(tc.name) or RiskLevel.SAFE

        # 3. PRE_TOOL_USE hook (emit) + policy/mode gates. These run BEFORE approval
        #    so that: (a) an approved tool is never later mode-blocked (wasted prompt),
        #    and (b) a trust-DB allow rule cannot bypass chat/plan mode-blocking.
        approval_prompted = False
        pre_ctx = None
        if self.hooks:
            from koboi.hooks.chain import HookContext, HookEvent

            pre_ctx = HookContext(
                event=HookEvent.PRE_TOOL_USE,
                agent=None,
                iteration=iteration,
                tool_name=tc.name,
                tool_arguments=tc.arguments,
            )
            pre_ctx = await self.hooks.emit(pre_ctx)
            for msg in pre_ctx.inject_messages:
                self.memory.add_context_message(msg, label="hook_inject")

            # 3a. Policy abort (always enforced, even in YOLO mode). Runs before
            #     approval so a policy-denied tool never wastes an approval prompt.
            if pre_ctx.abort:
                reason = pre_ctx.inject_message or "Blocked by policy"
                self._log(f"Tool aborted by policy: {tc.name}")
                self._audit("policy_denied", tool_name=tc.name, arguments=tc.arguments[:200], details=reason)
                return self._deny_or_skip(tc, f"Error: {reason}", "policy_denied", on_event)

            # 3b. Mode block check (must precede approval; skipped in YOLO mode).
            if not is_yolo and pre_ctx.metadata.get("mode_blocked"):
                reason = pre_ctx.metadata.get("mode_block_reason", "Blocked by current mode")
                self._log(f"Mode blocked: {tc.name}")
                return self._deny_or_skip(tc, f"Error: {reason}", "mode_blocked", on_event)

        # 4. Approval resolution (trust DB fast-path + risk-based handler) -- unified.
        if not is_yolo:
            outcome = await self._resolve_approval(tc, risk)
            if not outcome.proceed:
                return self._deny_tool(tc, risk, "denied", outcome.audit_details or "Denied by human", on_event)
            approval_prompted = outcome.prompted or outcome.reason == "skipped_via_trust"

        # 4c. Policy CONFIRM (M0): when policy asks for confirmation, route to the
        #     approval gate -- but only if a handler is configured AND the risk path
        #     (step 4) did not already resolve it. Inert otherwise (Q1).
        if (
            pre_ctx is not None
            and not is_yolo
            and not approval_prompted
            and self.approval_handler is not None
            and pre_ctx.metadata.get("policy_needs_confirmation")
        ):
            policy_reason = pre_ctx.metadata.get("policy_reason", "Policy requires confirmation")
            confirm_outcome = await self._resolve_approval(
                tc, risk, policy_reason=policy_reason, already_prompted=approval_prompted
            )
            if not confirm_outcome.proceed:
                return self._deny_tool(tc, risk, "policy_denied", f"Policy confirm denied: {policy_reason}", on_event)

        # 6. Execute tool
        tool_result = await self.tools.execute(tc.name, tc.arguments)

        # 7. POST_TOOL_USE hook
        if self.hooks:
            from koboi.hooks.chain import HookContext, HookEvent

            post_ctx = HookContext(
                event=HookEvent.POST_TOOL_USE,
                agent=None,
                iteration=iteration,
                tool_name=tc.name,
                tool_arguments=tc.arguments,
                tool_result=tool_result,
            )
            post_ctx = await self.hooks.emit(post_ctx)
            for msg in post_ctx.inject_messages:
                self.memory.add_context_message(msg, label="hook_inject")

        # 8. Record result
        self._log(f"tool result: {tool_result[:200]}")
        self.memory.add_tool_result(tc.id, tool_result)

        self._audit(
            "tool_execute",
            tool_name=tc.name,
            arguments=tc.arguments[:200],
            result=tool_result[:200],
            risk_level=risk.value,
        )

        if on_event:
            on_event(
                "tool_result",
                {
                    "tool_name": tc.name,
                    "tool_call_id": tc.id,
                    "result": tool_result,
                },
            )

        return ToolPipelineResult(
            tool_call_id=tc.id,
            tool_name=tc.name,
            result=tool_result,
        )
