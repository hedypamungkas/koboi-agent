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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from collections.abc import Callable

from koboi.guardrails.approval_types import ApprovalOutcome
from koboi.harness.utils import parse_exit_code
from koboi.modes import AgentMode
from koboi.types import RiskLevel, ToolCall, ToolExecOutcome

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

# Self-healing P0-D: actionable hints appended to an errored tool_result. The
# leading "Error:" prefix from ToolRegistry.execute_outcome() is preserved so
# internal callers that string-match it (orchestrator node-failure detection)
# keep working; this only adds guidance for the LLM.
_ERROR_HINTS: dict[str, str] = {
    "tool_not_found": "This tool does not exist; pick an available tool.",
    "invalid_args": "The arguments were invalid; fix the JSON/types and retry.",
    "timeout": "The tool timed out; simplify the request or retry.",
    "execution_error": "Review the arguments and retry, or try a different approach.",
}


def _format_tool_error(outcome: ToolExecOutcome, idempotent: bool) -> str:
    """Append an actionable, prefix-preserving hint to an errored tool result."""
    hint = _ERROR_HINTS.get(outcome.error_kind or "", _ERROR_HINTS["execution_error"])
    suffix = " (warning: this tool has side effects; a retry will re-run it)" if not idempotent else ""
    return f"{outcome.content} {hint}{suffix}"


@dataclass
class ToolPipelineResult:
    """Result of processing a single tool call through the pipeline."""

    tool_call_id: str
    tool_name: str
    result: str
    skipped: bool = False
    skip_reason: str = ""
    # Self-healing P0-D: structured error signal (additive; the P1 ReflectionHook
    # keys on this instead of fragile string-matching). ``errored`` = the tool ran
    # and failed; denied/skipped tools set ``error_kind`` but stay ``skipped=True``
    # (they never executed). ``idempotent`` is carried for P1 retry decisions.
    errored: bool = False
    error_kind: str | None = None
    idempotent: bool = True
    # Wave 3 parallel execution: hook inject_messages collected (not written to
    # memory) when the call ran with defer_record=True -- the loop replays them
    # in original call order AFTER the batch's tool results.
    injected_context: list[str] = field(default_factory=list)


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
        defer_record: bool = False,
        injected_context: list[str] | None = None,
    ) -> ToolPipelineResult:
        """Common deny/skip tail: persist the tool result + emit event + build result.

        ``message`` is used for both conversation memory and the streamed result,
        so there is one canonical string per deny path. With ``defer_record``
        (Wave 3 parallel batches) the memory write is skipped -- the loop
        replays ``pr.result`` in original call order.
        """
        if not defer_record:
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
            # Self-healing P0-D: classify the deny/skip so P1 reflection can route
            # (skip_reason strings are already a clean taxonomy: rate_limit /
            # policy_denied / mode_blocked / denied). skipped=True stays authoritative.
            error_kind=skip_reason,
            injected_context=list(injected_context or ()),
        )

    def _deny_tool(
        self,
        tc: ToolCall,
        risk: RiskLevel,
        skip_reason: str,
        details: str,
        on_event: EventCallback | None = None,
        defer_record: bool = False,
        injected_context: list[str] | None = None,
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
        return self._deny_or_skip(
            tc,
            "Error: Tool execution denied by user",
            skip_reason,
            on_event,
            defer_record=defer_record,
            injected_context=injected_context,
        )

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
        defer_record: bool = False,
    ) -> ToolPipelineResult:
        """Execute a single tool call through the full pipeline.

        Args:
            tc: The tool call to execute.
            iteration: Current loop iteration number.
            on_event: Optional callback(event_type, data) for streaming events.
            defer_record: Wave 3 parallel batches -- skip ALL memory writes
                (tool result + hook inject_messages); the caller replays them
                in original call order (concurrent pipeline runs would append
                in completion order, breaking Anthropic tool_result pairing
                and replay determinism). Default False = byte-identical
                behavior for every existing caller.

        Returns:
            ToolPipelineResult with the tool's output or skip reason.
        """
        self._log(f"tool: {tc.name}({tc.arguments[:100]})")  # 16.17: truncate args in logs

        injected: list[str] = []

        def _record_injects(msgs) -> None:
            for msg in msgs:
                if defer_record:
                    injected.append(msg)
                else:
                    self.memory.add_context_message(msg, label="hook_inject")

        is_yolo = self.mode_manager is not None and self.mode_manager.current_mode == AgentMode.YOLO

        # 1. Rate limiter check (skipped in YOLO mode)
        if not is_yolo and self.rate_limiter:
            rl_result = self.rate_limiter.check(tc.name)
            if not rl_result.passed:
                self._log(f"Rate limited: {rl_result.reason}")
                self._audit("rate_limit", tool_name=tc.name, details=rl_result.reason)
                return self._deny_or_skip(
                    tc, f"Error: {rl_result.reason}", "rate_limit", on_event, defer_record, injected
                )
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
            _record_injects(pre_ctx.inject_messages)

            # 3a. Policy abort (always enforced, even in YOLO mode). Runs before
            #     approval so a policy-denied tool never wastes an approval prompt.
            if pre_ctx.abort:
                reason = pre_ctx.inject_message or "Blocked by policy"
                self._log(f"Tool aborted by policy: {tc.name}")
                self._audit("policy_denied", tool_name=tc.name, arguments=tc.arguments[:200], details=reason)
                return self._deny_or_skip(tc, f"Error: {reason}", "policy_denied", on_event, defer_record, injected)

            # 3b. Mode block check (must precede approval; skipped in YOLO mode).
            if not is_yolo and pre_ctx.metadata.get("mode_blocked"):
                reason = pre_ctx.metadata.get("mode_block_reason", "Blocked by current mode")
                self._log(f"Mode blocked: {tc.name}")
                return self._deny_or_skip(tc, f"Error: {reason}", "mode_blocked", on_event, defer_record, injected)

        # 4. Approval resolution (trust DB fast-path + risk-based handler) -- unified.
        if not is_yolo:
            outcome = await self._resolve_approval(tc, risk)
            if not outcome.proceed:
                return self._deny_tool(
                    tc,
                    risk,
                    "denied",
                    outcome.audit_details or "Denied by human",
                    on_event,
                    defer_record=defer_record,
                    injected_context=injected,
                )
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
                return self._deny_tool(
                    tc,
                    risk,
                    "policy_denied",
                    f"Policy confirm denied: {policy_reason}",
                    on_event,
                    defer_record=defer_record,
                    injected_context=injected,
                )

        # 6. Execute tool (self-healing P0-D: structured outcome -> actionable msg)
        exec_outcome = await self.tools.execute_outcome(tc.name, tc.arguments)
        tool_result = exec_outcome.content
        errored = exec_outcome.errored
        error_kind = exec_outcome.error_kind
        td = self.tools.get_definition(tc.name)
        idempotent = td.idempotent if td is not None else True
        if errored:
            tool_result = _format_tool_error(exec_outcome, idempotent)
        elif parse_exit_code(tool_result) not in (None, 0):
            # Shell commands report failure as a normal "[exit code: N]" string
            # (the @tool contract is -> str, so run_shell can't raise on N != 0).
            # Lift it into the structured signal so ReflectionHook /
            # FailureClassifierHook see real command failures -- but keep the
            # output as-is: the command's own output IS the diagnostic.
            errored = True
            error_kind = "command_failed"

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
            # Self-healing P2a: surface P0-D's structured error signal onto the ctx so
            # FailureClassifierHook can tag failure_class without string-matching.
            post_ctx.metadata["tool_error_kind"] = error_kind
            post_ctx = await self.hooks.emit(post_ctx)
            _record_injects(post_ctx.inject_messages)
            # Honor a hook's modified_tool_result (e.g. a CommandHook rewriting the
            # tool output). Without this, the local `tool_result` below would ignore
            # any POST_TOOL_USE mutation -- memory/audit/on_event/return would all
            # use the original. Only override when the hook set a (new) value.
            if post_ctx.tool_result is not None:
                tool_result = post_ctx.tool_result

            # DoomLoopHook detects the pattern during POST_TOOL_USE and sets a
            # metadata flag, but it cannot re-emit DOOM_LOOP_DETECTED itself (hooks
            # react; the emitter emits). Fan it out here so subscribers that listen
            # on DOOM_LOOP_DETECTED -- AuditHook, TelemetryHook, LangfuseTracingHook,
            # NotificationHook -- actually fire. Without this, doom-loop side-channel
            # observability is silently inert even though recovery injection works.
            if post_ctx.metadata.get("doom_loop_detected"):
                doom_ctx = HookContext(
                    event=HookEvent.DOOM_LOOP_DETECTED,
                    agent=None,
                    iteration=iteration,
                    tool_name=tc.name,
                    tool_arguments=tc.arguments,
                    tool_result=tool_result,
                    metadata=post_ctx.metadata,
                )
                await self.hooks.emit(doom_ctx)

        # 8. Record result
        self._log(f"tool result: {tool_result[:200]}")
        if not defer_record:
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
            errored=errored,
            error_kind=error_kind,
            idempotent=idempotent,
            injected_context=injected,
        )
