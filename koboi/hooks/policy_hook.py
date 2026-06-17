"""koboi/hooks/policy_hook.py -- Hook for policy engine checking at PRE_TOOL_USE.

Evaluates tool calls against the policy engine before execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from koboi.hooks.chain import Hook, HookContext, HookEvent
from koboi.types import RiskLevel

if TYPE_CHECKING:
    from koboi.harness.policy import PolicyEngine
    from koboi.harness.policy_audit import PolicyAuditLog


class PolicyHook(Hook):
    """Hook for policy engine checking at PRE_TOOL_USE.

    Evaluates each tool call against the configured policy rules.
    DENY results abort the tool call. CONFIRM results flag for
    user confirmation (via metadata).
    """

    priority = 25

    def __init__(
        self,
        policy_engine: PolicyEngine,
        risk_lookup: dict[str, RiskLevel] | None = None,
        default_risk: RiskLevel = RiskLevel.SAFE,
        audit_log: PolicyAuditLog | None = None,
    ):
        self.policy_engine = policy_engine
        self.risk_lookup = risk_lookup or {}
        self.default_risk = default_risk
        self.audit_log = audit_log

    def handles(self) -> list[HookEvent]:
        return [HookEvent.PRE_TOOL_USE]

    async def execute(self, ctx: HookContext) -> HookContext:
        if not ctx.tool_name:
            return ctx

        # Determine risk level for the tool
        risk_level = self.risk_lookup.get(ctx.tool_name, self.default_risk)
        arguments = ctx.tool_arguments or ""

        # Evaluate policy
        decision = self.policy_engine.evaluate(ctx.tool_name, arguments, risk_level)

        # Store decision in metadata
        ctx.metadata["policy_decision"] = {
            "action": decision.action.value,
            "matched_rule": decision.matched_rule,
            "reason": decision.reason,
        }

        # Log to audit trail
        if self.audit_log is not None:
            self.audit_log.log(
                tool_name=ctx.tool_name,
                arguments=arguments,
                decision=decision.action.value,
                rule=decision.matched_rule or "",
                risk_level=risk_level.value if hasattr(risk_level, "value") else str(risk_level),
            )

        if decision.action.value == "deny":
            ctx.abort = True
            ctx.inject_message = f"Policy denied: {decision.reason}"
        elif decision.action.value == "confirm":
            ctx.metadata["policy_needs_confirmation"] = True
            ctx.metadata["policy_reason"] = decision.reason

        return ctx
