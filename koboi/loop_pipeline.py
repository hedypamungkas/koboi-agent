"""koboi/loop_pipeline.py -- ToolExecutionPipeline: shared tool execution logic.

Encapsulates the 8-step tool execution flow used by both AgentCore.run()
and AgentCore.run_stream(), eliminating ~50 lines of duplication.
"""

from __future__ import annotations

import asyncio
import logging as _logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from koboi.types import ToolCall, RiskLevel
from koboi.modes import AgentMode

if TYPE_CHECKING:
    from koboi.guardrails.rate_limiter import RateLimiter
    from koboi.guardrails.approval import ApprovalHandler
    from koboi.tools.registry import ToolRegistry
    from koboi.memory import ConversationMemory
    from koboi.hooks.chain import HookChain
    from koboi.logger import AgentLogger
    from koboi.modes import ModeManager

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
    """Encapsulates the tool execution flow: rate limit -> risk -> approval -> hooks -> execute -> record.

    Used by both run() and run_stream() to avoid duplicating the 8-step pipeline.
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

    def _log(self, msg: str) -> None:
        if self.verbose:
            _log.debug(msg)

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
        self._log(f"tool: {tc.name}({tc.arguments})")

        is_yolo = self.mode_manager is not None and self.mode_manager.current_mode == AgentMode.YOLO

        # 1. Rate limiter check (skipped in YOLO mode)
        if not is_yolo and self.rate_limiter:
            rl_result = self.rate_limiter.check(tc.name)
            if not rl_result.passed:
                self._log(f"Rate limited: {rl_result.reason}")
                self._audit("rate_limit", tool_name=tc.name, details=rl_result.reason)
                self.memory.add_tool_result(tc.id, f"Error: {rl_result.reason}")
                if on_event:
                    on_event(
                        "tool_result",
                        {
                            "tool_name": tc.name,
                            "tool_call_id": tc.id,
                            "result": f"Error: {rl_result.reason}",
                        },
                    )
                return ToolPipelineResult(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    result=f"Error: {rl_result.reason}",
                    skipped=True,
                    skip_reason="rate_limit",
                )
            # Record immediately after check passes so subsequent checks
            # see the correct count (prevents off-by-one burst over-limit).
            self.rate_limiter.record(tc.name)

        # 2. Risk level check
        risk = self.tools.get_risk_level(tc.name) or RiskLevel.SAFE

        # 3. Approval handler check (skipped in YOLO mode)
        if not is_yolo and self.approval_handler:
            if asyncio.iscoroutinefunction(self.approval_handler.should_approve):
                approved = await self.approval_handler.should_approve(tc.name, tc.arguments, risk)
            else:
                approved = self.approval_handler.should_approve(tc.name, tc.arguments, risk)
            if not approved:
                self._log(f"Tool denied: {tc.name}")
                self._audit(
                    "tool_denied",
                    tool_name=tc.name,
                    arguments=tc.arguments[:200],
                    risk_level=risk.value,
                    details="Denied by human",
                )
                self.memory.add_tool_result(tc.id, "Error: Tool execution denied by user")
                if on_event:
                    on_event(
                        "tool_result",
                        {
                            "tool_name": tc.name,
                            "tool_call_id": tc.id,
                            "result": "Error: Denied by user",
                        },
                    )
                return ToolPipelineResult(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    result="Error: Denied by user",
                    skipped=True,
                    skip_reason="denied",
                )

        # 4. PRE_TOOL_USE hook
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

            # 4b. Policy abort check (always enforced, even in YOLO mode)
            if pre_ctx.abort:
                reason = pre_ctx.inject_message or "Blocked by policy"
                self._log(f"Tool aborted by policy: {tc.name}")
                self._audit("policy_denied", tool_name=tc.name, arguments=tc.arguments[:200], details=reason)
                self.memory.add_tool_result(tc.id, f"Error: {reason}")
                if on_event:
                    on_event(
                        "tool_result",
                        {
                            "tool_name": tc.name,
                            "tool_call_id": tc.id,
                            "result": f"Error: {reason}",
                        },
                    )
                return ToolPipelineResult(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    result=f"Error: {reason}",
                    skipped=True,
                    skip_reason="policy_denied",
                )

            # 5. Mode block check (skipped in YOLO mode)
            if not is_yolo and pre_ctx.metadata.get("mode_blocked"):
                reason = pre_ctx.metadata.get("mode_block_reason", "Blocked by current mode")
                self._log(f"Mode blocked: {tc.name}")
                self.memory.add_tool_result(tc.id, f"Error: {reason}")
                if on_event:
                    on_event(
                        "tool_result",
                        {
                            "tool_name": tc.name,
                            "tool_call_id": tc.id,
                            "result": f"Error: {reason}",
                        },
                    )
                return ToolPipelineResult(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    result=f"Error: {reason}",
                    skipped=True,
                    skip_reason="mode_blocked",
                )

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
