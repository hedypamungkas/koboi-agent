"""koboi/hooks/guardrail_hook.py -- Optional hook for input/output guardrail checking.

Applies input validation at PRE_INPUT and output checking at POST_OUTPUT.

NOTE: This hook is NOT registered by default. The agent loop checks guardrails
directly (see loop.py:_validate_input / _process_output). This class exists as
an alternative integration path for users who prefer to run guardrails through
the hook system. To use it, add it to your hook chain manually.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from koboi.hooks.chain import Hook, HookContext, HookEvent
from koboi.types import GuardrailResult

if TYPE_CHECKING:
    from koboi.guardrails import InputGuardrail, OutputGuardrail


class GuardrailHook(Hook):
    """Hook for input checking at PRE_INPUT and output checking at POST_OUTPUT.

    Uses InputGuardrail and OutputGuardrail to validate data flowing through
    the agent loop.
    """

    def __init__(
        self,
        input_guardrail: InputGuardrail | None = None,
        output_guardrail: OutputGuardrail | None = None,
    ):
        self.input_guardrail = input_guardrail
        self.output_guardrail = output_guardrail

    def handles(self) -> list[HookEvent]:
        events = []
        if self.input_guardrail:
            events.append(HookEvent.PRE_INPUT)
        if self.output_guardrail:
            events.append(HookEvent.POST_OUTPUT)
        return events

    async def execute(self, ctx: HookContext) -> HookContext:
        if ctx.event == HookEvent.PRE_INPUT and self.input_guardrail:
            return await self._check_input(ctx)
        elif ctx.event == HookEvent.POST_OUTPUT and self.output_guardrail:
            return await self._check_output(ctx)
        return ctx

    async def _check_input(self, ctx: HookContext) -> HookContext:
        # Extract user input from messages if available
        user_input = ""
        if ctx.messages:
            for msg in reversed(ctx.messages):
                if msg.get("role") == "user":
                    user_input = msg.get("content", "")
                    break

        if not user_input:
            return ctx

        result: GuardrailResult = await self.input_guardrail.check(user_input)
        ctx.metadata["input_guardrail_result"] = {
            "passed": result.passed,
            "reason": result.reason,
            "action": result.action,
        }

        if not result.passed:
            ctx.abort = True
            ctx.metadata["guardrail_blocked"] = True
            ctx.inject_message = f"Input blocked: {result.reason}"

        return ctx

    async def _check_output(self, ctx: HookContext) -> HookContext:
        output = ""
        if ctx.llm_response:
            output = getattr(ctx.llm_response, "content", "") or ""

        if not output:
            return ctx

        result: GuardrailResult = await self.output_guardrail.check(output)
        ctx.metadata["output_guardrail_result"] = {
            "passed": result.passed,
            "reason": result.reason,
            "action": result.action,
        }

        if not result.passed:
            # For output, we warn but don't abort -- the metadata records it
            ctx.metadata["output_warning"] = result.reason

        return ctx
