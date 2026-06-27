"""koboi/loop.py -- AgentCore: async unified loop with built-in hook system."""

from __future__ import annotations

import logging as _logging
import time as _time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from koboi.events import (
    CompleteEvent,
    ErrorEvent,
    IterationEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from koboi.exceptions import (
    AgentAbortedError,
    AgentGuardrailError,
    AgentMaxIterationsError,
)
from koboi.guardrails.base import BaseGuardrail
from koboi.loop_pipeline import ToolExecutionPipeline
from koboi.memory import ConversationMemory
from koboi.tools.registry import ToolRegistry
from koboi.client import RetryClient
from koboi.tokens import estimate_tokens
from koboi.hooks.chain import HookEvent, HookChain, HookContext, AgentInfo
from koboi.types import ToolCall, AuditEntry, RunResult

if TYPE_CHECKING:
    from koboi.logger import AgentLogger
    from koboi.context.manager import ContextManager
    from koboi.rag.augmentation import AugmentationStrategy
    from koboi.guardrails.rate_limiter import RateLimiter
    from koboi.guardrails.audit import AuditTrail
    from koboi.guardrails.approval import ApprovalHandler
    from koboi.skills.registry import SkillRegistry
    from koboi.modes import ModeManager

_log = _logging.getLogger("koboi.loop")


def _extract_text(content: list) -> str:
    """Extract text portions from multimodal content blocks."""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return " ".join(parts)


SYSTEM_PROMPT_REACT = """\
You are an AI assistant that solves problems step by step.

Your workflow:
1. ANALYZE -- Understand what the user is asking
2. PLAN -- Think about the steps needed
3. EXECUTE -- Use available tools to get data
4. ANSWER -- After all information is gathered, provide a complete answer

Rules:
- NEVER fabricate data. Use tools to get real information.
- If no tool is available for a question, honestly say you cannot check it.
- Always explain your reasoning before and after using a tool."""


class AgentCore:
    """Core agent with async loop and built-in hook chain."""

    def __init__(
        self,
        client: RetryClient,
        memory: ConversationMemory | None = None,
        tools: ToolRegistry | None = None,
        max_iterations: int = 10,
        verbose: bool = False,
        logger: AgentLogger | None = None,
        system_prompt: str | None = None,
        context_manager: ContextManager | None = None,
        max_context_tokens: int = 8000,
        augmentation: AugmentationStrategy | None = None,
        input_guardrails: list[BaseGuardrail] | None = None,
        output_guardrails: list[BaseGuardrail] | None = None,
        rate_limiter: RateLimiter | None = None,
        audit_trail: AuditTrail | None = None,
        approval_handler: ApprovalHandler | None = None,
        skills: SkillRegistry | None = None,
        hook_chain: HookChain | None = None,
        mode_manager: ModeManager | None = None,
        # Backward-compatible singular kwargs
        input_guardrail: BaseGuardrail | None = None,
        output_guardrail: BaseGuardrail | None = None,
    ):
        self.logger = logger
        self.client = client
        self.memory = memory if memory is not None else ConversationMemory(logger=logger, system_prompt=system_prompt)
        self.tools = tools if tools is not None else ToolRegistry()
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.context_manager = context_manager
        self.max_context_tokens = max_context_tokens
        self.augmentation = augmentation
        # Merge plural and singular kwargs (plural takes precedence)
        self.input_guardrails = list(input_guardrails) if input_guardrails else []
        if input_guardrail is not None and input_guardrail not in self.input_guardrails:
            self.input_guardrails.insert(0, input_guardrail)
        self.output_guardrails = list(output_guardrails) if output_guardrails else []
        if output_guardrail is not None and output_guardrail not in self.output_guardrails:
            self.output_guardrails.insert(0, output_guardrail)
        self.rate_limiter = rate_limiter
        self.audit_trail = audit_trail
        self.approval_handler = approval_handler
        self.skills = skills
        self.hooks = hook_chain or HookChain()
        self.mode_manager = mode_manager
        self._last_user_message = ""
        self._skills_discovery_appended = False
        self._last_prompt_tokens: int = 0

    @property
    def input_guardrail(self) -> BaseGuardrail | None:
        """Backward-compatible accessor: returns first input guardrail or None."""
        return self.input_guardrails[0] if self.input_guardrails else None

    @input_guardrail.setter
    def input_guardrail(self, value: BaseGuardrail | None) -> None:
        self.input_guardrails = [value] if value is not None else []

    @property
    def output_guardrail(self) -> BaseGuardrail | None:
        """Backward-compatible accessor: returns first output guardrail or None."""
        return self.output_guardrails[0] if self.output_guardrails else None

    @output_guardrail.setter
    def output_guardrail(self, value: BaseGuardrail | None) -> None:
        self.output_guardrails = [value] if value is not None else []

    def _log(self, msg: str) -> None:
        if self.verbose:
            _log.debug(msg)

    async def _emit(self, event: HookEvent, **kwargs) -> HookContext:
        info = AgentInfo(
            model=self.client.model,
            iteration=kwargs.get("iteration", 0),
        )
        ctx = HookContext(event=event, agent=info, **kwargs)
        ctx = await self.hooks.emit(ctx)
        for msg in ctx.inject_messages:
            self.memory.add_context_message(msg, label="hook_inject")
        return ctx

    def _format_tool_calls_for_memory(self, tool_calls: list[ToolCall]) -> list[dict]:
        return [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in tool_calls
        ]

    async def _get_managed_messages(self) -> list[dict]:
        messages = self.memory.get_messages()

        if self.skills and not self._skills_discovery_appended:
            if self._last_user_message:
                discovery = self.skills.get_routed_discovery_prompt(self._last_user_message)
            else:
                discovery = self.skills.get_discovery_prompt()
            if discovery:
                for i, m in enumerate(messages):
                    if m.get("role") == "system":
                        messages[i] = {"role": "system", "content": m["content"] + discovery}
                        break
                else:
                    messages.insert(0, {"role": "system", "content": discovery})
                self._skills_discovery_appended = True

        if self.context_manager:
            pre = len(messages)
            messages = await self.context_manager.manage(messages, self.max_context_tokens)
            # Authoritative compaction signal: did manage() actually trim?
            # Stamped onto POST_COMPACT metadata so persistence hooks (e.g.
            # ReadBeforeWriteResetHook) only act on a real trim, not every iter.
            self._last_compacted = len(messages) < pre

        return messages

    async def _augment_memory(self, user_message: str) -> str:
        if not self.augmentation:
            return user_message
        return await self.augmentation.augment_for_memory(user_message)

    async def _augment_llm(self, messages: list[dict]) -> list[dict]:
        if not self.augmentation:
            return messages
        return await self.augmentation.augment_for_llm(messages)

    def _check_skill_activation(self, content: str) -> tuple[str, str] | None:
        import re as _re

        match = _re.search(r"\[ACTIVATE_SKILL:\s*([a-z0-9_-]+)\]", content)
        if match and self.skills and self.skills.get(match.group(1)):
            return (match.group(1), content.replace(match.group(0), "").strip())
        return None

    def _audit(self, event_type: str, **kwargs) -> None:
        if self.audit_trail:
            import time as _time

            self.audit_trail.record(AuditEntry(timestamp=_time.time(), event_type=event_type, **kwargs))

    @property
    def _pipeline(self) -> ToolExecutionPipeline:
        """Lazy-initialized tool execution pipeline."""
        if not hasattr(self, "_tool_pipeline"):
            self._tool_pipeline = ToolExecutionPipeline(
                tools=self.tools,
                memory=self.memory,
                rate_limiter=self.rate_limiter,
                approval_handler=self.approval_handler,
                hook_chain=self.hooks,
                logger=self.logger,
                verbose=self.verbose,
                audit_fn=self._audit,
                mode_manager=self.mode_manager,
            )
        return self._tool_pipeline

    async def _validate_input(self, text_part: str) -> None:
        """Run input guardrails and PRE_INPUT hook. Raises on block/abort."""
        for grd in self.input_guardrails:
            result = await grd.check(text_part)
            self._audit("input_check", details=f"guardrail={type(grd).__name__} passed={result.passed}")
            if not result.passed:
                self._log(f"Input blocked by {type(grd).__name__}: {result.reason}")
                raise AgentGuardrailError(result.reason, direction="input")

        ctx = await self._emit(HookEvent.PRE_INPUT, messages=self.memory.get_messages(), user_message=text_part)
        if ctx.abort:
            raise AgentAbortedError(ctx.inject_message or "Input rejected by hook")

    async def _process_output(self, output: str, response: object, iteration: int) -> str:
        """Run output guardrails, emit POST_OUTPUT, save to memory."""
        for grd in self.output_guardrails:
            out_result = await grd.check(output)
            self._audit("output_check", details=f"guardrail={type(grd).__name__} passed={out_result.passed}")
            if not out_result.passed:
                output = f"[GUARDRAIL WARNING ({type(grd).__name__}): {out_result.reason}]\n\n{output}"
                break

        await self._emit(HookEvent.POST_OUTPUT, iteration=iteration, llm_response=response)
        self.memory.add_assistant_message(output)
        return output

    def _activate_skill(self, content: str) -> tuple[str, str] | None:
        """Check for skill activation marker and activate if found."""
        activation = self._check_skill_activation(content)
        if not activation:
            return None
        skill_name, remaining = activation
        self.memory.add_assistant_message(content)
        if not self.skills.is_activated(skill_name):
            body = self.skills.activate(skill_name)
            if body:
                skill = self.skills.get(skill_name)
                self.memory.add_context_message(
                    f'<skill name="{skill_name}" dir="{skill.skill_dir}">\n{body}\n</skill>',
                    label=skill_name,
                )
                self._log(f"Skill activated: {skill_name}")
        return activation

    async def _prepare_iteration(self, iteration: int) -> list[dict]:
        """Run compaction, get managed messages, augment for LLM."""
        await self._emit(HookEvent.PRE_COMPACT, iteration=iteration)
        messages = await self._get_managed_messages()
        await self._emit(
            HookEvent.POST_COMPACT,
            iteration=iteration,
            messages=messages,
            metadata={"compacted": getattr(self, "_last_compacted", False)},
        )
        messages = await self._augment_llm(messages)
        return messages

    def _update_usage(self, response: object, total_usage: object | None) -> object:
        """Update token usage from response. Returns updated total_usage."""
        if not response.usage:
            return total_usage
        self._last_prompt_tokens = response.usage.prompt_tokens
        if self.context_manager:
            self.context_manager.last_actual_tokens = response.usage.prompt_tokens
        if total_usage is None:
            from koboi.types import TokenUsage

            total_usage = TokenUsage()
        total_usage.prompt_tokens += response.usage.prompt_tokens
        total_usage.completion_tokens += response.usage.completion_tokens
        return total_usage

    async def _prepare_run(self, user_message: str | list) -> tuple[str | list, list[dict] | None, float]:
        """Shared setup for run() and run_stream().

        Returns (user_message, tool_defs, start_time).
        """
        text_part = user_message if isinstance(user_message, str) else _extract_text(user_message)
        self._last_user_message = text_part
        await self._emit(HookEvent.SESSION_START)
        await self._validate_input(text_part)
        if isinstance(user_message, str):
            user_message = await self._augment_memory(user_message)
        self.memory.add_user_message(user_message)
        tool_defs = self.tools.get_definitions() or None
        return user_message, tool_defs, _time.monotonic()

    def _store_tool_response_in_memory(self, response: object) -> None:
        """Store assistant message with tool call details in conversation memory."""
        self.memory.add_assistant_message(
            response.content,
            self._format_tool_calls_for_memory(response.tool_calls),
        )

    async def run(self, user_message: str | list) -> RunResult:
        user_message, tool_defs, _start = await self._prepare_run(user_message)
        tool_calls_made: list[ToolCall] = []
        total_usage = None

        for i in range(self.max_iterations):
            messages = await self._prepare_iteration(i)
            tokens = estimate_tokens(messages)
            self._log(f"iteration {i + 1}: {len(messages)} messages, ~{tokens} tokens")

            await self._emit(HookEvent.PRE_LLM_CALL, iteration=i, messages=messages)
            response = await self.client.complete(messages=messages, tools=tool_defs)
            await self._emit(HookEvent.POST_LLM_CALL, iteration=i, llm_response=response)

            total_usage = self._update_usage(response, total_usage)

            if response.content and self.skills and self._activate_skill(response.content):
                continue

            if response.is_complete:
                output = await self._process_output(response.content, response, i)
                await self._emit(HookEvent.SESSION_END, iteration=i)
                return RunResult(
                    content=output,
                    iterations_used=i + 1,
                    tool_calls_made=tool_calls_made,
                    token_usage=total_usage,
                    success=True,
                    elapsed_seconds=_time.monotonic() - _start,
                    metadata={"model": self.client.model if hasattr(self.client, "model") else ""},
                )

            if response.tool_calls:
                self._log(f"LLM requested {len(response.tool_calls)} tool call(s)")
                self._store_tool_response_in_memory(response)
                for tc in response.tool_calls:
                    tool_calls_made.append(tc)
                    await self._pipeline.execute_tool_call(tc, iteration=i)

        await self._emit(HookEvent.SESSION_END, iteration=self.max_iterations)
        raise AgentMaxIterationsError(self.max_iterations)

    async def run_stream(self, user_message: str | list) -> AsyncGenerator:
        """Stream agent execution as a sequence of StreamEvents."""
        try:
            user_message, tool_defs, _start = await self._prepare_run(user_message)
        except (AgentGuardrailError, AgentAbortedError) as exc:
            yield ErrorEvent(error=exc)
            return

        _stream_tools_used: list[str] = []

        for i in range(self.max_iterations):
            messages = await self._prepare_iteration(i)
            tokens = estimate_tokens(messages)
            yield IterationEvent(iteration=i, messages_count=len(messages), tokens_estimated=tokens)

            await self._emit(HookEvent.PRE_LLM_CALL, iteration=i, messages=messages)

            final_response = None
            async for event in self.client.complete_stream(messages=messages, tools=tool_defs):
                if isinstance(event, TextDeltaEvent):
                    yield event
                elif isinstance(event, ToolCallEvent):
                    yield event
                elif isinstance(event, CompleteEvent):
                    final_response = event.response

            await self._emit(HookEvent.POST_LLM_CALL, iteration=i, llm_response=final_response)

            if final_response and final_response.usage:
                self._last_prompt_tokens = final_response.usage.prompt_tokens
                if self.context_manager:
                    self.context_manager.last_actual_tokens = final_response.usage.prompt_tokens

            if final_response is None:
                yield ErrorEvent(error=AgentMaxIterationsError(i + 1))
                return

            if final_response.content and self.skills and self._activate_skill(final_response.content):
                continue

            if final_response.is_complete:
                output = await self._process_output(final_response.content or "", final_response, i)
                await self._emit(HookEvent.SESSION_END, iteration=i)
                seen: set[str] = set()
                unique_tools = [t for t in _stream_tools_used if t not in seen and not seen.add(t)]  # type: ignore[func-returns-value]
                yield CompleteEvent(
                    response=final_response,
                    content=output,
                    elapsed_seconds=_time.monotonic() - _start,
                    iterations_used=i + 1,
                    tools_used=unique_tools,
                )
                return

            if final_response.tool_calls:
                self._store_tool_response_in_memory(final_response)
                for tc in final_response.tool_calls:
                    _stream_tools_used.append(tc.name)
                    yield ToolCallEvent(tool_name=tc.name, tool_call_id=tc.id, arguments=tc.arguments)

                    pipeline_result = await self._pipeline.execute_tool_call(tc, iteration=i)
                    yield ToolResultEvent(
                        tool_name=tc.name,
                        tool_call_id=tc.id,
                        result=pipeline_result.result,
                    )

        await self._emit(HookEvent.SESSION_END, iteration=self.max_iterations)
        yield ErrorEvent(error=AgentMaxIterationsError(self.max_iterations))

    async def chat(self, user_message: str | list) -> RunResult:
        return await self.run(user_message)

    def reset(self) -> None:
        self.memory.clear()
        if self.rate_limiter:
            self.rate_limiter.reset()
