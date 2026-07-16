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
    AgentHandoverError,
    AgentMaxIterationsError,
)
from koboi.guardrails.base import BaseGuardrail
from koboi.loop_pipeline import ToolExecutionPipeline
from koboi.memory import ConversationMemory
from koboi.tools.registry import ToolRegistry
from koboi.client import Client
from koboi.tokens import estimate_tokens
from koboi.hooks.chain import HookEvent, HookChain, HookContext, AgentInfo
from koboi.types import AgentResponse, AuditEntry, RunResult, TokenUsage, ToolCall

if TYPE_CHECKING:
    from koboi.logger import AgentLogger
    from koboi.context.manager import ContextManager
    from koboi.rag.augmentation import AugmentationStrategy
    from koboi.guardrails.rate_limiter import RateLimiter
    from koboi.guardrails.audit import AuditTrail
    from koboi.guardrails.approval import ApprovalHandler
    from koboi.skills.registry import SkillRegistry
    from koboi.modes import ModeManager
    from koboi.journal import StepJournal
    from koboi.trust import TrustStore
    from koboi.proactive_memory import ProactiveMemory

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
        client: Client,
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
        journal: StepJournal | None = None,
        trust_db: TrustStore | None = None,
        output_schema: dict | None = None,
        force_response_format_with_tools: bool = False,
        proactive_memory: ProactiveMemory | None = None,
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
        self._last_output_guardrail: dict | None = None  # R2: warn outcome -> RunResult.metadata
        self.rate_limiter = rate_limiter
        self.audit_trail = audit_trail
        self.approval_handler = approval_handler
        self.skills = skills
        self.hooks = hook_chain or HookChain()
        self.mode_manager = mode_manager
        self.journal = journal
        self.trust_db = trust_db
        # JSON Schema dict (provider-agnostic) for structured final output, or None.
        self.response_schema = output_schema
        # Gap B: opt in to response_format even on tool-carrying iterations on
        # providers that support it (OpenAI/Cloudflare); Anthropic keeps the
        # suppression (RF is emulated via a forced tool_use). See _resolve_response_format.
        self.force_response_format_with_tools = force_response_format_with_tools
        # Proactive long-term memory (opt-in; None unless memory.proactive.enabled).
        self.proactive_memory = proactive_memory
        # P2-A: turn counter. On a fresh agent this is 0; on resume it inherits
        # the journal's highest recorded turn so numbering stays continuous.
        self._turn_index: int = journal.turn_index if journal else 0
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

    def _resolve_response_format(self, tool_defs) -> dict | None:
        """Decide the response_format to pass to the LLM this iteration.

        Structured output shapes the FINAL answer; it is normally suppressed on
        tool-carrying iterations (undefined mid-tool-chain; on Anthropic it is
        emulated via a forced tool_use that is incompatible with real tool calls).
        ``force_response_format_with_tools`` opts back in on providers that
        support RF alongside tools (OpenAI/Cloudflare); Anthropic keeps the
        suppression regardless of the flag.
        """
        if self.response_schema is None:
            return None
        if not tool_defs:
            return self.response_schema
        if self.force_response_format_with_tools and getattr(self.client, "provider", "openai") != "anthropic":
            return self.response_schema
        return None

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

        # C/B: proactive long-term memory — ephemerally append recalled facts
        # (and the core block) to the system message AFTER compaction so they
        # reach the LLM this turn without persisting as conversation rows. Skipped
        # unless memory.proactive is enabled.
        if self.proactive_memory is not None and self._last_user_message:
            block = await self._proactive_block(self._last_user_message)
            if block:
                for i, m in enumerate(messages):
                    if m.get("role") == "system":
                        messages[i] = {"role": "system", "content": m["content"] + "\n\n" + block}
                        break
                else:
                    messages.insert(0, {"role": "system", "content": block})

        return messages

    async def _proactive_block(self, query: str) -> str:
        """Build the ephemeral proactive-memory injection (core block + recalled facts)."""
        pm = self.proactive_memory
        if pm is None:
            return ""
        parts: list[str] = []
        if pm.core_block_enabled:
            cb = pm.get_core_block()
            if cb:
                parts.append(cb)
        if pm.recall_enabled:
            try:
                recalled = await pm.recall(query)
            except Exception as exc:  # nosec - best-effort; never break the turn
                self._log(f"Proactive recall failed: {exc}")
                recalled = None
            if recalled:
                parts.append(recalled)
        return "\n\n".join(parts)

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
                trust_db=self.trust_db,
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
        _hr = ctx.metadata.get("handover_requested")  # B1.5: structural handover detection
        if _hr:
            raise AgentHandoverError(_hr.get("reason", "handover requested"), _hr.get("summary", ""))

    async def _process_output(self, output: str, response: object, iteration: int) -> str:
        """Run output guardrails, emit POST_OUTPUT, save to memory.

        ``block``/``deny``/``abort`` raises (denies the output); ``abstain``
        swaps the output for a refusal (A3 grounding guardrail); any other action
        (incl. ``warn`` and non-string/absent) prepends a warning and continues.
        The detailed reason is logged server-side only; the raised message carries
        just the guardrail name so a leaky ``reason`` can't re-leak via the error
        frame / durable job error.
        """
        # A3: thread retrieved context to output guardrails so a grounding
        # guardrail can judge faithfulness against the retrieved evidence.
        retrieved_context: list[str] = []
        if self.augmentation is not None:
            _results = getattr(self.augmentation, "last_results", None) or []
            retrieved_context = [r.chunk.content for r in _results]
        for grd in self.output_guardrails:
            out_result = await grd.check(output, context=retrieved_context)
            self._audit("output_check", details=f"guardrail={type(grd).__name__} passed={out_result.passed}")
            if not out_result.passed:
                action = out_result.action if isinstance(out_result.action, str) else ""
                if action.lower() in {"block", "deny", "abort"}:
                    self._log(f"Output blocked by {type(grd).__name__}: {out_result.reason}")
                    raise AgentGuardrailError(f"output blocked by {type(grd).__name__}", direction="output")
                if action.lower() == "abstain":
                    # A3.2: swap the output for a refusal. ``block`` is too harsh
                    # (it denies the whole turn); ``warn`` is too weak (it
                    # prepends). The guardrail supplies the refusal via
                    # ``sanitized_content`` (or a default).
                    self._last_output_guardrail = {
                        "guardrail": type(grd).__name__,
                        "reason": out_result.reason,
                        "action": "abstain",
                    }
                    output = out_result.sanitized_content or (
                        "I don't have enough grounded information to answer this confidently."
                    )
                    break
                self._last_output_guardrail = {
                    "guardrail": type(grd).__name__,
                    "reason": out_result.reason,
                    "action": "warn",
                }
                output = f"[GUARDRAIL WARNING ({type(grd).__name__}): {out_result.reason}]\n\n{output}"
                break

        ctx = await self._emit(HookEvent.POST_OUTPUT, iteration=iteration, llm_response=response)
        if ctx.abort:
            raise AgentAbortedError(ctx.inject_message or "Output rejected by hook")
        _hr = ctx.metadata.get("handover_requested")  # B1.5: structural handover detection
        if _hr:
            raise AgentHandoverError(_hr.get("reason", "handover requested"), _hr.get("summary", ""))
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
            # H3 / issue #46: ``!`cmd` `` preprocessing is fail-closed -- a skill
            # must declare ``allow-shell: true`` frontmatter AND the caller must
            # pass run_shell=True for blocks to execute. Model-activated skills
            # pass run_shell=False, so an untrusted SKILL.md can never run shell
            # on the activation path (supply-chain RCE guard).
            body = self.skills.activate(skill_name, run_shell=False)
            # R3: record activation to telemetry so evals can assert (t.activatedSkill
            # + skill_trigger_accuracy scorer). No-op when no TelemetryHook is wired.
            tel_hook = self.hooks.find_hook(lambda h: hasattr(h, "telemetry"))
            if tel_hook is not None:
                tel_hook.telemetry.record_skill_activation(skill_name)  # type: ignore[attr-defined]  # TelemetryHook found via hasattr lambda; Hook base has no telemetry attr
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

    def _update_usage(self, response: AgentResponse, total_usage: TokenUsage | None) -> TokenUsage | None:
        """Update token usage from response. Returns updated total_usage."""
        if not response.usage:
            return total_usage
        self._last_prompt_tokens = response.usage.prompt_tokens
        if self.context_manager:
            self.context_manager.last_actual_tokens = response.usage.prompt_tokens
        if total_usage is None:
            total_usage = TokenUsage()
        total_usage.prompt_tokens += response.usage.prompt_tokens
        total_usage.completion_tokens += response.usage.completion_tokens
        total_usage.reasoning_tokens += getattr(response.usage, "reasoning_tokens", 0)
        return total_usage

    async def _prepare_run(self, user_message: str | list) -> tuple[str | list, list[dict] | None, float]:
        """Shared setup for run() and run_stream().

        Returns (user_message, tool_defs, start_time).
        """
        text_part = user_message if isinstance(user_message, str) else _extract_text(user_message)
        self._last_user_message = text_part
        await self._emit(HookEvent.SESSION_START)
        # P2-A: a new run() call is a new user turn. Advances the journal's turn
        # counter so step rows for this invocation are numbered under a fresh
        # turn. (resume() does NOT advance -- it inherits the interrupted turn.)
        if self.journal:
            self.journal.advance_turn()
            self._turn_index = self.journal.turn_index
        await self._validate_input(text_part)
        if isinstance(user_message, str):
            user_message = await self._augment_memory(user_message)
        self.memory.add_user_message(user_message)
        tool_defs = self.tools.get_definitions() or None
        return user_message, tool_defs, _time.monotonic()

    def _store_tool_response_in_memory(self, response: AgentResponse) -> None:
        """Store assistant message with tool call details in conversation memory."""
        self.memory.add_assistant_message(
            response.content,
            self._format_tool_calls_for_memory(response.tool_calls),
        )

    # -- P2-A: step journal ------------------------------------------------

    def _journal_step(
        self,
        step_index: int,
        status: str,
        response: object | None = None,
        tool_calls: list[ToolCall] | None = None,
        is_terminal: bool = False,
        error: str | None = None,
    ) -> None:
        """Record one step in the journal (no-op when no journal is attached)."""
        if not self.journal:
            return
        usage = getattr(response, "usage", None) if response else None
        self.journal.record_step(
            turn_index=self._turn_index,
            step_index=step_index,
            status=status,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            tool_calls=tool_calls,
            is_terminal=is_terminal,
            error=error,
        )

    def _journal_max_iter(self) -> None:
        """Record the terminal 'max_iter' step (shared by run() and run_stream())."""
        self._journal_step(
            self.max_iterations - 1,
            status="max_iter",
            is_terminal=True,
            error=f"max_iterations={self.max_iterations}",
        )

    def _run_metadata(self, *, resumed: bool, last_step: int) -> dict:
        meta = {
            "model": self.client.model if hasattr(self.client, "model") else "",
            "session_id": getattr(self.memory, "session_id", None),
            "resumed": resumed,
            "turn_index": self._turn_index,
            "last_step": last_step,
        }
        # R4: stamp retrieved chunks so evals can assert on retrieval (t.retrievedChunk)
        # without a live LLM. last_results is overwritten each retrieval (not accumulated).
        if self.augmentation is not None:
            results = getattr(self.augmentation, "last_results", None) or []
            if results:
                meta["rag_results"] = [
                    {
                        "content": r.chunk.content,
                        "score": r.score,
                        "source": r.chunk.metadata.get("source", r.chunk.doc_id),
                        # Additive (RAG eval Tier 2/3): expose the retrieval method so
                        # semantic/hybrid evals can detect a silent degrade-to-keyword,
                        # and a stable doc_id for golden-qrels matching (vs fragile
                        # content-needle). No behavior change; existing readers ignore
                        # unknown keys.
                        "retrieval_method": r.retrieval_method,
                        "doc_id": r.chunk.doc_id,
                    }
                    for r in results
                ]
            # A1: retrieval confidence observability. Always stamped when RAG is on
            # (empty turns too) so consumers can see "retrieval ran, found nothing".
            # Scores are NOT comparable across methods (keyword=[0,1), bm25=unbounded,
            # semantic=[-1,1], hybrid=RRF~0.016, rerank:{p}=clamped, rerank:failed=base
            # unclamped), so `method` MUST travel with max_score. Empty -> sentinel.
            if results:
                top = max(results, key=lambda r: r.score)
                meta["retrieval_confidence"] = {
                    "max_score": top.score,
                    "method": top.retrieval_method,
                    "count": len(results),
                }
            else:
                meta["retrieval_confidence"] = {"max_score": None, "method": "none", "count": 0}
        # #9: stamp the query-rewrite outcome so evals/observability can inspect it.
        rw = getattr(self.augmentation, "last_rewrite", None)
        if rw:
            meta["rag_rewrite"] = rw
        # R2: stamp output-guardrail warn outcome so evals can assert (t.warned).
        if self._last_output_guardrail is not None:
            meta["guardrail_outcomes"] = [{"direction": "output", **self._last_output_guardrail}]
        return meta

    async def _repair_interrupted_turn(self) -> None:
        """Re-execute tool calls from the last assistant message whose
        results never landed in memory (the crash window before a tool result
        was persisted). Idempotency caveat: non-idempotent tools may re-run;
        DESTRUCTIVE tools will re-prompt for approval on the way through the
        pipeline, which is the safe default.

        16.7 fix: scan backwards for the last assistant message with
        tool_calls — not just ``msgs[-1]`` — so partial turns (some tools
        executed, some not) are correctly repaired.
        """
        msgs = self.memory.get_messages()
        if not msgs:
            return
        # Find the last assistant message that has tool_calls (scan backwards
        # so partial turns — where tool results follow the assistant message —
        # are also repaired).
        last_assistant = None
        for m in reversed(msgs):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                last_assistant = m
                break
        if last_assistant is None:
            return
        requested = last_assistant.get("tool_calls") or []
        answered = {m.get("tool_call_id") for m in msgs if m.get("role") == "tool"}
        missing = [tc for tc in requested if tc.get("id") not in answered]
        if not missing:
            return
        self._log(f"Resume: re-executing {len(missing)} missing tool call(s)")
        for tc_dict in missing:
            fn = tc_dict.get("function", {}) if isinstance(tc_dict, dict) else {}
            name = fn.get("name", "") if isinstance(fn, dict) else ""
            tc = ToolCall(
                id=tc_dict.get("id", "") if isinstance(tc_dict, dict) else "",
                name=name,
                arguments=fn.get("arguments", "{}") if isinstance(fn, dict) else "{}",
            )
            # Issue #8b: skip non-idempotent tools on resume so side-effecting
            # tools (charge_card, send_email, ...) cannot silently double-fire.
            # Record a synthetic result so the turn still has a tool message and
            # the loop can continue. Default ToolDefinition.idempotent=True means
            # existing tools keep re-running as before.
            tool_def = (
                self._pipeline.tools.get_definition(name)
                if getattr(self._pipeline, "tools", None) is not None
                else None
            )
            if tool_def is not None and not tool_def.idempotent:
                self._log(
                    f"Resume: skipping non-idempotent tool '{name}' re-execution "
                    f"(marked idempotent=False); recording synthetic result"
                )
                self.memory.add_tool_result(
                    tc.id,
                    f"[skipped on resume: non-idempotent tool '{name}'; re-invoke explicitly if needed]",
                )
                continue
            await self._pipeline.execute_tool_call(tc, iteration=0)

    async def _run_loop(self, tool_defs: list[dict] | None, _start: float, *, resumed: bool) -> RunResult:
        """Shared iteration loop for run() and resume().

        Journal writes are native (not hooks) so durability can't be bypassed.
        Each iteration: a 'running' marker at start, then 'skill'/'complete'/
        'tool_calls' as the outcome, and a terminal 'max_iter' row if the loop
        is exhausted.
        """
        tool_calls_made: list[ToolCall] = []
        pipeline_outcomes: list[dict] = []
        total_usage: TokenUsage | None = None
        self._last_output_guardrail = None  # R2: reset per run

        for i in range(self.max_iterations):
            messages = await self._prepare_iteration(i)
            self._journal_step(i, status="running")
            tokens = estimate_tokens(messages)
            self._log(f"iteration {i + 1}: {len(messages)} messages, ~{tokens} tokens")

            await self._emit(HookEvent.PRE_LLM_CALL, iteration=i, messages=messages)
            # Structured output (response_format): normally applied only on tool-less
            # iterations (it shapes the FINAL answer and is undefined mid-tool-chain;
            # on Anthropic it is emulated via a forced tool_use incompatible with real
            # tool calls). force_response_format_with_tools opts back in on providers
            # that support RF alongside tools (OpenAI/Cloudflare). See _resolve_response_format.
            _rf = self._resolve_response_format(tool_defs)
            response = await self.client.complete(messages=messages, tools=tool_defs, response_format=_rf)
            await self._emit(HookEvent.POST_LLM_CALL, iteration=i, llm_response=response)

            total_usage = self._update_usage(response, total_usage)

            if response.content and self.skills and self._activate_skill(response.content):
                self._journal_step(i, status="skill", response=response)
                continue

            if response.is_complete:
                output = await self._process_output(response.content, response, i)
                self._journal_step(i, status="complete", response=response, is_terminal=True)
                await self._emit(HookEvent.SESSION_END, iteration=i)
                return RunResult(
                    content=output,
                    iterations_used=i + 1,
                    tool_calls_made=tool_calls_made,
                    pipeline_outcomes=pipeline_outcomes,
                    token_usage=total_usage,
                    success=True,
                    elapsed_seconds=_time.monotonic() - _start,
                    metadata=self._run_metadata(resumed=resumed, last_step=i),
                )

            if response.tool_calls:
                self._log(f"LLM requested {len(response.tool_calls)} tool call(s)")
                self._store_tool_response_in_memory(response)
                for tc in response.tool_calls:
                    pr = await self._pipeline.execute_tool_call(tc, iteration=i)
                    # Only count tools that actually executed -- skipped/denied/blocked
                    # tools (rate-limit, approval, policy, mode) must NOT pollute
                    # tool_calls_made (and the derived tools_used) -- eval false positive.
                    if not pr.skipped:
                        tool_calls_made.append(tc)
                    pipeline_outcomes.append(
                        {
                            "tool_call_id": tc.id,
                            "tool_name": tc.name,
                            "skipped": pr.skipped,
                            "skip_reason": pr.skip_reason,
                        }
                    )
                self._journal_step(i, status="tool_calls", response=response, tool_calls=response.tool_calls)

        self._journal_max_iter()
        await self._emit(HookEvent.SESSION_END, iteration=self.max_iterations)
        raise AgentMaxIterationsError(self.max_iterations)

    async def run(self, user_message: str | list) -> RunResult:
        user_message, tool_defs, _start = await self._prepare_run(user_message)
        return await self._run_loop(tool_defs, _start, resumed=False)

    async def resume(self) -> RunResult:
        """Rehydrate-and-continue an interrupted session.

        Assumes memory was already rehydrated with the target session_id (the
        SQLiteMemory constructor reloads messages). Does NOT add a user message:
        it goes straight into the iteration loop on the existing messages.
        Prior 'running' step markers are marked 'interrupted', and any tool
        calls from the trailing assistant message that lack persisted results
        are re-executed before the loop resumes. The interrupted turn is
        inherited (no advance).
        """
        await self._emit(HookEvent.SESSION_START)
        if self.journal:
            open_rows = self.journal.list_open_running()
            if open_rows:
                self.journal.mark_interrupted(open_rows)
            self._turn_index = self.journal.turn_index
        await self._repair_interrupted_turn()
        tool_defs = self.tools.get_definitions() or None
        return await self._run_loop(tool_defs, _time.monotonic(), resumed=True)

    async def run_stream(self, user_message: str | list) -> AsyncGenerator:
        """Stream agent execution as a sequence of StreamEvents."""
        try:
            user_message, tool_defs, _start = await self._prepare_run(user_message)
        except (AgentGuardrailError, AgentAbortedError) as exc:
            yield ErrorEvent(error=exc)
            return

        self._last_output_guardrail = None  # R2: reset per run (parity with _run_loop)
        _stream_tools_used: list[str] = []
        # G8b: when output guardrails are configured, buffer TextDeltas and flush
        # them only after _process_output passes -- otherwise the tokens stream
        # (interactive SSE) / are appended (job replay buffer) BEFORE the guardrail
        # runs on the complete response, so a blocked output leaks. With no output
        # guardrail, stream live (current behavior; no latency cost).
        should_buffer = bool(self.output_guardrails)

        for i in range(self.max_iterations):
            messages = await self._prepare_iteration(i)
            self._journal_step(i, status="running")
            tokens = estimate_tokens(messages)
            yield IterationEvent(iteration=i, messages_count=len(messages), tokens_estimated=tokens)

            await self._emit(HookEvent.PRE_LLM_CALL, iteration=i, messages=messages)

            delta_buffer: list[TextDeltaEvent] = []
            final_response = None
            _rf = self._resolve_response_format(tool_defs)
            async for event in self.client.complete_stream(messages=messages, tools=tool_defs, response_format=_rf):
                if isinstance(event, TextDeltaEvent):
                    if should_buffer:
                        delta_buffer.append(event)
                    else:
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
                for d in delta_buffer:
                    yield d
                yield ErrorEvent(error=AgentMaxIterationsError(i + 1))
                return

            if final_response.content and self.skills and self._activate_skill(final_response.content):
                for d in delta_buffer:
                    yield d
                self._journal_step(i, status="skill", response=final_response)
                continue

            if final_response.is_complete:
                # May raise AgentGuardrailError (block) -- the buffer is then
                # discarded, so the blocked tokens never reach the stream.
                output = await self._process_output(final_response.content or "", final_response, i)
                for d in delta_buffer:
                    yield d
                self._journal_step(i, status="complete", response=final_response, is_terminal=True)
                await self._emit(HookEvent.SESSION_END, iteration=i)
                seen: set[str] = set()
                unique_tools = [t for t in _stream_tools_used if t not in seen and not seen.add(t)]  # type: ignore[func-returns-value]
                # M5: enrich CompleteEvent with Langfuse trace_id if available.
                trace_id = ""
                if self.hooks:
                    lf_hook = self.hooks.find_hook(lambda h: type(h).__name__ == "LangfuseTracingHook")
                    if lf_hook:
                        trace_id = getattr(lf_hook, "_trace_id", "") or ""
                yield CompleteEvent(
                    response=final_response,
                    content=output,
                    elapsed_seconds=_time.monotonic() - _start,
                    iterations_used=i + 1,
                    tools_used=unique_tools,
                    trace_id=trace_id,
                    # Parity with run(): stamp rag_results + guardrail_outcomes so the
                    # streaming path is eval/observable for retrieval (t.retrievedChunk).
                    metadata=self._run_metadata(resumed=False, last_step=i),
                )
                return

            if final_response.tool_calls:
                for d in delta_buffer:
                    yield d
                self._store_tool_response_in_memory(final_response)
                for tc in final_response.tool_calls:
                    yield ToolCallEvent(tool_name=tc.name, tool_call_id=tc.id, arguments=tc.arguments)

                    pipeline_result = await self._pipeline.execute_tool_call(tc, iteration=i)
                    # Mirror the non-streaming path: a skipped tool is not "used".
                    if not pipeline_result.skipped:
                        _stream_tools_used.append(tc.name)
                    yield ToolResultEvent(
                        tool_name=tc.name,
                        tool_call_id=tc.id,
                        result=pipeline_result.result,
                    )
                self._journal_step(
                    i, status="tool_calls", response=final_response, tool_calls=final_response.tool_calls
                )

        self._journal_max_iter()
        await self._emit(HookEvent.SESSION_END, iteration=self.max_iterations)
        yield ErrorEvent(error=AgentMaxIterationsError(self.max_iterations))

    async def chat(self, user_message: str | list) -> RunResult:
        return await self.run(user_message)

    def reset(self) -> None:
        self.memory.clear()
        if self.rate_limiter:
            self.rate_limiter.reset()
