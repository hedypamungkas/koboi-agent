"""koboi/hooks/reflection_hook.py -- tool-grounded reflection loop (self-healing P1).

P1 is the wedge that closes the headline gap (no unified reflection loop in
``AgentCore``). It is STRICTLY verifier-grounded -- it never self-critiques
without an external signal (Huang et al., "LLMs Cannot Self-Correct Reasoning
Yet", ICLR 2024: intrinsic self-correction is empirically weak):

  - POST_TOOL_USE -- a tool failed *identically* N times in a row. A side-LLM
    critiques WHY and injects an actionable "try a different approach; do NOT
    repeat the same call" note. One-off errors get only P0-D's message -- the
    critique fires only on *repeated* failure, to avoid cost on transient errors.
  - POST_OUTPUT -- the GroundingGuardrail verifier reports low coverage
    (``last_coverage`` < threshold). A side-LLM names the ungrounded claims and
    the hook asks the loop to re-iterate with a "ground these from the provided
    context or hedge" critique. This is self-correction against the EXISTING
    retrieved context (no retrieval change -- query-reformulation/re-retrieval is
    a P1b/P3 enhancement).

Mechanism (mirrors HandoverDetectionHook): the hook sets a metadata flag and
does NOT raise (``HookChain.emit`` swallows hook exceptions into ``ctx.abort``
-> ``AgentAbortedError``, the wrong class). POST_OUTPUT sets
``ctx.metadata["reflection_retry"]``; ``loop._process_output`` stashes it and
``_run_loop`` honors it by ``continue``-ing (bounded by this hook's ``max_turns``
budget + the loop's ``max_iterations`` backstop). POST_TOOL_USE uses
``ctx.inject_messages`` -- the loop naturally continues after tool execution, so
that path needs NO loop seam.

Opt-in (``self_healing.enabled``); default off (zero behavior change). Fail-soft:
any critic error returns ``ctx`` unchanged (never breaks the run, like
GroundingGuardrail). Critique inputs are redacted via ``redact.py``. Priority 60
(post-business) so it runs after HandoverDetectionHook (50) -- on very-low
coverage handover wins; on recoverable-low coverage reflection retries.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import TYPE_CHECKING, Any

from koboi.hooks.chain import Hook, HookContext, HookEvent
from koboi.redact import redact_tool_arguments, redact_value

if TYPE_CHECKING:
    from koboi.guardrails.grounding import GroundingGuardrail
    from koboi.harness.recovery_budget import RecoveryBudget
    from koboi.llm.base import LLMClient
    from koboi.tools.registry import ToolRegistry

_logger = logging.getLogger(__name__)

_TOOL_ERROR_CRITIQUE_PROMPT = (
    "A tool in an AI agent failed repeatedly with the same arguments. In one short "
    "sentence, suggest a concrete DIFFERENT approach. Do NOT suggest repeating the "
    "same call.\n\n"
    "Tool: {tool}\nError detail (redacted): {detail}\n"
    "Arguments (redacted): {args}\n\nSuggested different approach:"
)

# Wave 2.4: when TypecheckHook attached structured diagnostics, critique against
# the FIRST failing file:line -- far more actionable than the generic prompt over
# a truncated result[:500] (a 50-error mypy dump is meaningless past line 1).
_TYPECHECK_CRITIQUE_PROMPT = (
    "An AI agent's type/lint checker reported errors. The FIRST error is at "
    "{file}:{line}: {message}. In 1-2 sentences, name the SINGLE most likely root "
    "cause and the specific edit to fix it. Do NOT suggest re-running the checker.\n\n"
    "Top {n} diagnostic(s) (redacted): {diags}\n\nSuggested fix:"
)


def _format_typecheck_diags(diags: list[dict]) -> str:
    """One-line summary of the top diagnostics for the critique prompt."""
    return "; ".join(f"{d.get('file')}:{d.get('line')}: {d.get('message')}" for d in diags)


_GROUNDING_CRITIQUE_PROMPT = (
    "An AI agent's answer was checked for grounding against retrieved context and "
    "found to have low coverage ({coverage:.2f}). In 1-2 sentences, name the likely "
    "ungrounded claim(s) and instruct the agent to either ground them strictly from "
    "the provided context or to hedge/refuse.\n\n"
    "Answer (redacted): {answer}\n\nCritique:"
)

# P4 CRITIC: tool-grounded claim verification. registry.execute bypasses the
# approval/policy pipeline (MCP-bridge precedent), so the hook hardcodes a SAFE
# allowlist; config may subset, never extend beyond these.
_SAFE_VERIFIER_TOOLS = {"calculate", "web_search"}

_TYPED_DECOMPOSE_PROMPT = (
    "Decompose the following AI agent answer into atomic claims. For each claim, set "
    "'kind' to 'math' (a numerical/arithmetic statement), 'fact' (a checkable factual "
    "assertion), or 'other' (opinion/recommendation). For math claims, set 'hint' to the "
    "expression to evaluate (e.g. '2+2'). Respond ONLY as a JSON array, at most {max_claims} "
    'items: [{{"claim": "...", "kind": "math|fact|other", "hint": "..."}}].\n\n'
    "Answer (redacted): {answer}\n\nJSON:"
)

_FACT_NLI_PROMPT = (
    "Does the following web search snippet SUPPORT, REFUTE, or remain NEUTRAL toward the "
    "claim? Respond with exactly one word: SUPPORT, REFUTE, or NEUTRAL.\n\n"
    "Claim: {claim}\nSnippet (redacted): {snippet}\n\nVerdict:"
)

# Note on fail-soft: when the critic has no client or errors, ``_ask`` returns
# None and BOTH paths SKIP the reflection (no retry) -- consistent with
# GroundingGuardrail's "on judge error -> pass" convention (a broken critic must
# not cause spurious retries). P0-D's actionable error message still guides the
# LLM on the tool path; the abstain refusal still stands on the grounding path.


class ReflectionHook(Hook):
    """Tool-grounded reflection loop (self-healing P1). See module docstring."""

    priority = 60

    def __init__(
        self,
        client: LLMClient | None,
        grounding: GroundingGuardrail | None = None,
        max_turns: int = 3,
        fail_soft: bool = True,
        tool_error_threshold: int = 2,
        grounding_threshold: float = 0.6,
        budget: RecoveryBudget | None = None,
        tools: ToolRegistry | None = None,
        verifier_tools: list[str] | None = None,
        max_claims: int = 5,
    ) -> None:
        self._client = client
        self._grounding = grounding
        self._max_turns = int(max_turns)
        self._fail_soft = bool(fail_soft)
        self._tool_error_threshold = int(tool_error_threshold)
        self._grounding_threshold = float(grounding_threshold)
        # Shared per-run budget (router-owned); consumed here only when the router chose
        # "reflect" and the critique actually fires. None in standalone mode (P1 path).
        self._budget = budget
        # P4 CRITIC: tool registry for tool-grounded claim verification. The verifier
        # only ever calls tools in the hardcoded _SAFE_VERIFIER_TOOLS allowlist.
        self._tools = tools
        self._verifier_tools = [t for t in (verifier_tools or ["calculate", "web_search"]) if t in _SAFE_VERIFIER_TOOLS]
        self._max_claims = int(max_claims)
        # Per-run state (reset on SESSION_START).
        self._turns_used = 0
        self._tool_error_counts: dict[str, int] = {}

    def handles(self) -> list[HookEvent]:
        return [HookEvent.SESSION_START, HookEvent.POST_TOOL_USE, HookEvent.POST_OUTPUT]

    async def execute(self, ctx: HookContext) -> HookContext:
        try:
            if ctx.event == HookEvent.SESSION_START:
                self._turns_used = 0
                self._tool_error_counts = {}
            elif ctx.event == HookEvent.POST_TOOL_USE:
                await self._on_post_tool_use(ctx)
            elif ctx.event == HookEvent.POST_OUTPUT:
                await self._on_post_output(ctx)
        except Exception as exc:  # fail-soft: never break the run
            if self._fail_soft:
                _logger.warning("ReflectionHook fail-soft (event=%s): %s", ctx.event, exc)
            else:
                raise
        return ctx

    # -- POST_TOOL_USE: critique repeated identical tool failures ---------------

    async def _on_post_tool_use(self, ctx: HookContext) -> None:
        result = ctx.tool_result or ""
        # P2a surfaces the structured error_kind onto ctx.metadata; prefer it over the
        # fragile "Error:" prefix string-match (keep the prefix as a fallback).
        is_error = ctx.metadata.get("tool_error_kind") is not None or result.startswith("Error:")
        if not is_error:
            # Success resets the consecutive-failure counter for this (tool, args).
            self._tool_error_counts.pop(self._tool_key(ctx), None)
            return
        if self._turns_used >= self._max_turns:
            return  # reflection budget exhausted -> let the loop/LLM handle it
        key = self._tool_key(ctx)
        count = self._tool_error_counts.get(key, 0) + 1
        self._tool_error_counts[key] = count
        if count < self._tool_error_threshold:
            return  # one-off / early failure -> P0-D's actionable message suffices
        critique = await self._critique_tool_error(ctx, result)
        if critique is None:
            return  # no client / critic error -> fail-soft skip (P0-D's message still guides)
        self._turns_used += 1
        ctx.inject_messages.append(
            f"[REFLECTION] Tool '{ctx.tool_name}' has failed {count} time(s) with the "
            f"same arguments. {critique} (Do not repeat the exact same call.)"
        )

    @staticmethod
    def _tool_key(ctx: HookContext) -> str:
        args = ctx.tool_arguments or ""
        digest = hashlib.sha1(args.encode("utf-8", "replace"), usedforsecurity=False).hexdigest()[:12]
        return f"{ctx.tool_name}:{digest}"

    async def _critique_tool_error(self, ctx: HookContext, result: str) -> str | None:
        if self._client is None:
            return None
        diags = ctx.metadata.get("typecheck_diagnostics") or []
        if diags:
            # Wave 2.4: structured typecheck diagnostics are present -- critique
            # against the FIRST failing file:line rather than a truncated result.
            first = diags[0]
            top = diags[:3]
            prompt = _TYPECHECK_CRITIQUE_PROMPT.format(
                file=first.get("file") or "?",
                line=first.get("line") or "?",
                message=redact_value(first.get("message") or ""),
                n=len(top),
                diags=redact_value(_format_typecheck_diags(top)),
            )
        else:
            prompt = _TOOL_ERROR_CRITIQUE_PROMPT.format(
                tool=ctx.tool_name or "?",
                detail=redact_value(result[:500]),
                args=redact_tool_arguments(ctx.tool_arguments or "{}"),
            )
        return await self._ask(prompt)

    # -- POST_OUTPUT: low-grounding reground-and-retry --------------------------

    async def _on_post_output(self, ctx: HookContext) -> None:
        # P2a: when the ladder router is active it stamps recovery_plan and owns the
        # shared RecoveryBudget; only fire when chosen ("reflect"). When no router is
        # active (standalone), use the hook's own _turns_used budget (P1 behavior).
        plan = ctx.metadata.get("recovery_plan")
        if plan is not None and plan.get("rung") != "reflect":
            return  # ladder routed to a different rung this turn (e.g. handover)
        if self._grounding is None:
            return
        if plan is None and self._turns_used >= self._max_turns:
            return  # standalone budget exhausted -> fall through to abstain/handover
        coverage: Any = getattr(self._grounding, "last_coverage", None)
        if coverage is None or coverage >= self._grounding_threshold:
            return  # grounded enough, or no signal -> nothing to do
        # TODO(P2): if ctx.iteration == max_iterations-1 the loop's (i+1) < max guard
        # will skip the retry, so this critic call + budget are wasted. Pass
        # max_iterations so the hook can decline to critique with no iteration left.
        # P4 CRITIC: prefer tool-grounded verification (external signal -- what the
        # literature says actually enables self-correction) over intrinsic self-critique;
        # fall back to the intrinsic critique when no tools / no tool evidence.
        critique = await self._tool_verify_claims(self._answer_text(ctx))
        if critique is None:
            critique = await self._critique_grounding(coverage, self._answer_text(ctx))
        if critique is None:
            return  # no client / no tool evidence / critic error -> fail-soft skip
        if plan is None:
            self._turns_used += 1  # standalone: count against own budget
        elif self._budget is not None:
            self._budget.consume()  # router-mode: spend the shared budget on an actual reflect
        # Set the retry flag (loop._process_output stashes it; _run_loop honors it by
        # re-iterating and adds the critique to memory AFTER the answer). Do NOT use
        # inject_messages here -- _emit runs before add_assistant_message, so it would
        # land the critique BEFORE the answer.
        ctx.metadata["reflection_retry"] = {
            "reason": f"low grounding coverage ({coverage:.2f} < {self._grounding_threshold})",
            "coverage": float(coverage),
            "critique": critique,
        }

    @staticmethod
    def _answer_text(ctx: HookContext) -> str:
        resp = getattr(ctx, "llm_response", None)
        content = getattr(resp, "content", None)
        return content if isinstance(content, str) and content else ""

    async def _critique_grounding(self, coverage: float, answer: str) -> str | None:
        if self._client is None:
            return None
        prompt = _GROUNDING_CRITIQUE_PROMPT.format(coverage=float(coverage), answer=redact_value(answer[:2000]))
        return await self._ask(prompt)

    # -- P4 CRITIC: tool-grounded claim verification ----------------------------

    async def _tool_verify_claims(self, answer: str) -> str | None:
        """Verify the answer's claims via tools (P4 CRITIC). Return a tool-grounded
        critique, or None if no tools / no verifiable claims / all verified OK."""
        if self._tools is None or self._client is None or not self._verifier_tools:
            return None
        claims = await self._typed_decompose(answer)
        if not claims:
            return None
        findings: list[str] = []
        for c in claims[: self._max_claims]:
            kind = (c.get("kind") or "").strip()
            claim = (c.get("claim") or "").strip()
            hint = (c.get("hint") or "").strip()
            if kind == "math" and hint and "calculate" in self._verifier_tools:
                ev = await self._verify_math(claim, hint)
                if ev:
                    findings.append(ev)
            elif kind == "fact" and claim and "web_search" in self._verifier_tools:
                ev = await self._verify_fact(claim)
                if ev:
                    findings.append(ev)
        if not findings:
            return None
        return "[TOOL-VERIFIED] " + " ".join(findings)

    async def _typed_decompose(self, answer: str) -> list[dict]:
        prompt = _TYPED_DECOMPOSE_PROMPT.format(max_claims=self._max_claims, answer=redact_value(answer[:2000]))
        text = await self._ask(prompt)
        if not text:
            return []
        start = text.find("[")
        if start < 0:
            return []
        try:
            # raw_decode stops at the first JSON value, ignoring trailing prose.
            data, _ = json.JSONDecoder().raw_decode(text[start:])
        except Exception:
            return []
        return [c for c in data if isinstance(c, dict)]

    async def _verify_math(self, claim: str, hint: str) -> str | None:
        if self._tools is None:
            return None
        try:
            outcome = await self._tools.execute_outcome("calculate", json.dumps({"expression": hint}))
        except Exception as exc:
            _logger.warning("CRITIC verify_math failed: %s", exc)
            return None
        content = outcome.content or ""
        # calculate RETURNS (not raises) "Error calculating '...': ..." on a bad expression,
        # wrapped as errored=False -- guard explicitly so we don't parse digits from the error.
        if outcome.errored or "Error calculating" in content:
            return None
        computed = self._parse_number(content.split("=", 1)[-1])
        claim_nums = re.findall(r"-?\d+(?:\.\d+)?", claim)
        if computed is not None and claim_nums:
            claimed = float(claim_nums[-1])  # last number in the claim = asserted result
            if abs(computed - claimed) > 1e-9:
                return f"math claim '{claim}' is incorrect: '{hint}' = {computed:g} (not {claimed:g})."
        return None

    async def _verify_fact(self, claim: str) -> str | None:
        if self._tools is None:
            return None
        try:
            outcome = await self._tools.execute_outcome("web_search", json.dumps({"query": claim}))
        except Exception as exc:
            _logger.warning("CRITIC verify_fact failed: %s", exc)
            return None
        if outcome.errored:
            return None
        snippet = (outcome.content or "").strip()
        if not snippet:
            return None
        # NLI: only flag the claim if the snippet REFUTES it. A supporting/neutral snippet
        # must NOT trigger a spurious reflection retry on a correct answer (review C1).
        if await self._fact_nli(claim, snippet) != "refuted":
            return None
        return f"fact claim '{claim[:160]}' is contradicted by web_search: {snippet[:300]}."

    async def _fact_nli(self, claim: str, snippet: str) -> str:
        """Side-LLM NLI: does the snippet support/refute/neutral the claim?"""
        text = await self._ask(_FACT_NLI_PROMPT.format(claim=claim[:300], snippet=redact_value(snippet[:1000])))
        t = (text or "").lower()
        if "refut" in t:
            return "refuted"
        if "support" in t:
            return "supported"
        return "neutral"

    @staticmethod
    def _parse_number(text: str) -> float | None:
        m = re.search(r"-?\d+(?:\.\d+)?", text)
        try:
            return float(m.group()) if m else None
        except (ValueError, AttributeError):
            return None

    # -- side-LLM helper --------------------------------------------------------

    async def _ask(self, prompt: str) -> str | None:
        if self._client is None:
            return None
        try:
            resp = await self._client.complete(messages=[{"role": "user", "content": prompt}], tools=None)
        except Exception as exc:  # fail-soft: a judge hiccup must not break the run
            _logger.warning("ReflectionHook critic call failed: %s", exc)
            return None
        text = getattr(resp, "content", None)
        if isinstance(text, str):
            text = text.strip()
        return text or None
