"""koboi/guardrails/grounding.py -- runtime faithfulness guardrail (Wave 2 A3).

The "real answer-confidence": decomposes the agent's answer into atomic claims and
NLI-checks each against the retrieved context via a side-LLM. If coverage
(supported / total) < threshold, returns ``action="abstain"`` with a refusal -- the
loop (``loop.py`` A3.2) swaps the output for the refusal instead of returning a
confidently-retrieved-but-wrong answer.

This is the ONLY primitive that catches a confidently-retrieved-but-wrong answer in
the loop (empty retrieval is handled by the A2 abstention marker; low-relevance is
a future method-aware gate). It is a probabilistic LLM-judge catch, NOT a
deterministic guarantee -- fail-soft: any judge error passes-through (never breaks
the run, mirroring ``ProactiveMemory.extract_and_store``).

Opt-in via config (cost/latency: 1 + N side-LLM calls per terminal answer where N =
claim count)::

    guardrails:
      output:
        - name: grounding_check
          provider: openai
          model: gpt-4o-mini
          api_key: ${OPENAI_API_KEY:}
          base_url: ${OPENAI_BASE_URL:}
          threshold: 0.8

The eval-only ``RAGASScorer`` (``koboi/eval/scorers/ragas_scorer.py``) stays for
offline calibration; this guardrail ports the *technique* (claim-decomposition +
NLI) to a runtime path WITHOUT dragging ragas/langchain/datasets into the base
install.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from koboi.guardrails.base import BaseGuardrail
from koboi.types import GuardrailResult

if TYPE_CHECKING:
    from koboi.llm.base import LLMClient
    from koboi.logger import AgentLogger

_logger = logging.getLogger(__name__)

DEFAULT_REFUSAL = "I don't have enough grounded information to answer this confidently."

_DECOMPOSE_PROMPT = (
    "Break the following answer into atomic factual claims. Output ONLY a JSON "
    "array of short strings, no prose. Example: [\"claim one\", \"claim two\"].\n\n"
    "Answer:\n{answer}"
)
_NLI_PROMPT = (
    "You are a strict grounding checker. Given the retrieved context, decide if the "
    "claim is SUPPORTED (directly entailed by the context), CONTRADICTED (the context "
    "says the opposite), or UNSUPPORTED (not in the context). Reply with exactly one "
    "word: SUPPORTED, CONTRADICTED, or UNSUPPORTED.\n\n"
    "Context:\n{context}\n\nClaim:\n{claim}"
)


class GroundingGuardrail(BaseGuardrail):
    """Runtime faithfulness guardrail (A3). See module docstring."""

    def __init__(
        self,
        *,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: str = "",
        base_url: str = "",
        threshold: float = 0.8,
        refusal_text: str | None = None,
        timeout: float = 60.0,
        logger: AgentLogger | None = None,
        **kwargs: object,
    ) -> None:
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._threshold = float(threshold)
        self._refusal_text = refusal_text or DEFAULT_REFUSAL
        self._timeout = float(timeout)
        self._logger = logger
        self._client: LLMClient | None = None  # lazily built on first check
        # Observability: last computed coverage (None when cost-gated/skipped).
        self.last_coverage: float | None = None

    def _get_client(self) -> LLMClient | None:
        if self._client is None:
            try:
                from koboi.llm.factory import create_client

                self._client = create_client(
                    provider=self._provider,
                    model=self._model,
                    api_key=self._api_key,
                    base_url=self._base_url,
                    timeout=self._timeout,
                )
            except Exception as exc:  # nosec - fail-soft, never break the run
                _logger.warning("GroundingGuardrail judge client build failed: %s", exc)
                self._client = None
        return self._client

    async def check(self, content: str, context: list[str] | None = None) -> GuardrailResult:
        # Cost-gate: no retrieved context -> A2 already cued abstention; no point
        # judging an answer with nothing to ground against.
        if not content or not context:
            self.last_coverage = None
            return GuardrailResult(passed=True)
        client = self._get_client()
        if client is None:
            self.last_coverage = None
            return GuardrailResult(passed=True)  # fail-soft: no judge available
        try:
            claims = await self._decompose(client, content)
            if not claims:
                self.last_coverage = None
                return GuardrailResult(passed=True)
            ctx_text = "\n---\n".join(context)
            supported = 0
            for claim in claims:
                if await self._nli(client, ctx_text, claim) == "SUPPORTED":
                    supported += 1
            coverage = supported / len(claims)
            self.last_coverage = coverage
            if coverage >= self._threshold:
                return GuardrailResult(passed=True)
            return GuardrailResult(
                passed=False,
                reason=f"grounding coverage {coverage:.2f} < {self._threshold}",
                action="abstain",
                sanitized_content=self._refusal_text,
            )
        except Exception as exc:  # nosec - fail-soft, never break the run
            _logger.warning("GroundingGuardrail judge call failed: %s", exc)
            self.last_coverage = None
            return GuardrailResult(passed=True)

    async def _decompose(self, client: LLMClient, answer: str) -> list[str]:
        resp = await client.complete(
            messages=[{"role": "user", "content": _DECOMPOSE_PROMPT.format(answer=answer)}],
            tools=None,
        )
        text = (resp.content or "").strip()
        # Prefer a JSON array; tolerate fenced/prose-wrapped output.
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                claims = json.loads(text[start : end + 1])
                if isinstance(claims, list):
                    return [str(c).strip() for c in claims if str(c).strip()]
            except json.JSONDecodeError:
                pass
        # Fallback: non-empty lines as claims.
        return [ln.strip("- ").strip() for ln in text.splitlines() if ln.strip()]

    async def _nli(self, client: LLMClient, ctx_text: str, claim: str) -> str:
        resp = await client.complete(
            messages=[{"role": "user", "content": _NLI_PROMPT.format(context=ctx_text, claim=claim)}],
            tools=None,
        )
        verdict = (resp.content or "").strip().upper()
        # Order matters: "SUPPORTED" is a substring of "UNSUPPORTED", so check
        # UNSUPPORTED (and CONTRADICTED) first to avoid a false SUPPORTED match.
        for v in ("UNSUPPORTED", "CONTRADICTED", "SUPPORTED"):
            if v in verdict:
                return v
        return "UNSUPPORTED"
