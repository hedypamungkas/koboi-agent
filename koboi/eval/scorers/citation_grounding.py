"""koboi/eval/scorers/citation_grounding.py -- citation resolution scorer.

ALCE-style citation grounding for the numbered-citation format the RAG augmentation
emits (``[1] [Source: company_policy.md]\\n<chunk content>``). Verifies that every
citation marker in the answer resolves to a chunk that was actually retrieved -- a
mock-safe (no-LLM) proxy for citation correctness.

Two marker kinds are supported:
- positional ``[n]``  -> resolves iff ``1 <= n <= len(rag_results)``
- named ``[Source: x]`` -> resolves iff ``x`` is among the retrieved chunk sources

A live (LLM) leg that NLI-checks each cited span against its chunk is deferred to the
Tier-2 RAGAS suite; this scorer is the deterministic format-vs-correctness gate.
"""

from __future__ import annotations

import re

from koboi.types import EvalCase, EvalScore
from koboi.eval.scorers.base import BaseScorer

CITATION_NUM = re.compile(r"\[(\d+)\]")
SOURCE_NAMED = re.compile(r"\[Source:\s*([^\]]+)\]", re.IGNORECASE)


def _rag_sources(rag_results: list) -> set[str]:
    sources: set[str] = set()
    for c in rag_results or []:
        if isinstance(c, dict):
            s = c.get("source")
            if s:
                sources.add(str(s))
    return sources


def citation_precision(output: str, rag_results: list) -> tuple[float, int, int]:
    """Return ``(precision, resolved_count, total_cited)`` for an answer.

    ``precision`` is the fraction of citation markers that resolve to a retrieved
    chunk. ``total_cited == 0`` is a vacuous pass (precision 1.0, nothing to verify).
    """
    n_chunks = len(rag_results or [])
    sources = _rag_sources(rag_results)
    nums = [int(x) for x in CITATION_NUM.findall(output)]
    named = [m.strip() for m in SOURCE_NAMED.findall(output)]
    total = len(nums) + len(named)
    if total == 0:
        return 1.0, 0, 0
    resolved = sum(1 for n in nums if 1 <= n <= n_chunks)
    resolved += sum(1 for s in named if s in sources)
    return resolved / total, resolved, total


class CitationGroundingScorer(BaseScorer):
    """Citation precision scorer over ``context['rag_results']``."""

    def __init__(self, k: int | None = None):
        # ``k`` is accepted for API symmetry with RetrievalMetricScorer but unused:
        # citation resolution checks against the whole retrieved set, not a top-k.
        self.k = k

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        rag = context.get("rag_results") or []
        precision, resolved, total = citation_precision(output, rag)
        if total == 0:
            return EvalScore("citation_grounding", 1.0, "no citations to verify")
        reason = f"{resolved}/{total} citations resolve to {len(rag)} retrieved chunk(s)"
        return EvalScore("citation_grounding", round(precision, 3), reason)
