"""koboi/self_consistency.py -- self-consistency aggregation for structured output (P4).

Self-consistency (Wang et al. 2022): sample N completions of the terminal answer,
pick the most common. For structured (JSON) output the exact-match majority on a
normalized form is well-defined and deterministic -- no judge, no embeddings.
Free-text aggregation is deferred (no in-codebase similarity primitive). Opt-in.
"""

from __future__ import annotations

import json

from koboi.types import AgentResponse, TokenUsage


def aggregate_structured(samples: list[AgentResponse]) -> tuple[AgentResponse, float]:
    """Pick the most common normalized-JSON response across N samples (P4).

    Returns ``(canonical, agreement)`` where ``agreement`` is the majority fraction
    (count_majority / N). Token usage is summed across all samples for cost
    accounting. If any sample is not valid JSON, falls back to ``samples[0]``.
    """
    if not samples:
        raise ValueError("aggregate_structured requires >=1 sample")
    if len(samples) == 1:
        return samples[0], 1.0

    norms: list[str] = []
    for s in samples:
        try:
            obj = json.loads(s.content) if s.content else None
        except Exception:
            return samples[0], 0.0  # a sample wasn't valid JSON -> no aggregation (0.0 = not aggregated)
        norms.append(json.dumps(obj, sort_keys=True) if obj is not None else (s.content or ""))

    counts: dict[str, int] = {}
    for n in norms:
        counts[n] = counts.get(n, 0) + 1
    winner = max(counts, key=lambda k: counts[k])
    agreement = counts[winner] / len(samples)

    canonical = next(s for s, n in zip(samples, norms, strict=False) if n == winner)
    total = _sum_usage(samples)
    if total is not None:
        canonical = AgentResponse(
            content=canonical.content,
            tool_calls=canonical.tool_calls,
            usage=total,
            model=canonical.model,
            base_url=canonical.base_url,
        )
    return canonical, agreement


def _sum_usage(samples: list[AgentResponse]) -> TokenUsage | None:
    if not any(getattr(s, "usage", None) for s in samples):
        return None
    total = TokenUsage()
    for s in samples:
        u = getattr(s, "usage", None)
        if u:
            total.prompt_tokens += u.prompt_tokens
            total.completion_tokens += u.completion_tokens
            total.reasoning_tokens += getattr(u, "reasoning_tokens", 0)
    return total
