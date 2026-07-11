"""Live statistical-confidence gate (Tier 3; CRITICAL dimension, weight 0.08).

Runs ``ragas_faithfulness`` over the frozen Acme qrels and gates on the bootstrap 95%
CI **lower bound** (not the mean) via the shipped ``bootstrap_ci``. This is the
mechanism that turns a point-estimate "passes" into an auditable, statistically
defensible claim -- the answer to an auditor asking "at what confidence?". It is the
prerequisite (alongside Tier 2) for any unqualified *external* "production-ready"
assertion.

LIVE ONLY (needs the RAGAS judge LLM): self-skips under ``--mock`` via
``t.require_live()``. Scale the frozen set from the hand-authored N to N>=100 (tighter
CI half-width) via ``scripts/generate_rag_golden.py``, then commit the regenerated
``evals/fixtures/acme_qrels.json`` and lower the ``half_width`` target.

Thresholds PROVISIONAL (uncalibrated).
"""

import json
from pathlib import Path

from koboi.eval.t import Matches, Severity  # noqa: F401  (Severity re-exported for authors)
from koboi.eval.scorers.ci import bootstrap_ci
from koboi.types import EvalCase

CONFIG = {
    "agent": {
        "name": "ragas-golden-suite-eval",
        "description": "Live statistical-confidence probe over the Acme golden qrels",
        "system_prompt": (
            "Use ONLY the provided context to answer. If the context doesn't contain "
            "the answer, say you don't know."
        ),
        "max_iterations": 4,
    },
    "llm": {
        "provider": "openai",
        "model": "${OPENAI_MODEL:gpt-4o-mini}",
        "api_key": "${OPENAI_API_KEY:dummy}",
        "base_url": "${OPENAI_BASE_URL:}",
    },
    "rag": {
        "enabled": True,
        "chunker": "paragraph",
        "retriever": "keyword",
        "top_k": 5,
        "augmentation": "on_the_fly",
        "documents": [
            {"path": "./data/sample/company_policy.md"},
            {"path": "./data/sample/employee_handbook.md"},
            {"path": "./data/sample/product_catalog.md"},
        ],
    },
}

TAGS = ["rag", "live", "faithfulness", "ci", "golden"]
_QRELS = json.loads(Path("evals/fixtures/acme_qrels.json").read_text())["qrels"]


async def test_faithfulness_ci_lower_bound(t):
    """95% CI lower bound of per-query faithfulness must clear the gate (not the mean)."""
    if not t.require_live():
        return

    import koboi.eval  # noqa: F401 - ensures ragas_* scorers are registered
    from koboi.eval.registry import ScorerRegistry

    scorer = ScorerRegistry.create("ragas_faithfulness")
    samples: list[float] = []
    for q in _QRELS:
        await t.send(q["query"])
        rag = (t.last.metadata or {}).get("rag_results", []) or []
        case = EvalCase(
            name="golden",
            user_message=q["query"],
            context_docs=[str(c.get("content", "")) for c in rag],
            expected_answer=" ".join(q["gold_needles"]),
        )
        score = await scorer.score(case, t.reply, {})
        samples.append(score.value)

    ci = bootstrap_ci(samples)
    t.check(
        ci.lower,
        Matches(
            fn=lambda v: v >= 0.80,
            description=f"faithfulness 95%-CI lower bound >= 0.80 (N={ci.n})",
        ),
        name="faithfulness_ci_lower_bound",
        severity=Severity.GATE,
    )
    # Half-width is SOFT at the hand-authored N; tighten the target after scaling to N>=100.
    t.check(
        ci.half_width,
        Matches(fn=lambda w: w <= 0.20, description="CI half-width <= 0.20 (scale to N>=100 to tighten)"),
        name="faithfulness_ci_half_width",
        severity=Severity.SOFT,
    )
