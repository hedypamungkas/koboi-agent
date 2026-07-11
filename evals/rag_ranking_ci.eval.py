"""Mock-safe statistical-confidence leg for retrieval (Tier 1).

Runs the keyword retriever over the frozen Acme qrels
(``evals/fixtures/acme_qrels.json``), collects per-query Recall@10, and gates on the
**bootstrap 95% CI lower bound** -- not the point estimate. This is the mechanism that
turns a hand-waved 'passes' into an auditable, statistically defensible claim (the
answer to an auditor asking 'at what confidence?').

NOTE: N=24 yields a wide half-width; the Tier-3 pre-release suite regenerates the
golden set at N>=100 via RAGASDataGenerator for tight bounds. Tier 1 demonstrates the
gating mechanism over the deterministic, no-API-key qrels.

Mock-safe (no LLM): retrieval is pre-LLM.

Run:  koboi eval-test evals/rag_ranking_ci.eval.py --mock --strict
"""

import json
from pathlib import Path

from koboi.eval.t import Matches, Severity, scripted_response
from koboi.eval.scorers.retrieval_metric import recall_at_k
from koboi.eval.scorers.ci import bootstrap_ci

CONFIG = {
    "agent": {
        "name": "rag-ranking-ci-eval",
        "description": "Statistical-confidence probe over the frozen Acme qrels",
        "system_prompt": "Use the provided context to answer.",
        "max_iterations": 4,
    },
    "llm": {"provider": "openai", "model": "gpt-4o-mini", "api_key": "dummy"},
    "rag": {
        "enabled": True,
        "chunker": "paragraph",
        "retriever": "keyword",
        "top_k": 10,
        "augmentation": "on_the_fly",
        "documents": [
            {"path": "./data/sample/company_policy.md"},
            {"path": "./data/sample/employee_handbook.md"},
            {"path": "./data/sample/product_catalog.md"},
        ],
    },
}

# One scripted terminal reply per qrel query (retrieval is what we measure, not the
# reply). Extra slack avoids index exhaustion across the loop.
MOCK_RESPONSES = [scripted_response("ok")] * 60
TAGS = ["rag", "ranking", "ci"]

_QRELS = json.loads(Path("evals/fixtures/acme_qrels.json").read_text())["qrels"]


async def test_recall_at_10_ci_lower_bound(t):
    """95% CI lower bound of per-query Recall@10 must clear the gate (not the mean)."""
    samples: list[float] = []
    for q in _QRELS:
        await t.send(q["query"])
        rag = (t.last.metadata or {}).get("rag_results", []) or []
        retrieved = [str(c.get("content", "")) for c in rag]
        samples.append(recall_at_k(retrieved, q["gold_needles"], 10))

    ci = bootstrap_ci(samples)
    t.check(
        ci.lower,
        Matches(fn=lambda v: v >= 0.80, description="recall@10 95%-CI lower bound >= 0.80"),
        name="recall_at_10_ci_lower_bound",
        severity=Severity.GATE,
    )
    # Half-width is reported (SOFT): at N=24 it is intentionally loose; Tier 3 tightens.
    t.check(
        ci.half_width,
        Matches(fn=lambda w: w <= 0.40, description="CI half-width <= 0.40 (N=24; Tier 3 tightens)"),
        name="recall_ci_half_width",
        severity=Severity.SOFT,
    )
