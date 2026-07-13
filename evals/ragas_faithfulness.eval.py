"""Live RAGAS faithfulness gate (Tier 2; CRITICAL dimension, weight 0.18).

Reuses the shipped, registered ``RAGASScorer`` via ``t.judge`` over the Acme corpus.
Faithfulness decomposes the answer into atomic claims and NLI-checks each against the
retrieved context -- the single most-cited production gate across Ragas/TruLens/ARES,
and the dimension the pre-existing eval method could not evidence (the scorer was
shipped but never invoked).

LIVE ONLY: self-skips under ``--mock`` or a bare install via ``t.require_live()``, so
the mock PR gate (``eval-test evals/ --mock --strict``) stays green. Runs for real on
a manual run::

    pip install -e ".[eval-ragas]"
    koboi eval-test evals/ --tags live

Thresholds (``min_score``) are PROVISIONAL -- calibrate against real runs. Judge
severity is SOFT (non-deterministic) until calibrated.
"""

from koboi.eval.t import Severity  # noqa: F401  (re-exported for eval authors)

CONFIG = {
    "agent": {
        "name": "ragas-faithfulness-eval",
        "description": "Live RAGAS faithfulness probe over the Acme corpus",
        "system_prompt": (
            "Use ONLY the provided context to answer. If the context doesn't contain "
            "the answer, say you don't know. Do not state facts that aren't in the context."
        ),
        "max_iterations": 4,
    },
    "llm": {
        "provider": "openai",
        "model": "${OPENAI_MODEL:gpt-4o-mini}",
        "api_key": "${OPENAI_API_KEY:dummy}",  # dummy under --mock (client is swapped); real key live
        "base_url": "${OPENAI_BASE_URL:}",
    },
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

# No MOCK_RESPONSES -> live mode. Self-skips under --mock via t.require_live().
TAGS = ["rag", "live", "faithfulness"]


async def test_faithfulness_annual_leave(t):
    """The annual-leave answer must be fully grounded in retrieved context (>= 0.9)."""
    if not t.require_live():
        return
    await t.send("How many annual leave days does a permanent employee get?")
    await t.judge("ragas_faithfulness", min_score=0.9, expected_answer="12 days per year", severity=Severity.SOFT)
    t.completed()


async def test_composite_no_hallucination(t):
    """Composite RAGAS (faithfulness + relevancy + precision + recall) >= 0.8."""
    if not t.require_live():
        return
    await t.send("What is the notice period for resignation at Acme Corp?")
    await t.judge("ragas_composite", min_score=0.8, expected_answer="30 days", severity=Severity.SOFT)
    t.completed()
