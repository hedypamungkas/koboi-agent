"""Live noise-robustness faithfulness leg (Tier 2 polish; w0.09).

Loads the Acme corpus PLUS an off-topic distractor fixture
(``evals/fixtures/distractor_noise.md``) whose vocabulary overlaps real policy terms
("$50/month", "30 days", "12 weeks", "annual", "employees") so it competes for
retrieval. The model must stay FAITHFUL to the real policy despite the injected noise
(the "lost-in-the-middle" robustness check -- arXiv:2307.03001). Asserts
``ragas_faithfulness`` clears the gate.

LIVE ONLY; self-skips under ``--mock``. Threshold PROVISIONAL (uncalibrated).
"""

from koboi.eval.t import Severity  # noqa: F401  (re-exported for eval authors)

CONFIG = {
    "agent": {
        "name": "rag-noise-faithfulness-eval",
        "description": "Live faithfulness-under-noise probe (Acme + distractor fixture)",
        "system_prompt": (
            "Use ONLY the provided context to answer. Base your answer on the Acme "
            "Corp policy documents, not any distractor content. If unsure, say so."
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
            {"path": "./evals/fixtures/distractor_noise.md"},
        ],
    },
}

TAGS = ["rag", "live", "noise", "faithfulness"]


async def test_faithful_under_injected_noise(t):
    """The annual-leave answer must stay faithful despite distractor passages (>= 0.8)."""
    if not t.require_live():
        return
    await t.send("How many annual leave days does a permanent employee get?")
    await t.judge("ragas_faithfulness", min_score=0.8, expected_answer="12 days per year", severity=Severity.SOFT)
    t.completed()
