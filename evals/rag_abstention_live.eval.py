"""Live negative-rejection / refusal-correctness leg (Tier 2 polish; w0.09).

The Tier-1 mock abstention eval covers the RETRIEVAL leg (empty/spurious retrieval).
This live leg covers the ANSWER leg: on an out-of-scope query (whose stopwords retrieve
spurious chunks -- the keyword retriever has no stopword filter), the MODEL must still
*refuse* rather than confabulate. ``t.abstains`` passes on empty retrieval OR a refusal
marker; in live mode this therefore asserts the model actually abstains.

LIVE ONLY; self-skips under ``--mock`` via ``t.require_live()``. Threshold: refusal is a
GATE (the model must not hallucinate on OOS).
"""

from koboi.eval.t import Severity  # noqa: F401  (re-exported for eval authors)

CONFIG = {
    "agent": {
        "name": "rag-abstention-live-eval",
        "description": "Live refusal-correctness probe on an OOS query",
        "system_prompt": (
            "Use ONLY the provided context to answer. If the context doesn't contain "
            "the answer, say you don't know. Do not make up information."
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

TAGS = ["rag", "live", "abstention"]


async def test_model_refuses_on_oos_query(t):
    """An OOS query (spurious retrieval due to no stopword filter) must still be refused."""
    if not t.require_live():
        return
    await t.send("Explain the mating rituals of deep-sea anglerfish per Acme policy.")
    t.abstains()  # GATE: empty retrieval OR refusal marker
    t.completed()
