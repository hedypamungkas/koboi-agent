"""Live end-to-end answer-correctness gate (Tier 2; CRITICAL dimension, weight 0.13).

Reuses the registered ``ragas_recall`` / ``ragas_relevancy`` / ``ragas_precision``
scorers via ``t.judge``. This is the dimension the pre-existing RAG eval explicitly
skipped ("we assert on retrieval, not the answer") -- nothing evidenced that the
grounded pipeline produces a *correct* answer.

LIVE ONLY: self-skips under ``--mock`` / bare install via ``t.require_live()``. Runs
on the ``eval-ragas-nightly`` job (``pip install -e ".[eval-ragas]"`` +
``koboi eval-test evals/ --tags live``). Thresholds PROVISIONAL; SOFT until calibrated.
"""

from koboi.eval.t import Severity  # noqa: F401  (re-exported for eval authors)

CONFIG = {
    "agent": {
        "name": "rag-answer-correctness-eval",
        "description": "Live answer-correctness probe over the Acme corpus",
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

TAGS = ["rag", "live", "answer_correctness"]


async def test_factual_recall_annual_leave(t):
    """Context recall: the reference answer's claims must be supported by context (>= 0.8)."""
    if not t.require_live():
        return
    await t.send("How many annual leave days does a permanent employee get?")
    await t.judge("ragas_recall", min_score=0.8, expected_answer="12 days", severity=Severity.SOFT)
    t.completed()


async def test_answer_relevancy(t):
    """The answer must be relevant to the question (ragas_relevancy >= 0.7)."""
    if not t.require_live():
        return
    await t.send("Who is the CEO of Acme Corp?")
    await t.judge("ragas_relevancy", min_score=0.7, severity=Severity.SOFT)
    t.completed()


async def test_contract_not_permanent_disambiguation(t):
    """Negative disambiguation: the contract-employee answer is 6 days, NOT permanent 12."""
    if not t.require_live():
        return
    await t.send("How many annual leave days is a contract employee entitled to?")
    await t.judge("ragas_recall", min_score=0.8, expected_answer="6 days", severity=Severity.SOFT)
    t.completed()
