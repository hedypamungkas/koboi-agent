"""Live semantic-retrieval ranking gate (Tier 2 tail; HIGH dimension, weight 0.17).

Evidences the SemanticRetriever headline capability with REAL embeddings: a paraphrase
whose vocabulary differs from the corpus (so keyword retrieval misses) must surface the
target chunk, AND ``retrieval_method`` must be ``"semantic"`` (not the silent
``"semantic (fallback to keyword)"`` degrade). Pre-change, every semantic/hybrid test
ran on mock/None embeddings -- this is the first eval that proves real embedding recall.

LIVE ONLY (needs the embedding endpoint via an ``embedding:`` config): self-skips under
``--mock`` via ``t.require_live(extra=None)``. Run on eval-ragas-nightly (set
``OPENAI_API_KEY`` + ``OPENAI_EMBEDDING_MODEL``). Thresholds PROVISIONAL (uncalibrated).
"""

from koboi.eval.t import Matches, Severity  # noqa: F401  (re-exported for eval authors)

CONFIG = {
    "agent": {
        "name": "rag-semantic-ranking-eval",
        "description": "Live semantic-retrieval ranking probe (real embeddings)",
        "system_prompt": "Use the provided context to answer.",
        "max_iterations": 4,
    },
    "llm": {
        "provider": "openai",
        "model": "${OPENAI_MODEL:gpt-4o-mini}",
        "api_key": "${OPENAI_API_KEY:dummy}",
        "base_url": "${OPENAI_BASE_URL:}",
    },
    "embedding": {
        "provider": "openai",
        "model": "${EMBEDDING_MODEL:text-embedding-3-small}",
        "api_key": "${EMBEDDING_API_KEY:}",
        "base_url": "${EMBEDDING_BASE_URL:}",
    },
    "rag": {
        "enabled": True,
        "chunker": "paragraph",
        "retriever": "semantic",
        "top_k": 5,
        "augmentation": "on_the_fly",
        "documents": [
            {"path": "./data/sample/company_policy.md"},
            {"path": "./data/sample/employee_handbook.md"},
            {"path": "./data/sample/product_catalog.md"},
        ],
    },
}

TAGS = ["rag", "live", "semantic"]


async def test_paraphrase_retrieves_target_with_real_embeddings(t):
    """'vacation' is not in the corpus (it says 'annual leave'), so keyword retrieval
    misses; real semantic embeddings must map vacation -> annual leave and rank the
    target chunk, with retrieval_method == 'semantic' (no keyword fallback)."""
    if not t.require_live(extra=None):
        return
    await t.send("How many vacation days do permanent staff get?")
    rag = (t.last.metadata or {}).get("rag_results", []) or []
    methods = {str(c.get("retrieval_method", "")) for c in rag}
    t.check(
        methods,
        Matches(fn=lambda m: any(x == "semantic" for x in m), description="retrieval_method == 'semantic' (no fallback)"),
        name="semantic_method_no_fallback",
        severity=Severity.GATE,
    )
    t.rankingMetric("12 days", k=5, metric="recall", min_score=1.0)
    t.completed()
