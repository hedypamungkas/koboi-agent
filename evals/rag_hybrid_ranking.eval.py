"""Live hybrid (RRF) retrieval ranking gate (Tier 2 tail; HIGH dimension, weight 0.17).

HybridReciprocal Rank Fusion fuses the keyword and semantic legs; a target that ranks
low on keyword but high on semantic must be promoted into the top-k, and
``retrieval_method`` must be ``"hybrid"``. LIVE ONLY (needs embeddings); self-skips
under ``--mock``. Thresholds PROVISIONAL.
"""

from koboi.eval.t import Matches, Severity  # noqa: F401  (re-exported for eval authors)

CONFIG = {
    "agent": {
        "name": "rag-hybrid-ranking-eval",
        "description": "Live hybrid (RRF) retrieval ranking probe (real embeddings)",
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
        "retriever": "hybrid",
        "top_k": 5,
        "augmentation": "on_the_fly",
        "documents": [
            {"path": "./data/sample/company_policy.md"},
            {"path": "./data/sample/employee_handbook.md"},
            {"path": "./data/sample/product_catalog.md"},
        ],
    },
}

TAGS = ["rag", "live", "hybrid"]


async def test_rrf_promotes_semantic_hit(t):
    """RRF must fuse keyword + semantic legs and return retrieval_method == 'hybrid',
    with the target chunk in the top-k."""
    if not t.require_live(extra=None):
        return
    await t.send("How many vacation days do permanent staff get?")
    rag = (t.last.metadata or {}).get("rag_results", []) or []
    methods = {str(c.get("retrieval_method", "")) for c in rag}
    t.check(
        methods,
        Matches(fn=lambda m: any(x == "hybrid" for x in m), description="retrieval_method == 'hybrid'"),
        name="hybrid_method",
        severity=Severity.GATE,
    )
    t.rankingMetric("12 days", k=5, metric="recall", min_score=1.0)
    t.completed()
