"""Live HyDE (Hypothetical Document Embeddings) probe (Tier 2 polish; w0.17 sub-leg).

HyDE generates a hypothetical answer for the query, embeds it, and uses that for the
semantic leg -- improving recall on hard paraphrases whose vocabulary differs from the
corpus. This eval evidences the capability end-to-end: with ``rag.hyde: true`` on a
hybrid retriever, ``rag_rewrite`` metadata is populated (HyDE actually ran) and the
target is retrieved. A paired recall-lift measurement (hyde:true vs false over a query
set) is a future refinement; this proves HyDE runs and retrieves.

LIVE ONLY (needs the embedding endpoint + a chat client for the hypothetical); self-skips
under ``--mock`` via ``t.require_live(extra=None)``. Threshold PROVISIONAL.
"""

from koboi.eval.t import Matches, Severity  # noqa: F401  (re-exported for eval authors)

CONFIG = {
    "agent": {
        "name": "rag-hyde-eval",
        "description": "Live HyDE probe (hybrid + hyde:true, real embeddings)",
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
        "model": "${OPENAI_EMBEDDING_MODEL:text-embedding-3-small}",
        "api_key": "${OPENAI_API_KEY:dummy}",
        "base_url": "${OPENAI_BASE_URL:}",
    },
    "rag": {
        "enabled": True,
        "chunker": "paragraph",
        "retriever": "hybrid",
        "top_k": 5,
        "augmentation": "on_the_fly",
        "hyde": True,
        "documents": [
            {"path": "./data/sample/company_policy.md"},
            {"path": "./data/sample/employee_handbook.md"},
            {"path": "./data/sample/product_catalog.md"},
        ],
    },
}

TAGS = ["rag", "live", "hyde"]


async def test_hyde_produces_rewrite_and_retrieves(t):
    """HyDE must run (rag_rewrite populated) and the target must be retrieved."""
    if not t.require_live(extra=None):
        return
    await t.send("How many vacation days do permanent staff get?")
    rw = (t.last.metadata or {}).get("rag_rewrite")
    t.check(
        rw,
        Matches(fn=lambda x: bool(x), description="rag_rewrite populated (HyDE ran)"),
        name="hyde_rewrite_populated",
        severity=Severity.GATE,
    )
    t.rankingMetric("12 days", k=5, metric="recall", min_score=1.0)
    t.completed()
