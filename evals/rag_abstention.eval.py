"""Mock-safe negative-rejection / abstention gate (Tier 1).

A RAG system is production-safe only if it (a) abstains when the corpus has no
answer and (b) answers when it does. ``t.abstains`` passes when retrieval was empty
OR the reply contains a refusal marker -- the coverage/abstention partner.

- Out-of-scope query (zero term overlap with the corpus) -> empty rag_results +
  refusal -> abstains.
- In-corpus query -> non-empty retrieval + NOT abstaining (coverage leg).
- ``relevance_threshold=0.99`` drops all sub-threshold chunks on a borderline query
  (empty-retrieval path guard), exercised directly on the augmentation.

Mock-safe (no LLM): retrieval + threshold filtering are pre-LLM.

Run:  koboi eval-test evals/rag_abstention.eval.py --mock --strict
"""

from koboi.eval.t import Equals, Matches, Severity, scripted_response

CONFIG = {
    "agent": {
        "name": "rag-abstention-eval",
        "description": "Negative-rejection / abstention probe (mock-deterministic)",
        "system_prompt": "Use the provided context to answer. If the context doesn't contain the answer, say so honestly.",
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

MOCK_RESPONSES = [
    scripted_response("I don't have information about that in the provided context."),
]
TAGS = ["smoke", "rag", "abstention"]


async def test_zero_overlap_oos_empty_retrieval(t):
    """A query with zero corpus term overlap must retrieve nothing (the empty path)."""
    await t.send("xyzzy qwerty frobnicate zxcvbn plugh")
    rag = (t.last.metadata or {}).get("rag_results", []) or []
    t.check(len(rag), Equals(0), name="zero_overlap_empty_retrieval", severity=Severity.GATE)


async def test_realistic_oos_abstains(t):
    """A realistic OOS query (with stopwords) may retrieve spurious low-score chunks
    -- the keyword retriever applies NO stopword filtering, so 'the'/'of' match corpus
    chunks (a documented retrieval weakness). The agent must still abstain (refuse)."""
    await t.send("Explain the mating rituals of deep-sea anglerfish.")
    t.abstains()


async def test_in_corpus_query_retrieves(t):
    """Coverage leg: an in-corpus query must retrieve relevant context (non-empty).

    Mock-safe proxy for 'answerable' -- asserting the *reply* correctly uses the
    context needs a live LLM (Tier 2 RAGAS answer-correctness); here we assert only
    that retrieval surfaces the answer-bearing chunk.
    """
    await t.send("How many annual leave days does a permanent employee get?")
    rag = (t.last.metadata or {}).get("rag_results", []) or []
    t.check(
        len(rag) > 0,
        Matches(fn=lambda n: n > 0, description="in-corpus retrieval non-empty"),
        name="in_corpus_retrieval",
        severity=Severity.GATE,
    )


async def test_relevance_threshold_drops_low_score(t):
    """``relevance_threshold=0.99`` filters out all sub-threshold (keyword) chunks."""
    from koboi.rag.augmentation import OnTheFlyAugmentation
    from koboi.rag.retriever import KeywordRetriever
    from koboi.rag.types import Chunk

    chunks = [Chunk(id="c1", doc_id="policy", content="Annual leave for permanent employees is 12 days per year.")]
    aug = OnTheFlyAugmentation(retriever=KeywordRetriever(chunks), top_k=10, relevance_threshold=0.99)
    _context, results = await aug._retrieve_and_format("annual leave days")  # noqa: SLF001
    t.check(len(results), Equals(0), name="threshold_sweep_drops_all", severity=Severity.GATE)
