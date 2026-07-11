"""Mock-safe RAG ranking-quality gate (Tier 1).

Asserts real IR ranking metrics (Recall@k / MRR / nDCG@k) over the FULL agent
pipeline (augmentation -> keyword retriever -> rag_results stamp) via
``t.rankingMetric``. This is the rank-aware counterpart to ``t.retrievedChunk``,
which is Hit@k=infinity: it passes whenever the gold appears *anywhere* in the
retrieved set and cannot detect a gold chunk buried under distractors.

Mock-safe: retrieval is a pre-LLM prompt-augmentation step, so it is fully
deterministic under the scripted client (the reply text is irrelevant).

Run:  koboi eval-test evals/rag_ranking.eval.py --mock --strict
"""

from koboi.eval.t import Severity, scripted_response

CONFIG = {
    "agent": {
        "name": "rag-ranking-eval",
        "description": "Ranking-quality probe (keyword retriever, mock-deterministic)",
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

MOCK_RESPONSES = [scripted_response("Retrieved relevant context.")]
TAGS = ["smoke", "rag", "ranking"]


async def test_recall_at_10_annual_leave(t):
    """The permanent-leave chunk (gold '12 days') must appear in the top-10."""
    await t.send("How many annual leave days does a permanent employee get?")
    t.rankingMetric("12 days", k=10, metric="recall", min_score=1.0)
    t.completed()


async def test_ceo_within_top_k(t):
    """GATE: the org-structure chunk (gold 'John Smith') must be within the top-k
    window (hit@10 = 1.0) -- a retriever/index change that drops it below top-10 fails
    here. SOFT: the actual MRR is reported so the real rank (~6 for the keyword
    retriever on this entity query) is visible without gating on it."""
    await t.send("Who is the CEO of Acme Corp?")
    t.rankingMetric("John Smith", k=10, metric="hit", min_score=1.0)
    t.rankingMetric("John Smith", k=10, metric="mrr", min_score=0.0, severity=Severity.SOFT)
    t.completed()


async def test_ndcg_price_query(t):
    """The AcmeERP chunk must rank highly (nDCG@10 >= 0.5)."""
    await t.send("What is the price of AcmeERP Enterprise?")
    t.rankingMetric(["AcmeERP", "$15,000"], k=10, metric="ndcg", min_score=0.5)
    t.completed()


async def test_gold_not_buried_regression(t):
    """Regression guard ``t.retrievedChunk`` cannot catch: gold must stay within the
    top-k window. ``retrievedChunk`` passes if gold appears *anywhere* (Hit@k=infinity);
    ``hit@10`` fails when a retriever change buries the parental-leave fact ('12 weeks')
    below top-10. The actual MRR is reported SOFT (keyword retriever ranks this ~6)."""
    await t.send("What is the parental leave duration at Acme Corp?")
    t.rankingMetric("12 weeks", k=10, metric="hit", min_score=1.0)
    t.rankingMetric("12 weeks", k=10, metric="mrr", min_score=0.0, severity=Severity.SOFT)
    t.completed()
