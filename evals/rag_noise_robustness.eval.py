"""Mock-safe noise-robustness gate (Tier 1).

Asserts that a distinctive gold chunk ranks highly despite corpus noise -- e.g. the
annual-leave fact '12 days' appears in BOTH the policy (permanent employees) and the
handbook (paid time off: 12 days annually), so a query targeting *permanent* leave
must surface the policy chunk near the top, not be buried by the handbook PTO line.

The live leg (faithfulness drop <= 5% when distractors are injected -- the
lost-in-the-middle guard via ragas noise_sensitivity) lives in
``evals/rag_noise_faithfulness.eval.py`` (a manual live run; there is no automated nightly job).

Mock-safe (no LLM): retrieval is pre-LLM.

Run:  koboi eval-test evals/rag_noise_robustness.eval.py --mock --strict
"""

from koboi.eval.t import scripted_response

CONFIG = {
    "agent": {
        "name": "rag-noise-eval",
        "description": "Noise-robustness / disambiguation probe (mock-deterministic)",
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
TAGS = ["smoke", "rag", "noise"]


async def test_permanent_leave_ranks_above_pto_noise(t):
    """The policy chunk ('Permanent employees: 12 days') must rank high (rank <= 2),
    not be displaced by the handbook's 'Paid time off: 12 days annually' line."""
    await t.send("How many annual leave days are permanent employees entitled to?")
    t.rankingMetric("permanent", k=10, metric="mrr", min_score=0.5)
    t.completed()


async def test_top5_has_relevant_chunk(t):
    """At least one of the top-5 retrieved chunks must be relevant (precision@5 >= 0.2)."""
    await t.send("What is the price of the AcmeCRM Business product?")
    t.rankingMetric(["AcmeCRM", "$25"], k=5, metric="precision", min_score=0.2)
    t.completed()
