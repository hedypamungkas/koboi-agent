"""Mock-safe HARD gate for the cross-encoder rerank WIRING (Tier 1).

Proves the cross-encoder wrapper is wired into the pipeline AND invoked end-to-end
through the loop -- without any network egress or cost. Technique: point the rerank
backend at an unreachable local address (``127.0.0.1:1``) with a dummy key and
``fallback: false``. Connection is refused ~instantly (not a 30s timeout), the backend
fails soft (returns None), and the wrapper stamps ``retrieval_method='rerank:failed(jina,...)'``.
Asserting that stamp is present proves the wrapper ran; asserting the gold still
surfaces proves fail-soft preserved base retrieval (retrieval never breaks on a rerank
hiccup). Correctness (response parsing) is the unit tests' job (tests/test_rag_rerank.py).

Mock-safe: retrieval is a pre-LLM step, fully deterministic under the scripted client.

Run:  koboi eval-test evals/rag_rerank_wiring.eval.py --mock --strict
"""

from koboi.eval.t import Severity, scripted_response

CONFIG = {
    "agent": {
        "name": "rag-rerank-wiring-eval",
        "description": "Cross-encoder rerank wiring probe (mock-safe, zero egress)",
        "system_prompt": "Use the provided context to answer.",
        "max_iterations": 4,
    },
    "llm": {"provider": "openai", "model": "gpt-4o-mini", "api_key": "dummy"},
    "rag": {
        "enabled": True,
        "chunker": "paragraph",
        "retriever": "bm25",
        "top_k": 3,
        "augmentation": "on_the_fly",
        # Dict -> cross-encoder path. Unreachable base_url + dummy key + fallback=false:
        # the call fails fast (ECONNREFUSED), the wrapper stamps 'rerank:failed(jina,...)',
        # proving it is wired + invoked. No real API call, no cost, no egress.
        "rerank": {
            "provider": "jina",
            "api_key": "dummy-key",
            "base_url": "http://127.0.0.1:1",
            "timeout": 1.0,
            "fallback": False,
        },
        "documents": [
            {"path": "./data/sample/company_policy.md"},
            {"path": "./data/sample/employee_handbook.md"},
            {"path": "./data/sample/product_catalog.md"},
        ],
    },
}

MOCK_RESPONSES = [scripted_response("Retrieved relevant context.")]
TAGS = ["smoke", "rag", "rerank"]


async def test_rerank_wrapper_is_invoked(t):
    """GATE: the cross-encoder wrapper must run and stamp ``rerank:...`` on the results.
    A wiring regression (e.g. the dict branch in build_rag dropped, or the wrapper bypassed)
    leaves ``retrieval_method`` as the bare base method (``bm25``) -- this fails here."""
    await t.send("How many annual leave days does a permanent employee get?")
    rag = (t.last.metadata or {}).get("rag_results", []) or []
    methods = {str(r.get("retrieval_method", "")) for r in rag}
    from koboi.eval.t import Matches

    t.check(
        any("rerank:" in m for m in methods),
        Matches(
            fn=lambda v: v,
            description=f"cross-encoder rerank invoked (some retrieval_method starts with 'rerank:'); got {sorted(methods)}",
        ),
        name="rerank_wrapper_invoked",
        severity=Severity.GATE,
    )
    t.completed()


async def test_failsoft_preserves_retrieval(t):
    """GATE: even with the rerank backend failing, retrieval must still surface the gold
    fact -- fail-soft returns the base retriever's results so the pipeline never breaks."""
    await t.send("How many annual leave days does a permanent employee get?")
    t.rankingMetric("12 days", k=3, metric="recall", min_score=1.0)
    t.completed()
