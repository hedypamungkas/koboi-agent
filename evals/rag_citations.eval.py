"""Mock-safe citation-grounding gate (Tier 1).

The RAG augmentation emits numbered citations (``[1] [Source: company_policy.md]\\n...``).
``t.citationResolves`` checks the format-vs-correctness property the format-only tests
miss: every ``[n]`` marker in the answer must resolve to a chunk that was actually
retrieved (no hallucinated/out-of-range sources). A live ALCE-style citation-precision
leg (NLI-checking each cited span against its chunk) is deferred to Tier 2.

Mock-safe: the scripted reply carries the citation markers we assert on; resolution
is checked against ``rag_results`` (pre-LLM).

Run:  koboi eval-test evals/rag_citations.eval.py --mock --strict
"""

from koboi.eval.t import Matches, Severity, scripted_response
from koboi.eval.scorers.citation_grounding import citation_precision

CONFIG = {
    "agent": {
        "name": "rag-citations-eval",
        "description": "Citation resolution probe (mock-deterministic)",
        "system_prompt": "Use the provided context to answer and cite sources as [1], [2].",
        "max_iterations": 4,
    },
    "llm": {"provider": "openai", "model": "gpt-4o-mini", "api_key": "dummy"},
    "rag": {
        "enabled": True,
        "chunker": "paragraph",
        "retriever": "keyword",
        "top_k": 3,
        "augmentation": "on_the_fly",
        "documents": [
            {"path": "./data/sample/company_policy.md"},
            {"path": "./data/sample/employee_handbook.md"},
            {"path": "./data/sample/product_catalog.md"},
        ],
    },
}

MOCK_RESPONSES = [
    scripted_response("Permanent employees get 12 days of annual leave [1]."),
]
TAGS = ["smoke", "rag", "citations"]


async def test_citation_one_resolves(t):
    """The reply's [1] must map to a retrieved chunk (1 <= 1 <= len(rag_results))."""
    await t.send("How many annual leave days does a permanent employee get?")
    t.citationResolves(1)
    t.completed()


async def test_all_citations_resolve(t):
    """Every [n] marker in the reply must be in range."""
    await t.send("What is the remote work policy at Acme Corp?")
    t.citationResolves()
    t.completed()


async def test_dangling_citation_is_detected(t):
    """A dangling [9] (with only ~3 chunks retrieved) must score precision < 1.0.

    Proves ``citationResolves``/``citation_precision`` is not vacuous -- it returns
    < 1.0 when a citation does not resolve.
    """
    fake_rag = [{"source": "company_policy.md"}, {"source": "employee_handbook.md"}, {"source": "product_catalog.md"}]
    precision, _resolved, _total = citation_precision("See [9] for the unsupported claim.", fake_rag)
    t.check(
        precision,
        Matches(fn=lambda p: p < 1.0, description="dangling [9] -> precision < 1.0"),
        name="dangling_citation_detected",
        severity=Severity.GATE,
    )
