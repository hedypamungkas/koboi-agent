"""Mock-safe metadata-filter / relevance-scoping gate (Tier 1).

``rag.filter`` constrains which chunks a retriever considers (relevance scoping --
freshness/source/type; NOT an ACL boundary). Exercises the keyword retriever's
``metadata_filter`` directly with controlled chunk metadata: equality scoping, ``$in``
multi-source, and exclusion of non-matching chunks.

No agent drive; the retriever is exercised directly and asserted via ``t.check``.

Run:  koboi eval-test evals/rag_metadata_filter.eval.py --mock --strict
"""

from koboi.eval.t import Matches, Severity, scripted_response
from koboi.rag.retriever import KeywordRetriever
from koboi.rag.types import Chunk

CONFIG = {
    "agent": {
        "name": "rag-metadata-filter-eval",
        "description": "Metadata-filter / relevance-scoping probe",
        "system_prompt": "Use the provided context to answer.",
        "max_iterations": 4,
    },
    "llm": {"provider": "openai", "model": "gpt-4o-mini", "api_key": "dummy"},
}

MOCK_RESPONSES = [scripted_response("ok")]
TAGS = ["rag", "metadata_filter"]


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            id="p1",
            doc_id="company_policy.md",
            content="Annual leave for permanent employees is 12 days per year.",
            metadata={"source": "company_policy.md"},
        ),
        Chunk(
            id="h1",
            doc_id="employee_handbook.md",
            content="Paid time off: 12 days annually for all staff.",
            metadata={"source": "employee_handbook.md"},
        ),
        Chunk(
            id="c1",
            doc_id="product_catalog.md",
            content="AcmeCRM Business price is 25 per user per month.",
            metadata={"source": "product_catalog.md"},
        ),
    ]


async def test_equality_filter_scopes_to_policy(t):
    """{source: company_policy.md} returns only policy chunks."""
    results = await KeywordRetriever(_chunks()).retrieve(
        "annual leave days", top_k=10, metadata_filter={"source": "company_policy.md"}
    )
    sources = {r.chunk.metadata.get("source") for r in results}
    t.check(
        sources,
        Matches(fn=lambda s: s == {"company_policy.md"}, description="only company_policy.md"),
        name="equality_scope",
        severity=Severity.GATE,
    )


async def test_in_filter_multi_source(t):
    """{$in: [policy, catalog]} returns only those two sources (excludes handbook)."""
    results = await KeywordRetriever(_chunks()).retrieve(
        "leave price days", top_k=10, metadata_filter={"source": {"$in": ["company_policy.md", "product_catalog.md"]}}
    )
    sources = {r.chunk.metadata.get("source") for r in results}
    t.check(
        sources,
        Matches(fn=lambda s: "employee_handbook.md" not in s, description="handbook excluded by $in"),
        name="in_filter_excludes",
        severity=Severity.GATE,
    )


async def test_filter_does_not_shrink_relevance(t):
    """A filter matching nothing yields empty retrieval (graceful, not an error)."""
    results = await KeywordRetriever(_chunks()).retrieve(
        "annual leave", top_k=10, metadata_filter={"source": "nonexistent.md"}
    )
    t.check(
        len(results),
        Matches(fn=lambda n: n == 0, description="no chunks match -> empty"),
        name="no_match_empty",
        severity=Severity.GATE,
    )
