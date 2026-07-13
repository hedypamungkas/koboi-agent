"""Tier 2 production smoke (Q3): comparative -- both subjects covered, balanced.

Catches one-sided research (the agent deep-diving one subject and short-shrifting the other).
``coverage_map`` is not surfaced to event metadata, so this asserts both subjects appear in the
report + a source count on each side via the full bar. Live + real provider (Firecrawl + gpt-5.4).

Run:  FIRECRAWL_API_KEY=... OPENAI_API_KEY=... OPENAI_MODEL=gpt-5.4 \
      koboi eval-test evals/deep_research_prod_comparative.eval.py --strict
"""

import os

from koboi.eval.t import Contains, Severity

CONFIG = {
    "agent": {
        "name": "deep-research-prod-comparative",
        "description": "Production smoke: comparative deep research",
        "system_prompt": "You plan and run iterative, cited web research.",
        "mode": "act",
        "max_iterations": 20,
    },
    "llm": {
        "provider": "openai",
        "model": os.getenv("OPENAI_MODEL", "gpt-5.4"),
        "api_key": os.getenv("OPENAI_API_KEY", "dummy"),
        "base_url": os.getenv("OPENAI_BASE_URL", ""),
    },
    "orchestration": {"enabled": True, "execution": {"mode": "deep_research"}},
    "research": {"max_depth": 2, "coverage_threshold": 0.7, "max_searches": 40, "max_fetches": 50},
    "websearch": {
        "search": {"provider": "firecrawl", "firecrawl": {"api_key": os.getenv("FIRECRAWL_API_KEY", "")}},
        "fetch": {"provider": "firecrawl", "firecrawl": {"api_key": os.getenv("FIRECRAWL_API_KEY", "")}},
    },
    "sandbox": {"backend": "restricted", "workdir": "./workspace"},
}

MOCK_RESPONSES = []
TAGS = ["deep_research", "live", "prod", "comparative"]


async def test_comparative_covers_both_subjects(t):
    """A comparative query must cover BOTH subjects with citations -- not one-sided."""
    if not t.live_ready(extra=None):
        return
    await t.send(
        "Compare Rust vs Go for production systems programming in 2026: performance, "
        "concurrency model, ecosystem maturity, and where each is the better choice."
    )
    t.completed()
    GATE = Severity.GATE
    # Both subjects appear in the report.
    t.check(t.reply, Contains("Rust"), name="mentions Rust", severity=GATE)
    t.check(t.reply, Contains("Go"), name="mentions Go", severity=GATE)
    # Decomposition + grounding bar.
    md = t.last.metadata or {}
    t.check(md.get("plan_nodes"), lambda n: n is not None and n >= 4, name="plan_nodes>=4", severity=GATE)
    t.check(md.get("nodes_failed"), lambda n: n == 0, name="nodes_failed==0", severity=GATE)
    t.check(len(md.get("research_sources") or []), lambda n: n >= 3, name="sources>=3", severity=GATE)
    t.citation(min_citations=4)
    if t.require_live(extra="ragas"):
        await t.judge("deep_research_faithfulness", min_score=0.7, name="faithfulness>=0.7")
