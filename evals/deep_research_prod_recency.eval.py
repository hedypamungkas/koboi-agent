"""Tier 2 production smoke (Q2): recency -- sources must reflect recent information.

Catches stale-knowledge leakage: the model answering from training data instead of fetched
sources. Uses the heuristic :class:`RecencyScorer` (fraction of sources mentioning a year within
1 year of today). Live + real provider (Firecrawl + gpt-5.4).

Run:  FIRECRAWL_API_KEY=... OPENAI_API_KEY=... OPENAI_MODEL=gpt-5.4 \
      koboi eval-test evals/deep_research_prod_recency.eval.py --strict
"""

import os

from koboi.eval.scorers.recency_scorer import RecencyScorer
from koboi.eval.t import Severity

CONFIG = {
    "agent": {
        "name": "deep-research-prod-recency",
        "description": "Production smoke: recency-sensitive deep research",
        "system_prompt": "You plan and run iterative, cited web research.",
        "mode": "act",
        "max_iterations": 20,
    },
    "llm": {
        "provider": "openai",
        "model": os.getenv("OPENAI_MODEL", "gpt-5.4"),
        "api_key": os.getenv("OPENAI_API_KEY", "dummy"),
        "base_url": os.getenv("OPENAI_BASE_URL", ""),
        "timeout": 300,  # gpt-5.4 planning via a slow gateway can exceed the 120 default
        "max_retries": 3,
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
TAGS = ["deep_research", "live", "prod", "recency"]


async def test_recency_sources_are_recent(t):
    """A recency-sensitive query must surface sources from within the last ~1 year."""
    if not t.live_ready(extra=None):
        return
    await t.send(
        "Research the most important developments in open-source AI agent frameworks "
        "over the last year, with specific dates and version numbers."
    )
    t.completed()
    # At least half the gathered sources mention a year within 1 year of today.
    await t.judge(RecencyScorer(recent_years=1), min_score=0.5, name="recency>=0.5")
    # Plus the report itself references a recent year.
    t.check(t.reply, lambda r: "2025" in r or "2026" in r, name="report mentions a recent year", severity=Severity.GATE)
