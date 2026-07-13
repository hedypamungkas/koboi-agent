"""Tier 2 production smoke (Q1): multi-faceted factual deep research -- the FULL bar.

Live, real provider (Firecrawl + gpt-5.4). THE core regression-catcher for the shallow-report
failure mode (1/3 runs once produced 2 nodes / coverage 0.16 / 6.7K). Asserts the quantitative
passing grade from docs/deep-research-smoke.md:

  plan_nodes >= 4, coverage >= 0.6 OR drilled to max_depth, nodes_failed == 0,
  used_searches/fetches <= caps, sources >= 3, citations >= 5 all resolve,
  report >= 8000 chars, faithfulness >= 0.7.

Run:  FIRECRAWL_API_KEY=... OPENAI_API_KEY=... OPENAI_MODEL=gpt-5.4 \
      koboi eval-test evals/deep_research_prod_multifaceted.eval.py --strict
Re-run 3x -- the shallow-report bug was a 1/3 flake; this must hold every run.
"""

import os

from koboi.eval.t import Severity

CONFIG = {
    "agent": {
        "name": "deep-research-prod-multifaceted",
        "description": "Production smoke: multi-faceted factual deep research",
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
    "research": {
        "max_depth": 3,
        "coverage_threshold": 0.7,
        "max_searches": 50,
        "max_fetches": 60,
        "citations": "numbered",
    },
    "websearch": {
        "search": {"provider": "firecrawl", "firecrawl": {"api_key": os.getenv("FIRECRAWL_API_KEY", "")}},
        "fetch": {"provider": "firecrawl", "firecrawl": {"api_key": os.getenv("FIRECRAWL_API_KEY", "")}},
    },
    "sandbox": {"backend": "restricted", "workdir": "./workspace"},
}

MOCK_RESPONSES = []
TAGS = ["deep_research", "live", "prod", "multifaceted"]


async def test_multifaceted_research_meets_bar(t):
    """A multi-faceted research query must hit the full production bar (see module docstring)."""
    if not t.live_ready(extra=None):
        return
    await t.send(
        "Research solid-state battery breakthroughs in 2025-2026: the science, the commercial "
        "players, the technical challenges, and the competitive landscape."
    )
    t.completed()

    md = t.last.metadata or {}
    max_depth = 3
    GATE = Severity.GATE  # the structural bar is a hard contract (a bar miss fails the eval)
    # Decomposition + node health + budget.
    t.check(md.get("plan_nodes"), lambda n: n is not None and n >= 4, name="plan_nodes>=4", severity=GATE)
    t.check(
        (md.get("coverage"), md.get("depth")),
        lambda pair: pair[0] is not None and (pair[0] >= 0.6 or pair[1] == max_depth),
        name="coverage>=0.6 OR drilled to max_depth",
        severity=GATE,
    )
    t.check(md.get("nodes_failed"), lambda n: n == 0, name="nodes_failed==0", severity=GATE)
    t.check(md.get("used_searches"), lambda n: n is not None and n <= 50, name="used_searches<=cap", severity=GATE)
    t.check(md.get("used_fetches"), lambda n: n is not None and n <= 60, name="used_fetches<=cap", severity=GATE)
    # Source diversity + grounding.
    t.check(len(md.get("research_sources") or []), lambda n: n >= 3, name="sources>=3", severity=GATE)
    t.citation(min_citations=5)
    # Report depth (shallow-report guard).
    t.check(len(t.reply), lambda n: n >= 8000, name="report>=8000 chars", severity=GATE)
    # Claim grounding (RAGAS faithfulness -- needs [eval-ragas] extra + a judge LLM; SOFT: dep-gated).
    if t.require_live(extra="ragas"):
        await t.judge("deep_research_faithfulness", min_score=0.7, name="faithfulness>=0.7")
