"""Live integration test for deep_research (W6 C2).

Env-gated -- SKIPPED without ``OPENAI_API_KEY`` (CI has no key; web is mock -> $0 search cost).
Builds the real deep_research agent from ``configs/deep_research_demo.yaml`` and asserts a cited
report + ``research_sources`` metadata (propagated by C1a). Complementary to the ``t.*`` eval
(``evals/deep_research_citations.eval.py``) -- this is the pytest-level live path.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[2]
_DEEP_RESEARCH_CONFIG = _REPO_ROOT / "configs" / "deep_research_demo.yaml"

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="needs OPENAI_API_KEY (live LLM; web.search.provider defaults to mock -> $0 search)",
)


async def test_deep_research_produces_cited_report():
    from koboi.facade import KoboiAgent

    agent = KoboiAgent.from_config(str(_DEEP_RESEARCH_CONFIG))
    result = await agent.run("Research the Python programming language release cycle.")

    # Cited report: a [n] marker or a Sources footer is present.
    content = result.content or ""
    assert "[1]" in content or "## Sources" in content, f"no citations in report: {content[:200]!r}"

    # C1a: research_sources metadata propagated into RunResult.metadata.
    sources = (result.metadata or {}).get("research_sources")
    assert sources, f"expected research_sources metadata, got keys={list((result.metadata or {}).keys())}"

    # depth is bounded (at least 1 round ran).
    depth = (result.metadata or {}).get("depth", 0)
    assert depth >= 1
