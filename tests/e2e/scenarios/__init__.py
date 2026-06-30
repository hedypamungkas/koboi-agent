"""Scenario catalog for the E2E suite.

Each module exposes a ``SCENARIOS: list[Scenario]``. ``all_scenarios()``
collects them for the parametrized runner. Categories:
  multi_turn   — multi-LLM-call conversations (memory across turns)
  multi_tool   — tool orchestration chains
  rag          — retrieval-augmented factual Q&A
  skills       — skill-activated domain agents
  jobs         — autonomous background jobs
  stress       — concurrency / long-context / burst
"""

from __future__ import annotations

from tests.e2e.framework.scenario import Scenario
from tests.e2e.scenarios import (
    jobs,
    multi_tool,
    multi_turn,
    rag_heavy,
    skills,
    stress,
)

__all__ = ["all_scenarios", "by_category"]


def all_scenarios() -> list[Scenario]:
    return (
        multi_turn.SCENARIOS
        + multi_tool.SCENARIOS
        + rag_heavy.SCENARIOS
        + skills.SCENARIOS
        + jobs.SCENARIOS
        + stress.SCENARIOS
    )


def by_category(category: str) -> list[Scenario]:
    return [s for s in all_scenarios() if s.category == category]
