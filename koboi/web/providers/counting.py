"""koboi/web/providers/counting.py -- budget-metering wrappers (W4).

Wrap a real search/fetch provider to count each call against a shared ``ResearchBudget``
(deep_research hard caps). When the budget is exhausted, return an empty/error result
gracefully -- the research loop then stops at the next round -- instead of crashing the node.
Not registered (it's a wrapper, not a selectable provider): ``_run_deep_research`` constructs
it around the configured provider + the run's ``ctx.budget``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from koboi.web.base import BaseFetchProvider, BaseSearchProvider
from koboi.web.types import FetchResult, SearchResult

if TYPE_CHECKING:
    from koboi.orchestration.research import ResearchBudget


class CountingSearchProvider(BaseSearchProvider):
    """Delegating search provider that charges one search per call against the budget."""

    def __init__(self, inner: BaseSearchProvider, budget: ResearchBudget) -> None:
        self._inner = inner
        self._budget = budget

    async def search(self, query: str, *, max_results: int = 10) -> list[SearchResult]:
        if not self._budget.remaining():
            return []
        self._budget.record_searches(1)
        return await self._inner.search(query, max_results=max_results)


class CountingFetchProvider(BaseFetchProvider):
    """Delegating fetch provider that charges one fetch per call against the budget."""

    def __init__(self, inner: BaseFetchProvider, budget: ResearchBudget) -> None:
        self._inner = inner
        self._budget = budget

    async def fetch(self, url: str, *, render: str = "auto", timeout: int = 15) -> FetchResult:
        if not self._budget.remaining():
            return FetchResult(url=url, content="", status=0, metadata={"error": "research budget exhausted"})
        self._budget.record_fetches(1)
        return await self._inner.fetch(url, render=render, timeout=timeout)
