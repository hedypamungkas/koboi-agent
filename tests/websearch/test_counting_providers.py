"""Tests for koboi/websearch/providers/counting.py -- budget-metering wrappers (W4 A0/A4)."""

from __future__ import annotations

import pytest

from koboi.orchestration.research import ResearchBudget
from koboi.websearch.base import BaseFetchProvider, BaseSearchProvider
from koboi.websearch.providers.counting import CountingFetchProvider, CountingSearchProvider
from koboi.websearch.types import FetchResult, SearchResult


class _InnerSearch(BaseSearchProvider):
    def __init__(self) -> None:
        self.calls = 0

    async def search(self, query: str, *, max_results: int = 10) -> list[SearchResult]:
        self.calls += 1
        return [SearchResult(title="t", url="https://x.example", snippet="s")]


class _InnerFetch(BaseFetchProvider):
    def __init__(self) -> None:
        self.calls = 0

    async def fetch(self, url: str, *, render: str = "auto", timeout: int = 15) -> FetchResult:
        self.calls += 1
        return FetchResult(url=url, content="body", content_type="text")


class TestCountingSearchProvider:
    async def test_delegates_and_records(self):
        inner = _InnerSearch()
        budget = ResearchBudget(max_searches=5)
        provider = CountingSearchProvider(inner, budget)
        await provider.search("q")
        assert inner.calls == 1
        assert budget.used_searches == 1

    async def test_cap_returns_empty_without_delegating(self):
        inner = _InnerSearch()
        budget = ResearchBudget(max_searches=1)
        budget.record_searches(1)  # pre-exhaust
        provider = CountingSearchProvider(inner, budget)
        results = await provider.search("q")
        assert results == []
        assert inner.calls == 0  # did not delegate past the cap


class TestCountingFetchProvider:
    async def test_delegates_and_records(self):
        inner = _InnerFetch()
        budget = ResearchBudget(max_fetches=5)
        provider = CountingFetchProvider(inner, budget)
        await provider.fetch("https://x.example")
        assert inner.calls == 1
        assert budget.used_fetches == 1

    async def test_cap_returns_error_without_delegating(self):
        inner = _InnerFetch()
        budget = ResearchBudget(max_fetches=1)
        budget.record_fetches(1)  # pre-exhaust
        provider = CountingFetchProvider(inner, budget)
        result = await provider.fetch("https://x.example")
        assert result.metadata.get("error") == "research budget exhausted"
        assert inner.calls == 0


class TestMedium10BudgetChargedBeforeFailure:
    """M10: when the inner provider raises, the budget is consumed (charged) but NOT refunded.

    This LOCKS the current behavior so a future refund implementation doesn't silently change
    semantics. A flaky provider eating budget is a known tradeoff documented in the code.
    """

    async def test_search_budget_consumed_on_inner_failure(self):
        class _BoomSearch(BaseSearchProvider):
            async def search(self, query: str, *, max_results: int = 10):
                raise RuntimeError("provider down")

        budget = ResearchBudget(max_searches=5)
        provider = CountingSearchProvider(_BoomSearch(), budget)
        with pytest.raises(RuntimeError, match="provider down"):
            await provider.search("q")
        assert budget.used_searches == 1  # charged before delegation, NOT refunded on failure

    async def test_fetch_budget_consumed_on_inner_failure(self):
        class _BoomFetch(BaseFetchProvider):
            async def fetch(self, url: str, *, render: str = "auto", timeout: int = 15):
                raise RuntimeError("provider down")

        budget = ResearchBudget(max_fetches=5)
        provider = CountingFetchProvider(_BoomFetch(), budget)
        with pytest.raises(RuntimeError, match="provider down"):
            await provider.fetch("https://example.com")
        assert budget.used_fetches == 1  # charged before delegation, NOT refunded
