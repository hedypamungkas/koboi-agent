"""Tests for koboi/web/providers/counting.py -- budget-metering wrappers (W4 A0/A4)."""

from __future__ import annotations

from koboi.orchestration.research import ResearchBudget
from koboi.web.base import BaseFetchProvider, BaseSearchProvider
from koboi.web.providers.counting import CountingFetchProvider, CountingSearchProvider
from koboi.web.types import FetchResult, SearchResult


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
