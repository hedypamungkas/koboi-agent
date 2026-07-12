"""Tests for koboi/web search providers + the web_search tool wrapper.

All provider HTTP is mocked (no network). Mirrors the httpx-mocking style of
tests/test_web_tools.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from koboi.tools.builtin.web import web_search
from koboi.web.providers.brave import BraveSearchProvider
from koboi.web.providers.ddg import DDGSearchProvider
from koboi.web.providers.firecrawl import FirecrawlSearchProvider
from koboi.web.providers.mock import MockSearchProvider


def _response(*, status: int = 200, json_payload: dict | None = None, content: bytes | None = None) -> httpx.Response:
    if json_payload is not None:
        return httpx.Response(status, json=json_payload, request=httpx.Request("GET", "https://example.com"))
    return httpx.Response(status, content=content or b"", request=httpx.Request("GET", "https://example.com"))


def _mock_async_client(response: httpx.Response) -> MagicMock:
    """An httpx.AsyncClient double: async CM whose .get/.post return ``response``."""
    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    client.post = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


class TestMockProvider:
    async def test_search_python(self):
        results = await MockSearchProvider().search("python")
        assert any("Python Documentation" in r.title for r in results)
        assert any("docs.python.org" in r.url for r in results)

    async def test_search_no_results(self):
        assert await MockSearchProvider().search("zzznonexistent") == []

    async def test_max_results_cap(self):
        results = await MockSearchProvider().search("python", max_results=1)
        assert len(results) <= 1


class TestBraveProvider:
    async def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        provider = BraveSearchProvider(api_key="")
        with pytest.raises(ValueError, match="api_key"):
            await provider.search("q")

    async def test_parses_results(self):
        payload = {
            "web": {
                "results": [
                    {"title": "Result One", "url": "https://one.example", "description": "desc one"},
                    {"title": "Result Two", "url": "https://two.example", "description": "desc two"},
                ]
            }
        }
        with patch(
            "koboi.web.providers.brave.httpx.AsyncClient",
            return_value=_mock_async_client(_response(json_payload=payload)),
        ):
            results = await BraveSearchProvider(api_key="k").search("query")
        assert len(results) == 2
        assert results[0].title == "Result One"
        assert results[0].url == "https://one.example"
        assert results[0].snippet == "desc one"

    async def test_skips_entries_without_url(self):
        payload = {"web": {"results": [{"title": "No URL"}, {"title": "OK", "url": "https://ok.example"}]}}
        with patch(
            "koboi.web.providers.brave.httpx.AsyncClient",
            return_value=_mock_async_client(_response(json_payload=payload)),
        ):
            results = await BraveSearchProvider(api_key="k").search("q")
        assert len(results) == 1
        assert results[0].url == "https://ok.example"


class TestFirecrawlProvider:
    async def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        provider = FirecrawlSearchProvider(api_key="")
        with pytest.raises(ValueError, match="api_key"):
            await provider.search("q")

    async def test_parses_results(self):
        payload = {"data": [{"title": "FC One", "url": "https://fc.example", "description": "fc desc"}]}
        with patch(
            "koboi.web.providers.firecrawl.httpx.AsyncClient",
            return_value=_mock_async_client(_response(json_payload=payload)),
        ):
            results = await FirecrawlSearchProvider(api_key="k").search("q")
        assert len(results) == 1
        assert results[0].title == "FC One"
        assert results[0].snippet == "fc desc"


class TestDDGProvider:
    async def test_parses_html(self):
        html = '<a class="result__a" href="https://ddg.example">DDG Hit</a><a class="result__snippet">a desc</a>'
        with patch(
            "koboi.web.providers.ddg.httpx.AsyncClient",
            return_value=_mock_async_client(_response(content=html.encode())),
        ):
            results = await DDGSearchProvider().search("q")
        assert len(results) == 1
        assert results[0].title == "DDG Hit"
        assert results[0].url == "https://ddg.example"


class TestWebSearchWrapper:
    async def test_default_mock_provider(self):
        # No _deps -> default provider (mock) -> "react" is in the offline index.
        result = await web_search("react")
        assert "React" in result

    async def test_no_results_message(self):
        result = await web_search("zzznonexistent")
        assert "No results" in result

    async def test_with_injected_provider(self):
        result = await web_search("python", _deps={"search_provider": MockSearchProvider()})
        assert "Python Documentation" in result

    async def test_provider_error_becomes_error_string(self):
        class _Boom:
            async def search(self, *_a, **_kw):
                raise RuntimeError("boom")

        result = await web_search("q", _deps={"search_provider": _Boom()})
        assert "Error: search failed" in result

    async def test_tool_config_max_results(self):
        # _tool_config.max_results is forwarded to the provider (mock ignores the cap
        # for a single-topic query, so just assert it runs without error via injection).
        result = await web_search(
            "python", _deps={"search_provider": MockSearchProvider()}, _tool_config={"max_results": 2}
        )
        assert "Python Documentation" in result


class TestMedium9BraveHttpError:
    """M9: Brave raise_for_status() propagates; web_search wraps it as an error string."""

    async def test_brave_401_raises_http_status_error(self):
        error_resp = httpx.Response(401, request=httpx.Request("GET", "https://api.search.brave.com"))
        with patch("koboi.web.providers.brave.httpx.AsyncClient", return_value=_mock_async_client(error_resp)):
            with pytest.raises(httpx.HTTPStatusError):
                await BraveSearchProvider(api_key="bad-key").search("q")

    async def test_web_search_wraps_http_error(self):
        """The web_search tool catches the HTTPStatusError and returns an error string."""
        from koboi.tools.builtin.web import web_search

        error_resp = httpx.Response(429, request=httpx.Request("GET", "https://api.search.brave.com"))

        class _BoomProvider:
            async def search(self, *_a, **_kw):
                raise httpx.HTTPStatusError("rate limited", request=error_resp.request, response=error_resp)

        result = await web_search("q", _deps={"search_provider": _BoomProvider()})
        assert "Error: search failed" in result
