"""Tests for koboi/web fetch providers (ReadabilityFetchProvider + FirecrawlFetchProvider)
and the web_fetch tool's delegation path. All HTTP is mocked (no network)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from koboi.tools.builtin.web import web_fetch
from koboi.web import build_fetch_provider
from koboi.web.providers.firecrawl import FirecrawlFetchProvider
from koboi.web.providers.readability import ReadabilityFetchProvider, _TRAFILATURA_AVAILABLE

_SSRF = "koboi.tools.builtin.web._check_url_ssrf"


@pytest.fixture(autouse=True)
def _bypass_ssrf():
    # Default: bypass the real SSRF guard so successful-fetch tests don't hit DNS. The
    # SSRF-rejection tests override this by patching _check_url_ssrf to raise.
    with patch(_SSRF, return_value=None):
        yield


def _response(*, status: int = 200, content: bytes = b"", json_payload: dict | None = None) -> httpx.Response:
    if json_payload is not None:
        return httpx.Response(status, json=json_payload, request=httpx.Request("POST", "https://example.com"))
    return httpx.Response(status, content=content, request=httpx.Request("GET", "https://example.com"))


def _mock_async_client(response: httpx.Response) -> MagicMock:
    """An httpx.AsyncClient double: async CM whose .get/.post return ``response``."""
    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    client.post = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


class TestBuildFetchProvider:
    def test_default_is_readability_httpx(self):
        assert isinstance(build_fetch_provider(None), ReadabilityFetchProvider)

    def test_build_firecrawl_from_config(self):
        provider = build_fetch_provider({"fetch": {"provider": "firecrawl", "firecrawl": {"api_key": "k"}}})
        assert isinstance(provider, FirecrawlFetchProvider)
        assert provider._api_key == "k"

    def test_unknown_falls_back_to_httpx(self, caplog):
        provider = build_fetch_provider({"fetch": {"provider": "nonexistent_xyz"}})
        assert isinstance(provider, ReadabilityFetchProvider)
        assert any("Unknown fetch provider" in r.message for r in caplog.records)


class TestReadabilityFetchProvider:
    async def test_invalid_scheme(self):
        result = await ReadabilityFetchProvider().fetch("ftp://example.com")
        assert result.content == ""
        assert result.metadata.get("error") == "invalid scheme"

    async def test_fetches_and_extracts_static_html(self):
        html = b"<html><head><title>Test</title></head><body><p>Hello world</p></body></html>"
        with patch(
            "koboi.web.providers.readability.httpx.AsyncClient",
            return_value=_mock_async_client(_response(status=200, content=html)),
        ):
            result = await ReadabilityFetchProvider().fetch("https://example.com")
        assert result.status == 200
        assert "Hello world" in result.content

    async def test_ssrf_blocks_metadata_ip(self):
        with patch(_SSRF, side_effect=ValueError("internal IP")):
            result = await ReadabilityFetchProvider().fetch("http://169.254.169.254/latest")
        assert result.metadata.get("error") == "internal IP"
        assert result.content == ""

    async def test_truncates_long_content(self):
        html = b"<html><body><p>" + (b"A" * 50000) + b"</p></body></html>"
        with patch(
            "koboi.web.providers.readability.httpx.AsyncClient",
            return_value=_mock_async_client(_response(status=200, content=html)),
        ):
            result = await ReadabilityFetchProvider(max_chars=100).fetch("https://example.com")
        assert result.truncated is True
        assert len(result.content) <= 100

    @pytest.mark.skipif(not _TRAFILATURA_AVAILABLE, reason="trafilatura ([web] extra) not installed")
    async def test_trafilatura_path_produces_markdown(self):
        # Only runs when the [web] extra is installed locally; CI exercises the fallback.
        html = b"<html><body><article><p>Real article content here.</p></article></body></html>"
        with patch(
            "koboi.web.providers.readability.httpx.AsyncClient",
            return_value=_mock_async_client(_response(status=200, content=html)),
        ):
            result = await ReadabilityFetchProvider().fetch("https://example.com")
        assert "Real article content here" in result.content
        assert result.content_type == "markdown"


class TestFirecrawlFetchProvider:
    async def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        provider = FirecrawlFetchProvider(api_key="")
        with pytest.raises(ValueError, match="api_key"):
            await provider.fetch("https://example.com")

    async def test_parses_scrape(self):
        payload = {"data": {"markdown": "# Title\n\nBody text.", "metadata": {"title": "Title"}}}
        with patch(
            "koboi.web.providers.firecrawl.httpx.AsyncClient",
            return_value=_mock_async_client(_response(status=200, json_payload=payload)),
        ):
            result = await FirecrawlFetchProvider(api_key="k").fetch("https://example.com")
        assert result.content == "# Title\n\nBody text."
        assert result.content_type == "markdown"
        assert result.title == "Title"

    async def test_ssrf_blocks_metadata_ip(self):
        with patch(_SSRF, side_effect=ValueError("internal IP")):
            result = await FirecrawlFetchProvider(api_key="k").fetch("http://169.254.169.254/latest")
        assert result.metadata.get("error") == "internal IP"
        assert result.content == ""

    async def test_http_error_returned(self):
        with patch(
            "koboi.web.providers.firecrawl.httpx.AsyncClient",
            return_value=_mock_async_client(_response(status=402)),
        ):
            result = await FirecrawlFetchProvider(api_key="k").fetch("https://example.com")
        assert result.status == 402
        assert result.metadata.get("error", "").startswith("HTTP")


class TestWebFetchDelegation:
    async def test_delegates_to_injected_provider(self):
        # When fetch_provider is wired, web_fetch delegates (returns result.content).
        provider = ReadabilityFetchProvider()
        html = b"<html><body><p>Delegated content</p></body></html>"
        with patch(
            "koboi.web.providers.readability.httpx.AsyncClient",
            return_value=_mock_async_client(_response(status=200, content=html)),
        ):
            out = await web_fetch("https://example.com", _deps={"fetch_provider": provider})
        assert "Delegated content" in out

    async def test_provider_error_becomes_error_string(self):
        class _Boom:
            async def fetch(self, *_a, **_kw):
                raise RuntimeError("boom")

        out = await web_fetch("https://example.com", _deps={"fetch_provider": _Boom()})
        assert "Error: fetch failed" in out

    async def test_no_deps_falls_through_to_inline_loop(self):
        # Direct call with no _deps -> inline SSRF loop -> invalid scheme short-circuits here.
        out = await web_fetch("ftp://example.com")
        assert "Error" in out


class TestHigh4RedirectSSRF:
    """HIGH-4: per-hop SSRF guard on redirects -- a 302 to 169.254.169.254 MUST be blocked."""

    async def test_redirect_to_metadata_ip_blocked(self):
        """A public URL that 302-redirects to the cloud metadata IP is blocked on hop 1."""
        redirect = httpx.Response(
            302,
            headers={"location": "http://169.254.169.254/latest/meta-data/"},
            request=httpx.Request("GET", "http://attacker.example/redir"),
        )
        ok = httpx.Response(
            200,
            content=b"final content",
            request=httpx.Request("GET", "http://safe-target.example/ok"),
        )
        state = {"n": 0}

        async def fake_get(_self, _url, **_kw):
            r = redirect if state["n"] == 0 else ok
            state["n"] += 1
            return r

        # DO NOT bypass the SSRF guard (remove the autouse bypass) -- test the real guard.
        with (
            patch(_SSRF, side_effect=None),  # initial URL passes
            patch("httpx.AsyncClient.get", new=fake_get),
        ):
            # Re-patch: first call (initial URL) passes, second call (redirect target) blocks.
            with patch(_SSRF, side_effect=[None, ValueError("internal IP")]):
                result = await ReadabilityFetchProvider().fetch("http://attacker.example/redir")
        assert result.metadata.get("error") == "internal IP"
        assert result.content == ""

    async def test_safe_redirect_followed(self):
        """A public -> public 302 redirect is followed and the body is returned."""
        redirect = httpx.Response(
            302,
            headers={"location": "http://safe-target.example/ok"},
            request=httpx.Request("GET", "http://safe.example/r"),
        )
        ok = httpx.Response(
            200, content=b"final content", request=httpx.Request("GET", "http://safe-target.example/ok")
        )
        state = {"n": 0}

        async def fake_get(_self, _url, **_kw):
            r = redirect if state["n"] == 0 else ok
            state["n"] += 1
            return r

        with (
            patch(_SSRF, return_value=None),  # all hops pass SSRF
            patch("httpx.AsyncClient.get", new=fake_get),
        ):
            result = await ReadabilityFetchProvider().fetch("http://safe.example/r")
        assert "final content" in result.content


class TestMedium9FirecrawlHttpError:
    """M9: Firecrawl scrape returns a structured error on HTTP 4xx/5xx."""

    async def test_firecrawl_402_returns_error(self):
        """FirecrawlFetchProvider returns FetchResult(metadata={'error': 'HTTP 402'}), not a crash."""
        error_resp = httpx.Response(402, request=httpx.Request("POST", "https://api.firecrawl.dev/v1/scrape"))
        with patch(
            "koboi.web.providers.firecrawl.httpx.AsyncClient",
            return_value=_mock_async_client(error_resp),
        ):
            result = await FirecrawlFetchProvider(api_key="k").fetch("https://example.com")
        assert result.status == 402
        assert result.metadata.get("error", "").startswith("HTTP")
