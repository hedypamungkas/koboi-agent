"""Tests for koboi/tools/builtin/web.py — web_search and web_fetch expanded coverage."""

from __future__ import annotations

import pytest

from koboi.tools.builtin.web import (
    _format_results,
    _search_mock,
    _extract_html_content,
    _check_url_ssrf,
    _DDGResultParser,
    web_search,
    web_fetch,
)


class TestSearchMock:
    def test_search_python(self):
        result = _search_mock("python")
        assert "Python Documentation" in result
        assert "docs.python.org" in result

    def test_search_no_results(self):
        result = _search_mock("xyznonexistent12345")
        assert "No results found" in result

    def test_search_multi_token(self):
        result = _search_mock("python asyncio")
        assert "Async IO" in result

    def test_format_results(self):
        results = [
            {"title": "Test", "url": "https://example.com", "snippet": "A test result"},
        ]
        out = _format_results("test", results)
        assert "Test" in out
        assert "https://example.com" in out

    def test_format_results_dedup(self):
        results = [
            {"title": "A", "url": "https://a.com", "snippet": "first"},
            {"title": "A", "url": "https://a.com", "snippet": "dup"},
        ]
        out = _format_results("q", results)
        assert out.count("https://a.com") == 1


class TestWebSearch:
    @pytest.mark.asyncio
    async def test_mock_provider(self):
        result = await web_search("react")
        assert "React" in result

    @pytest.mark.asyncio
    async def test_mock_provider_no_results(self):
        result = await web_search("xyznonexistent")
        assert "No results" in result


class TestDDGResultParser:
    def test_parse_results(self):
        html = """
        <a class="result__a" href="https://example.com">Example</a>
        <a class="result__snippet">A description</a>
        """
        parser = _DDGResultParser()
        parser.feed(html)
        assert len(parser.results) == 1
        assert parser.results[0]["title"] == "Example"
        assert parser.results[0]["url"] == "https://example.com"

    def test_parse_empty(self):
        parser = _DDGResultParser()
        parser.feed("<html><body></body></html>")
        assert len(parser.results) == 0


class TestExtractHtmlContent:
    def test_basic_html(self):
        html = "<html><head><title>Test</title></head><body><p>Hello world</p></body></html>"
        result = _extract_html_content(html)
        assert "Test" in result
        assert "Hello world" in result

    def test_html_with_scripts(self):
        html = "<html><body><script>alert('xss')</script><p>Content</p></body></html>"
        result = _extract_html_content(html)
        assert "alert" not in result
        assert "Content" in result

    def test_html_with_meta(self):
        html = '<html><head><meta name="description" content="A page"></head><body>Body</body></html>'
        result = _extract_html_content(html)
        assert "A page" in result

    def test_noscript_fallback(self):
        html = "<html><body><noscript>Fallback content</noscript></body></html>"
        result = _extract_html_content(html)
        assert "Fallback content" in result

    def test_empty_body_with_title(self):
        html = "<html><head><title>Only Title</title></head><body></body></html>"
        result = _extract_html_content(html)
        assert "Only Title" in result


class TestCheckUrlSsrf:
    def test_private_ip_localhost(self):
        with pytest.raises(ValueError, match="internal"):
            _check_url_ssrf("http://127.0.0.1/admin")

    def test_private_ip_10_network(self):
        with pytest.raises(ValueError, match="internal"):
            _check_url_ssrf("http://10.0.0.1/secret")

    def test_invalid_url_no_hostname(self):
        with pytest.raises(ValueError, match="hostname"):
            _check_url_ssrf("not-a-url")


class TestWebFetch:
    @pytest.mark.asyncio
    async def test_invalid_scheme(self):
        result = await web_fetch("ftp://example.com")
        assert "Error" in result
        assert "http" in result

    @pytest.mark.asyncio
    async def test_unreachable_host(self):
        result = await web_fetch("http://this-domain-does-not-exist-12345.invalid")
        # Either DNS failure or connection error
        assert "Error" in result
