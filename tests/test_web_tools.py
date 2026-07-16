"""Tests for koboi.tools.builtin.web module (web_fetch tool)."""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, AsyncMock, patch

import httpx
import pytest

from koboi.tools.builtin.web import (
    _check_url_ssrf,
    _extract_html_content,
    _resolve_and_check,
    web_fetch,
)


# ── Helpers ──


def _mock_response(
    status_code: int = 200,
    text: str = "OK",
    content: bytes | None = None,
    headers: dict | None = None,
) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        content=content or text.encode("utf-8"),
        headers=headers or {},
        request=httpx.Request("GET", "https://example.com"),
    )


def _mock_dns(*ips: str):
    """Return a patcher that makes socket.getaddrinfo resolve to the given IPs."""
    results = []
    for ip in ips:
        results.append((socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0)))
    return patch("koboi.tools.builtin.web.socket.getaddrinfo", return_value=results)


# ── TestStripHtml ──


class TestExtractHtmlContent:
    def test_removes_scripts_and_styles(self):
        html = "<html><script>alert('x')</script><style>body{}</style><p>Hello</p></html>"
        result = _extract_html_content(html)
        assert "Hello" in result
        assert "alert" not in result

    def test_removes_tags(self):
        html = "<div><b>Bold</b> and <i>italic</i></div>"
        assert "Bold and italic" in _extract_html_content(html)

    def test_collapses_whitespace(self):
        html = "<p>A</p>\n\n\n\n<p>B</p>"
        assert "\n\n\n" not in _extract_html_content(html)

    def test_empty_input(self):
        assert _extract_html_content("") == ""

    def test_extracts_title(self):
        html = "<html><head><title>My Page</title></head><body><p>Content</p></body></html>"
        result = _extract_html_content(html)
        assert "My Page" in result
        assert "Content" in result

    def test_extracts_meta_description(self):
        html = '<html><head><meta name="description" content="A great page"></head><body></body></html>'
        result = _extract_html_content(html)
        assert "A great page" in result

    def test_extracts_noscript_fallback(self):
        html = "<html><body><noscript><p>Real content here</p></noscript></body></html>"
        result = _extract_html_content(html)
        assert "Real content here" in result

    def test_spa_shell_returns_extracted_parts(self):
        html = '<html><head><title>SPA App</title></head><body><div id="root"></div></body></html>'
        result = _extract_html_content(html)
        assert "SPA App" in result

    def test_includes_title_and_body(self):
        html = "<html><head><title>Hello</title></head><body><p>Hello World</p></body></html>"
        result = _extract_html_content(html)
        assert "Hello" in result
        assert "Hello World" in result


# ── TestResolveAndCheck ──


class TestResolveAndCheck:
    def test_returns_ips_for_public_hostname(self):
        with _mock_dns("93.184.216.34"):
            result = _resolve_and_check("example.com")
        assert "93.184.216.34" in result

    def test_raises_for_private_ip(self):
        with _mock_dns("10.0.0.1"):
            with pytest.raises(ValueError, match="internal IP"):
                _resolve_and_check("internal.corp")

    def test_raises_for_dns_failure(self):
        with patch("koboi.tools.builtin.web.socket.getaddrinfo", side_effect=socket.gaierror):
            with pytest.raises(socket.gaierror):
                _resolve_and_check("nonexistent.invalid")

    def test_checks_all_ips(self):
        results = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
        ]
        with patch("koboi.tools.builtin.web.socket.getaddrinfo", return_value=results):
            with pytest.raises(ValueError, match="internal IP"):
                _resolve_and_check("mixed.example.com")

    def test_raises_for_ipv6_unspecified(self):
        # Issue #54: "::" (IPv6 unspecified, the IPv6 equivalent of 0.0.0.0/8)
        # was missing from the enumerated CIDR list and bypassed the SSRF guard.
        results = [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::", 0, 0, 0))]
        with patch("koboi.tools.builtin.web.socket.getaddrinfo", return_value=results):
            with pytest.raises(ValueError, match="internal IP"):
                _resolve_and_check("ipv6-unspecified.example")

    def test_check_url_ssrf_blocks_ipv6_unspecified_literal(self):
        # Issue #54: direct guard on an http://[::] URL (IP literal, no DNS hop).
        with pytest.raises(ValueError, match="internal IP"):
            _check_url_ssrf("http://[::]:1/x")


# ── TestCheckUrlSsrf ──


class TestCheckUrlSsrf:
    def test_rejects_no_hostname(self):
        with pytest.raises(ValueError, match="hostname not found"):
            _check_url_ssrf("http:///path")

    def test_passes_for_public_url(self):
        with _mock_dns("93.184.216.34"):
            _check_url_ssrf("https://example.com")


# ── TestWebFetchSuccess ──


class TestWebFetchSuccess:
    async def test_fetch_plain_text(self):
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=_mock_response(200, "Hello, World!"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        with _mock_dns("93.184.216.34"):
            with patch("koboi.tools.builtin.web.httpx.AsyncClient", return_value=mock_client):
                result = await web_fetch("https://example.com")
        assert "Hello, World!" in result
        assert "Error" not in result

    async def test_fetch_html_stripped(self):
        html = "<html><head><title>Test</title></head><body><p>Hello</p></body></html>"
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=_mock_response(200, html, content=html.encode()))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        with _mock_dns("93.184.216.34"):
            with patch("koboi.tools.builtin.web.httpx.AsyncClient", return_value=mock_client):
                result = await web_fetch("https://example.com")
        assert "<html>" not in result
        assert "Hello" in result

    async def test_invalid_scheme(self):
        result = await web_fetch("ftp://files.example.com")
        assert "Error" in result
        assert "http://" in result

    async def test_unresolvable_hostname(self):
        with patch("koboi.tools.builtin.web.socket.getaddrinfo", side_effect=socket.gaierror):
            result = await web_fetch("https://thisdoesnotexist.invalid")
        assert "Error" in result
        assert "resolve" in result


# ── TestWebFetchSSRFProtection ──


class TestWebFetchSSRFProtection:
    async def test_blocks_localhost(self):
        with _mock_dns("127.0.0.1"):
            result = await web_fetch("http://localhost/secret")
        assert "Error" in result
        assert "internal IP" in result

    async def test_blocks_10_network(self):
        with _mock_dns("10.0.0.5"):
            result = await web_fetch("http://internal.corp/data")
        assert "Error" in result
        assert "internal IP" in result

    async def test_blocks_172_16_network(self):
        with _mock_dns("172.16.0.1"):
            result = await web_fetch("http://private.local/data")
        assert "Error" in result
        assert "internal IP" in result

    async def test_blocks_192_168_network(self):
        with _mock_dns("192.168.1.1"):
            result = await web_fetch("http://router.local/admin")
        assert "Error" in result
        assert "internal IP" in result

    async def test_blocks_link_local(self):
        with _mock_dns("169.254.169.254"):
            result = await web_fetch("http://metadata.internal/latest")
        assert "Error" in result
        assert "internal IP" in result

    async def test_blocks_ipv6_loopback(self):
        results = [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 0, 0, 0))]
        with patch("koboi.tools.builtin.web.socket.getaddrinfo", return_value=results):
            result = await web_fetch("http://ipv6-loopback/test")
        assert "Error" in result
        assert "internal IP" in result

    async def test_blocks_ipv6_unspecified(self):
        # Issue #54: "::" (IPv6 unspecified) bypassed the SSRF guard.
        results = [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::", 0, 0, 0))]
        with patch("koboi.tools.builtin.web.socket.getaddrinfo", return_value=results):
            result = await web_fetch("http://ipv6-unspecified/x")
        assert "Error" in result
        assert "internal IP" in result

    async def test_blocks_zero_network(self):
        # H2: 0.0.0.0/8 ("this host"/unset) was missing from the allow-list.
        with _mock_dns("0.0.0.5"):
            result = await web_fetch("http://zero.example/x")
        assert "internal IP" in result

    async def test_blocks_cgnat_100_64(self):
        # H2: 100.64.0.0/10 (CGNAT) was missing from the allow-list.
        with _mock_dns("100.64.0.1"):
            result = await web_fetch("http://cgnat.example/x")
        assert "internal IP" in result

    async def test_blocks_ipv4_mapped_ipv6_loopback(self):
        # H2: ::ffff:0:0/96 (IPv4-mapped IPv6) was missing from the allow-list.
        results = [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::ffff:127.0.0.1", 0, 0, 0))]
        with patch("koboi.tools.builtin.web.socket.getaddrinfo", return_value=results):
            result = await web_fetch("http://mapped-v6.example/x")
        assert "internal IP" in result

    async def test_blocks_redirect_to_metadata(self):
        # H2: a public URL that 302-redirects to the metadata IP is blocked --
        # _check_url_ssrf is re-run on the redirect target (hop 1).
        redirect = httpx.Response(
            302,
            headers={"location": "http://169.254.169.254/latest/meta-data/"},
            request=httpx.Request("GET", "http://attacker.example/redir"),
        )

        async def fake_get(_self, _url, **_kw):
            return redirect

        with (
            patch("koboi.tools.builtin.web._check_url_ssrf", side_effect=[None, ValueError("internal IP")]),
            patch("httpx.AsyncClient.get", new=fake_get),
        ):
            result = await web_fetch("http://attacker.example/redir")
        assert "internal IP" in result

    async def test_follows_safe_redirect(self):
        # H2: a public -> public redirect is still followed and the body returned.
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
            patch("koboi.tools.builtin.web._check_url_ssrf", return_value=None),
            patch("httpx.AsyncClient.get", new=fake_get),
        ):
            result = await web_fetch("http://safe.example/r")
        assert "final content" in result
