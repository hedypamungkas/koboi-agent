"""koboi/tools/builtin/web -- Web search and URL fetching.

``web_search`` delegates to a registry-resolved search provider (``koboi.websearch``); the
active provider is injected via the tool registry's dep store (``search_provider``) and
defaults to ``mock`` for offline safety / back-compat. ``web_fetch`` still uses the
in-process httpx + SSRF guard here (Wave 1 refactors it onto a fetch-provider registry).

The mock/ddg helpers are migrated to ``koboi.websearch.providers`` and re-exported below for
back-compat (tests and the ``WEB_SEARCH_PROVIDER`` env path still import them from here).
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import logging
import os
import re
import socket
from typing import cast
from urllib.parse import urlparse

import httpx

from koboi.tools.registry import tool
from koboi.types import RiskLevel
from koboi.websearch.base import BaseSearchProvider
from koboi.websearch.types import SearchResult

_logger = logging.getLogger(__name__)

# ── web_search (delegates to the koboi.websearch search-provider registry) ──

WEB_SEARCH_PROVIDER = os.getenv("WEB_SEARCH_PROVIDER", "mock")

# Back-compat re-exports: the mock/ddg logic now lives in koboi.websearch.providers, but tests
# and the WEB_SEARCH_PROVIDER env path import these names from koboi.tools.builtin.web.
from koboi.websearch.providers.ddg import (  # noqa: E402,F401
    _DDGResultParser,
    _search_duckduckgo,
)
from koboi.websearch.providers.mock import (  # noqa: E402,F401
    SEARCH_INDEX,
    _format_results,
    _search_mock,
)

# Process-wide default provider for direct (no-registry) callers + the WEB_SEARCH_PROVIDER
# env back-compat path. Lazily built and cached per resolved provider name.
_DEFAULT_SEARCH_PROVIDER_CACHE: dict[str, BaseSearchProvider] = {}


def _default_search_provider() -> BaseSearchProvider:
    """Build the default search provider from ``WEB_SEARCH_PROVIDER`` (back-compat).

    Direct callers (e.g. unit tests invoking ``await web_search(q)`` with no registry)
    hit this path. ``WEB_SEARCH_PROVIDER=duckduckgo`` maps to the ``ddg`` provider name.
    """
    name = "ddg" if WEB_SEARCH_PROVIDER == "duckduckgo" else WEB_SEARCH_PROVIDER
    cached = _DEFAULT_SEARCH_PROVIDER_CACHE.get(name)
    if cached is not None:
        return cached
    from koboi.websearch.registry import search_provider_registry

    entry = search_provider_registry.get(name) or search_provider_registry.get("mock")
    if entry is None:  # pragma: no cover - builtins always register mock
        raise RuntimeError("No search providers registered")
    provider = cast(BaseSearchProvider, entry.cls())
    _DEFAULT_SEARCH_PROVIDER_CACHE[name] = provider
    return provider


def _format_search_results(query: str, results: list[SearchResult]) -> str:
    """Format ``SearchResult`` objects as the legacy ``web_search`` output string."""
    lines = [f"Search results for '{query}':"]
    seen: set[str] = set()
    for r in results:
        if r.url in seen:
            continue
        seen.add(r.url)
        lines.append(f"  - {r.title}: {r.url}")
        lines.append(f"    {r.snippet}")
    if len(lines) == 1:
        return f"No results found for '{query}'."
    return "\n".join(lines)


@tool(
    name="web_search",
    group="web",
    description="Search the internet. REQUIRED parameter: 'query' (string, search topic). Does NOT accept other parameters — do not send repo_path, city, expression, key, value, or path.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query, e.g. 'python asyncio tutorial'",
            },
        },
        "required": ["query"],
    },
    deps=["search_provider"],
)
async def web_search(query: str, _deps: dict | None = None, _tool_config: dict | None = None) -> str:
    provider = (_deps or {}).get("search_provider") or _default_search_provider()
    max_results = 10
    if _tool_config:
        try:
            max_results = int(_tool_config.get("max_results", 10))
        except (TypeError, ValueError):
            max_results = 10
    try:
        results = await provider.search(query, max_results=max_results)
    except Exception as e:  # noqa: BLE001 - boundary: any provider failure becomes an error string
        return f"Error: search failed — {e}"
    return _format_search_results(query, results)


# ── web_fetch helpers ──

MAX_RESPONSE_SIZE = 50000
MAX_OUTPUT = 20000
MAX_TIMEOUT = 30
MAX_RETRIES = 2
RETRYABLE_STATUS = {429, 500, 502, 503, 529}

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
)

# Networks that ``ipaddress`` property checks do NOT cover but the SSRF guard must
# still block. CPython flags RFC 6598 CGNAT (100.64.0.0/10) as is_private=False, so
# it needs an explicit net here. The property checks below handle everything else
# (loopback, RFC 1918, link-local, IPv6 unspecified ``::``, ULA fc00::/7, multicast,
# reserved, IPv4-mapped IPv6 like ::ffff:127.0.0.1).
_SSRF_EXTRA_BLOCKED_NETWORKS = [
    ipaddress.ip_network("100.64.0.0/10"),  # RFC 6598 CGNAT / shared address space
]


def _resolve_and_check(hostname: str) -> list[str]:
    """Resolve hostname and reject internal/special-purpose IPs (SSRF defense).

    Rejects any resolved IP whose ``ipaddress`` properties mark it as loopback,
    private, link-local, unspecified (``::`` / ``0.0.0.0``), multicast, or reserved --
    plus the explicit nets in ``_SSRF_EXTRA_BLOCKED_NETWORKS`` (CGNAT). This is broader
    than an enumerated CIDR list and closes the IPv6 unspecified ``::`` bypass (#54):
    ``::`` matches no IPv6 CIDR in the old list but is ``is_unspecified``.
    """
    addrs = socket.getaddrinfo(hostname, None)
    if not addrs:
        raise ValueError(f"DNS resolution returned no addresses for '{hostname}'")

    resolved: list[str] = []
    for _, _, _, _, sa in addrs:
        ip_str = str(sa[0])
        ip = ipaddress.ip_address(ip_str)
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_unspecified
            or ip.is_multicast
            or ip.is_reserved
            or any(ip in net for net in _SSRF_EXTRA_BLOCKED_NETWORKS)
        ):
            raise ValueError("URL points to unauthorized internal IP address")
        resolved.append(ip_str)
    return resolved


def _check_url_ssrf(url: str) -> None:
    """Parse URL, resolve hostname, verify it does not point to a private network."""
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("invalid URL -- hostname not found")
    _resolve_and_check(hostname)


def _extract_html_content(text: str) -> str:
    """Extract readable content from HTML, handling both static and SPA pages."""
    title = ""
    m = re.search(r"<title[^>]*>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
    if m:
        title = m.group(1).strip()

    meta_desc = ""
    m = re.search(r'<meta\s+name=["\']description["\']\s+content=["\'](.*?)["\']', text, re.IGNORECASE)
    if m:
        meta_desc = m.group(1).strip()

    noscript_parts: list[str] = []
    for m in re.finditer(r"<noscript[^>]*>(.*?)</noscript>", text, re.IGNORECASE | re.DOTALL):
        ns = m.group(1).strip()
        if ns:
            noscript_parts.append(ns)

    body = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", text, flags=re.IGNORECASE)
    body = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", body, flags=re.IGNORECASE)
    body = re.sub(r"<[^>]+>", "", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    if not body and (title or noscript_parts):
        parts = [p for p in [title, meta_desc] if p]
        parts.extend(noscript_parts)
        return "\n\n".join(parts)

    header_parts = [p for p in [title, meta_desc] if p]
    if header_parts:
        return "\n\n".join(header_parts) + "\n\n" + body
    return body


async def _aclose_stream(stream_ctx: object) -> None:
    """Best-effort close an ``AsyncClient.stream(...)`` context manager.

    Used by ``web_fetch`` to release the streaming response as soon as the bounded
    read is done or an error/redirect short-circuits the hop (CWE-400 / #56).
    """
    if stream_ctx is None:
        return
    with contextlib.suppress(Exception):
        await stream_ctx.__aexit__(None, None, None)  # type: ignore[attr-defined]


@tool(
    name="web_fetch",
    group="web",
    description="Fetch content from URL and return text. Like curl/wget.",
    risk_level=RiskLevel.MODERATE,
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch, e.g. 'https://example.com'",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds. Default: 15. Max: 30.",
            },
        },
        "required": ["url"],
    },
    deps=["fetch_provider"],
)
async def web_fetch(url: str, timeout: int = 15, _deps: dict | None = None, _tool_config: dict | None = None) -> str:
    # When a fetch provider is wired (agent / deep-research path), delegate to it. Direct
    # callers with no registry fall through to the inline SSRF+redirect+retry loop below,
    # so the existing offline test suite (which patches this module's httpx/socket/_check_url_ssrf)
    # keeps working unchanged.
    provider = (_deps or {}).get("fetch_provider")
    if provider is not None:
        try:
            result = await provider.fetch(url, timeout=timeout)
        except Exception as e:  # noqa: BLE001 - boundary: any provider failure becomes an error string
            return f"Error: fetch failed — {e}"
        if result.metadata.get("error"):
            return f"Error: {result.metadata['error']}"
        return result.content

    if not url.startswith(("http://", "https://")):
        return "Error: URL must start with http:// or https://"

    timeout = max(1, min(timeout, MAX_TIMEOUT))

    # H2: SSRF defense. follow_redirects=False + a manual loop so the
    # DNS/private-range check runs on EVERY hop (initial URL + each Location).
    # Blocks redirect-to-metadata (302 -> 169.254.169.254) and re-resolves DNS
    # per hop (defeats DNS rebinding). Redirects capped at MAX_REDIRECTS.
    #
    # CWE-400 / GHSA-qf8c-xp5r-p869 (#56): the body is read via a STREAMING
    # request with a hard byte bound, so an oversized response cannot be fully
    # buffered before the size limit is applied. The stream is opened per hop and
    # closed the moment we redirect/retry/finish (try/finally below).
    MAX_REDIRECTS = 5
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=False,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        current_url = url
        response = None
        stream_ctx = None
        buf = bytearray()
        oversized = False
        try:
            for _redir in range(MAX_REDIRECTS + 1):
                try:
                    await asyncio.to_thread(_check_url_ssrf, str(current_url))
                except socket.gaierror:
                    return f"Error: failed to resolve hostname '{urlparse(str(current_url)).hostname}'"
                except ValueError as e:
                    return f"Error: {e}"

                last_error = ""
                for attempt in range(MAX_RETRIES + 1):
                    try:
                        cm = client.stream("GET", current_url)
                        response = await cm.__aenter__()
                        stream_ctx = cm  # track only after a successful enter
                    except httpx.ConnectError as e:
                        return f"Error: connection failed -- {e}"
                    except httpx.TimeoutException:
                        return f"Error: request timed out after {timeout}s"

                    if response.status_code < 400:
                        break  # success/redirect -- leave the stream OPEN for now

                    if response.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES:
                        last_error = f"HTTP {response.status_code}: {response.reason_phrase}"
                        wait = 2**attempt
                        _logger.warning(
                            "Retrying %s (status %d, attempt %d/%d)",
                            current_url,
                            response.status_code,
                            attempt + 1,
                            MAX_RETRIES,
                        )
                        await _aclose_stream(stream_ctx)
                        stream_ctx = None
                        response = None
                        await asyncio.sleep(wait)
                        continue

                    err = f"Error: HTTP {response.status_code} -- {response.reason_phrase}"
                    await _aclose_stream(stream_ctx)
                    stream_ctx = None
                    response = None
                    return err
                else:
                    return f"Error: Max retries exceeded: {last_error}"

                # Follow the redirect manually so the next hop is SSRF-checked.
                if response.status_code in (301, 302, 303, 307, 308):
                    loc = response.headers.get("location")
                    if not loc:
                        break  # no Location -> stop hopping; keep stream OPEN, read body below
                    await _aclose_stream(stream_ctx)
                    stream_ctx = None
                    response = None
                    current_url = str(httpx.URL(current_url).join(loc))
                    continue
                break  # 2xx success -> exit the redirect loop, stream stays OPEN
            else:
                return "Error: too many redirects"

            # Content-Length pre-check: reject without consuming the body when the
            # header itself declares an oversized payload.
            cl = response.headers.get("Content-Length")
            if cl is not None:
                try:
                    cl_int = int(cl)
                except (TypeError, ValueError):
                    cl_int = None
                if cl_int is not None and cl_int > MAX_RESPONSE_SIZE:
                    return f"Error: response too large ({cl_int} bytes exceeds {MAX_RESPONSE_SIZE} byte limit)"

            # Bounded streaming read: STOP once the cap is crossed so the rest of
            # the body is never pulled into memory.
            async for chunk in response.aiter_bytes():
                buf.extend(chunk)
                if len(buf) > MAX_RESPONSE_SIZE:
                    oversized = True
                    break
        finally:
            await _aclose_stream(stream_ctx)

    raw = bytes(buf[: MAX_RESPONSE_SIZE + 1])
    truncated = oversized or len(buf) > MAX_RESPONSE_SIZE

    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        text = raw.decode("latin-1", errors="replace")

    if "html" in text[:500].lower():
        text = _extract_html_content(text)

    if truncated:
        text = text[:MAX_OUTPUT]
        text += f"\n... (response truncated at {MAX_OUTPUT} chars, body exceeds {MAX_RESPONSE_SIZE} bytes)"
    elif len(text) > MAX_OUTPUT:
        text = text[:MAX_OUTPUT] + f"\n... (response truncated, total {len(text)} chars)"

    return text
