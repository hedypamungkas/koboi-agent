"""koboi/tools/builtin/web -- Web search and URL fetching."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import socket
import ipaddress
from html.parser import HTMLParser
from urllib.parse import quote_plus, urlparse

import httpx

from koboi.tools.registry import tool
from koboi.types import RiskLevel

_logger = logging.getLogger(__name__)

# ── web_search helpers ──

WEB_SEARCH_PROVIDER = os.getenv("WEB_SEARCH_PROVIDER", "mock")

SEARCH_INDEX: dict[str, list[dict]] = {
    "python": [
        {
            "title": "Python Documentation",
            "url": "https://docs.python.org/3/",
            "snippet": "Official Python 3 documentation — tutorials, library reference, and guides.",
        },
        {
            "title": "Real Python",
            "url": "https://realpython.com/",
            "snippet": "Python tutorials, guides, and best practices for developers.",
        },
    ],
    "asyncio": [
        {
            "title": "Async IO in Python",
            "url": "https://docs.python.org/3/library/asyncio.html",
            "snippet": "Coroutines, event loops, tasks, and futures for async programming.",
        },
    ],
    "react": [
        {
            "title": "React Documentation",
            "url": "https://react.dev/",
            "snippet": "Learn React — components, hooks, state management.",
        },
    ],
    "typescript": [
        {
            "title": "TypeScript Handbook",
            "url": "https://www.typescriptlang.org/docs/handbook/",
            "snippet": "TypeScript type system, interfaces, generics, and modules.",
        },
    ],
    "fastapi": [
        {
            "title": "FastAPI Documentation",
            "url": "https://fastapi.tiangolo.com/",
            "snippet": "Modern Python web framework with automatic OpenAPI docs.",
        },
    ],
    "docker": [
        {
            "title": "Docker Documentation",
            "url": "https://docs.docker.com/",
            "snippet": "Container platform — build, ship, and run applications.",
        },
    ],
    "git": [
        {
            "title": "Git Documentation",
            "url": "https://git-scm.com/doc",
            "snippet": "Version control system — branching, merging, and collaboration.",
        },
    ],
    "ai": [
        {
            "title": "Anthropic API Docs",
            "url": "https://docs.anthropic.com/",
            "snippet": "Claude API — messages, tool use, and streaming.",
        },
        {
            "title": "OpenAI API Docs",
            "url": "https://platform.openai.com/docs",
            "snippet": "GPT models, embeddings, and fine-tuning.",
        },
    ],
    "agent": [
        {
            "title": "LangChain Docs",
            "url": "https://python.langchain.com/",
            "snippet": "Framework for building LLM-powered applications and agents.",
        },
        {"title": "CrewAI", "url": "https://docs.crewai.com/", "snippet": "Multi-agent orchestration framework."},
    ],
    "mcp": [
        {
            "title": "Model Context Protocol",
            "url": "https://modelcontextprotocol.io/",
            "snippet": "Open standard for connecting AI assistants to external tools and data.",
        },
    ],
}


def _format_results(query: str, results: list[dict]) -> str:
    lines = [f"Search results for '{query}':"]
    seen = set()
    for r in results:
        if r["url"] not in seen:
            seen.add(r["url"])
            lines.append(f"  - {r['title']}: {r['url']}")
            lines.append(f"    {r['snippet']}")
    return "\n".join(lines)


def _search_mock(query: str) -> str:
    q = query.lower().strip()
    results = []

    query_tokens = set(q.split())
    for key, entries in SEARCH_INDEX.items():
        key_tokens = set(key.split())
        if query_tokens & key_tokens:
            results.extend(entries)

    if not results:
        available = ", ".join(sorted(SEARCH_INDEX.keys()))
        return f"No results found for '{query}'. Available topics: {available}"

    return _format_results(query, results)


class _DDGResultParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results: list[dict] = []
        self._in_title = False
        self._in_snippet = False
        self._current_url = ""
        self._current_title = ""
        self._current_snippet = ""

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        css_class = attrs_dict.get("class", "")

        if tag == "a" and "result__a" in css_class:
            self._in_title = True
            self._current_url = attrs_dict.get("href", "")
            self._current_title = ""
        elif tag == "a" and "result__snippet" in css_class:
            self._in_snippet = True
            self._current_snippet = ""

    def handle_data(self, data):
        if self._in_title:
            self._current_title += data
        elif self._in_snippet:
            self._current_snippet += data

    def handle_endtag(self, tag):
        if tag == "a" and self._in_title:
            self._in_title = False
        elif tag == "a" and self._in_snippet:
            self._in_snippet = False
            if self._current_title.strip() and self._current_url:
                self.results.append(
                    {
                        "title": self._current_title.strip(),
                        "url": self._current_url.strip(),
                        "snippet": self._current_snippet.strip() or "(no description)",
                    }
                )


async def _search_duckduckgo(query: str) -> str:
    encoded = quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded}"

    try:
        async with httpx.AsyncClient(
            timeout=15,
            headers={"User-Agent": "AI-Agent-Framework/1.0"},
            follow_redirects=True,
        ) as client:
            resp = await client.get(url)
            html = resp.text[:50000]
    except Exception as e:
        return f"Error: search failed — {e}"

    parser = _DDGResultParser()
    parser.feed(html)

    if not parser.results:
        return f"No results found for '{query}'"

    return _format_results(query, parser.results[:10])


@tool(
    name="web_search",
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
)
async def web_search(query: str) -> str:
    if WEB_SEARCH_PROVIDER == "duckduckgo":
        return await _search_duckduckgo(query)
    return _search_mock(query)


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

PRIVATE_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]


def _resolve_and_check(hostname: str) -> list[str]:
    """Resolve hostname and check all IPs against private networks."""
    addrs = socket.getaddrinfo(hostname, None)
    if not addrs:
        raise ValueError(f"DNS resolution returned no addresses for '{hostname}'")

    resolved = []
    for _, _, _, _, sa in addrs:
        ip_str = sa[0]
        ip = ipaddress.ip_address(ip_str)
        for net in PRIVATE_NETWORKS:
            if ip in net:
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


@tool(
    name="web_fetch",
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
)
async def web_fetch(url: str, timeout: int = 15) -> str:
    if not url.startswith(("http://", "https://")):
        return "Error: URL must start with http:// or https://"

    timeout = max(1, min(timeout, MAX_TIMEOUT))

    # SSRF check: resolve DNS and verify the IP is not private
    try:
        await asyncio.to_thread(_check_url_ssrf, url)
    except socket.gaierror:
        return f"Error: failed to resolve hostname '{urlparse(url).hostname}'"
    except ValueError as e:
        return f"Error: {e}"

    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        max_redirects=10,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        last_error = ""
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await client.get(url)
            except httpx.ConnectError as e:
                return f"Error: connection failed -- {e}"
            except httpx.TimeoutException:
                return f"Error: request timed out after {timeout}s"

            if response.status_code < 400:
                break

            if response.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES:
                last_error = f"HTTP {response.status_code}: {response.reason_phrase}"
                wait = 2**attempt
                _logger.warning(
                    "Retrying %s (status %d, attempt %d/%d)",
                    url,
                    response.status_code,
                    attempt + 1,
                    MAX_RETRIES,
                )
                await asyncio.sleep(wait)
                continue

            return f"Error: HTTP {response.status_code} -- {response.reason_phrase}"
        else:
            return f"Error: Max retries exceeded: {last_error}"

        raw = response.content[: MAX_RESPONSE_SIZE + 1]
        truncated = len(response.content) > MAX_RESPONSE_SIZE

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
