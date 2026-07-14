"""koboi/websearch/providers/ddg.py -- DuckDuckGo HTML scrape (fallback search provider).

Migrated verbatim from ``koboi/tools/builtin/web.py`` (``_DDGResultParser`` +
``_search_duckduckgo``). Unreliable by nature -- the html.duckduckgo.com endpoint is
bot-blocked and the CSS classes drift -- so it is a fallback, never a default.
``DDGSearchProvider`` wraps the same scrape for the registry path.
"""

from __future__ import annotations

from html.parser import HTMLParser
from urllib.parse import quote_plus

import httpx

from koboi.websearch.base import BaseSearchProvider
from koboi.websearch.providers.mock import _format_results
from koboi.websearch.registry import register_search_provider
from koboi.websearch.types import SearchResult

_DDGG_ENDPOINT = "https://html.duckduckgo.com/html/"
_DDGG_HTML_CAP = 50000


class _DDGResultParser(HTMLParser):
    """Parse DuckDuckGo html endpoint result anchors into ``{title, url, snippet}`` dicts."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict] = []
        self._in_title = False
        self._in_snippet = False
        self._current_url = ""
        self._current_title = ""
        self._current_snippet = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        css_class = attrs_dict.get("class", "")

        if tag == "a" and "result__a" in css_class:
            self._in_title = True
            self._current_url = attrs_dict.get("href", "") or ""
            self._current_title = ""
        elif tag == "a" and "result__snippet" in css_class:
            self._in_snippet = True
            self._current_snippet = ""

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._current_title += data
        elif self._in_snippet:
            self._current_snippet += data

    def handle_endtag(self, tag: str) -> None:
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


async def _ddg_query(query: str) -> list[dict]:
    """Fetch the DDG html endpoint and parse results. Raises on network error."""
    encoded = quote_plus(query)
    url = f"{_DDGG_ENDPOINT}?q={encoded}"
    async with httpx.AsyncClient(
        timeout=15,
        headers={"User-Agent": "AI-Agent-Framework/1.0"},
        follow_redirects=True,
    ) as client:
        resp = await client.get(url)
        html = resp.text[:_DDGG_HTML_CAP]

    parser = _DDGResultParser()
    parser.feed(html)
    return parser.results


async def _search_duckduckgo(query: str) -> str:
    """Legacy DDG HTML scrape (string output). Kept for back-compat + the WEB_SEARCH_PROVIDER path."""
    try:
        results = await _ddg_query(query)
    except Exception as e:  # noqa: BLE001 - boundary: any failure becomes an error string
        return f"Error: search failed — {e}"

    if not results:
        return f"No results found for '{query}'"

    return _format_results(query, results[:10])


@register_search_provider("ddg", description="DuckDuckGo HTML scrape (fallback; no API key)")
class DDGSearchProvider(BaseSearchProvider):
    """Registry-backed DuckDuckGo scrape (unreliable -- use as a fallback only)."""

    def __init__(self, max_results: int = 10) -> None:
        self._max_results = max_results

    async def search(self, query: str, *, max_results: int = 10) -> list[SearchResult]:
        limit = max_results or self._max_results
        results = await _ddg_query(query)
        return [SearchResult(title=r["title"], url=r["url"], snippet=r.get("snippet", "")) for r in results[:limit]]
