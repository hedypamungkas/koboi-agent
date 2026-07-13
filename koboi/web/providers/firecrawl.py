"""koboi/web/providers/firecrawl.py -- Firecrawl search + fetch providers.

Search (``/v1/search``) returns hits and (optionally) their scraped markdown in one call
when ``scrape_results`` is set -- handy for deep research (search + fetch fused). Fetch
(``/v1/scrape``) returns one URL's cleaned markdown. A RAG crawl source (``/v1/crawl``)
lives in ``koboi/rag/sources.py`` (``source: firecrawl``).

Configure under ``web.search.firecrawl`` / ``web.fetch.firecrawl``::

    web:
      search:
        provider: firecrawl
        firecrawl:
          api_key: ${FIRECRAWL_API_KEY:}
          scrape_results: true   # embed markdown into each SearchResult snippet
      fetch:
        provider: firecrawl
        firecrawl:
          api_key: ${FIRECRAWL_API_KEY:}
          only_main_content: true
"""

from __future__ import annotations

import logging
import os

import httpx

from koboi.web.base import BaseFetchProvider, BaseSearchProvider
from koboi.web.registry import register_fetch_provider, register_search_provider
from koboi.web.types import FetchResult, SearchResult

_logger = logging.getLogger(__name__)

_FIRECRAWL_SEARCH_ENDPOINT = "https://api.firecrawl.dev/v1/search"
_FIRECRAWL_SCRAPE_ENDPOINT = "https://api.firecrawl.dev/v1/scrape"
_SNIPPET_PREVIEW = 500


@register_search_provider("firecrawl", description="Firecrawl search API (optionally with scraped markdown)")
class FirecrawlSearchProvider(BaseSearchProvider):
    """Firecrawl search via ``/v1/search``."""

    def __init__(
        self,
        api_key: str = "",
        scrape_results: bool = False,
        max_results: int = 10,
        timeout: int = 30,
    ) -> None:
        self._api_key = api_key or os.getenv("FIRECRAWL_API_KEY", "")
        self._scrape_results = scrape_results
        self._max_results = max_results
        self._timeout = timeout

    async def search(self, query: str, *, max_results: int = 10) -> list[SearchResult]:
        if not self._api_key:
            raise ValueError(
                "Firecrawl provider requires an api_key (web.search.firecrawl.api_key or FIRECRAWL_API_KEY)"
            )

        limit = max_results or self._max_results
        body: dict = {"query": query, "limit": limit}
        if self._scrape_results:
            # Ask Firecrawl to return markdown for each hit so the caller can skip a
            # separate fetch step (deep-research search->fetch fusion).
            body["scrapeOptions"] = {"formats": ["markdown"]}

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(_FIRECRAWL_SEARCH_ENDPOINT, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        items = data.get("data", []) if isinstance(data, dict) else []
        out: list[SearchResult] = []
        for r in items[:limit]:
            if not isinstance(r, dict):
                continue
            url = r.get("url", "") or ""
            if not url:
                continue
            snippet = r.get("description", "") or ""
            if self._scrape_results and r.get("markdown"):
                md = r["markdown"]
                snippet = (md[:_SNIPPET_PREVIEW] + "...") if len(md) > _SNIPPET_PREVIEW else md
            out.append(
                SearchResult(
                    title=r.get("title", "") or url,
                    url=url,
                    snippet=snippet,
                    score=None,
                    raw=r,
                )
            )
        _logger.debug("firecrawl search '%s' -> %d results", query, len(out))
        return out


@register_fetch_provider("firecrawl", description="Firecrawl /v1/scrape -> markdown")
class FirecrawlFetchProvider(BaseFetchProvider):
    """Single-URL fetch via Firecrawl ``/v1/scrape`` (cleaned markdown)."""

    def __init__(
        self,
        api_key: str = "",
        only_main_content: bool = True,
        timeout: int = 30,
    ) -> None:
        self._api_key = api_key or os.getenv("FIRECRAWL_API_KEY", "")
        self._only_main_content = only_main_content
        self._timeout = timeout

    async def fetch(self, url: str, *, render: str = "auto", timeout: int = 15) -> FetchResult:
        if not self._api_key:
            raise ValueError("Firecrawl fetch requires an api_key (web.fetch.firecrawl.api_key or FIRECRAWL_API_KEY)")
        if not url.startswith(("http://", "https://")):
            return FetchResult(url=url, content="", status=0, metadata={"error": "invalid scheme"})

        # Defense in depth: validate the URL client-side before handing it to the SaaS so the
        # agent can't use Firecrawl as a probe for internal topology / metadata endpoints.
        # Lazy import: koboi.tools.builtin.web imports this package at init time (circular).
        from koboi.tools.builtin.web import _check_url_ssrf as _guard  # noqa: PLC0415

        try:
            _guard(url)
        except ValueError as exc:
            return FetchResult(url=url, content="", status=0, metadata={"error": str(exc)})

        body = {"url": url, "formats": ["markdown"], "onlyMainContent": self._only_main_content}
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(_FIRECRAWL_SCRAPE_ENDPOINT, json=body, headers=headers)
        except httpx.HTTPError as exc:
            return FetchResult(url=url, content="", status=0, metadata={"error": f"transport error: {exc}"})
        if resp.status_code >= 400:
            return FetchResult(
                url=url, content="", status=resp.status_code, metadata={"error": f"HTTP {resp.status_code}"}
            )

        data = resp.json().get("data", {}) if resp.json().get("data") else {}
        if not isinstance(data, dict):
            data = {}
        content = data.get("markdown", "") or data.get("html", "") or ""
        meta = data.get("metadata") or {}
        title = meta.get("title", "") if isinstance(meta, dict) else ""
        content_type = "markdown" if data.get("markdown") else "text"
        if not content.strip():
            _logger.warning(
                "firecrawl scrape '%s' returned empty content (likely JS-rendered or paywalled) "
                "-- the node should skip this URL and try another source",
                url,
            )
        else:
            _logger.debug("firecrawl scrape '%s' -> %d chars", url, len(content))
        return FetchResult(
            url=url,
            content=content,
            title=title,
            content_type=content_type,
            status=resp.status_code,
            metadata=meta if isinstance(meta, dict) else {},
        )
