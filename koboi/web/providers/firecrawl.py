"""koboi/web/providers/firecrawl.py -- Firecrawl search provider.

Firecrawl's ``/v1/search`` returns hits and (optionally) their scraped markdown in one
call when ``scrape_results`` is set -- handy for deep research (search + fetch fused).
A ``FirecrawlFetchProvider`` (``/v1/scrape``) and a RAG crawl source are added in Wave 1.

Configure under ``web.search.firecrawl``::

    web:
      search:
        provider: firecrawl
        firecrawl:
          api_key: ${FIRECRAWL_API_KEY:}
          scrape_results: true   # embed markdown into each SearchResult snippet
"""

from __future__ import annotations

import logging
import os

import httpx

from koboi.web.base import BaseSearchProvider
from koboi.web.registry import register_search_provider
from koboi.web.types import SearchResult

_logger = logging.getLogger(__name__)

_FIRECRAWL_SEARCH_ENDPOINT = "https://api.firecrawl.dev/v1/search"
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
