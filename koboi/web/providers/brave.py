"""koboi/web/providers/brave.py -- Brave Search API provider.

REST + JSON, no JS rendering needed for results. Auth via ``X-Subscription-Token``.
Free tier ~2000 queries/month. Configure under ``web.search.brave``::

    web:
      search:
        provider: brave
        brave:
          api_key: ${BRAVE_API_KEY:}
          country: "US"        # optional
          freshness: "pw"      # optional (past week); omit for all time
"""

from __future__ import annotations

import logging
import os

import httpx

from koboi.web.base import BaseSearchProvider
from koboi.web.registry import register_search_provider
from koboi.web.types import SearchResult

_logger = logging.getLogger(__name__)

_BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_BRAVE_MAX_COUNT = 20  # Brave caps count at 20 per request.


@register_search_provider("brave", description="Brave Search API (REST + JSON)")
class BraveSearchProvider(BaseSearchProvider):
    """Brave Web Search via the REST API."""

    def __init__(
        self,
        api_key: str = "",
        country: str = "",
        freshness: str = "",
        max_results: int = 10,
        timeout: int = 15,
    ) -> None:
        self._api_key = api_key or os.getenv("BRAVE_API_KEY", "")
        self._country = country
        self._freshness = freshness
        self._max_results = max_results
        self._timeout = timeout

    async def search(self, query: str, *, max_results: int = 10) -> list[SearchResult]:
        if not self._api_key:
            raise ValueError("Brave provider requires an api_key (web.search.brave.api_key or BRAVE_API_KEY)")

        limit = max_results or self._max_results
        params: dict[str, str | int] = {"q": query, "count": min(limit, _BRAVE_MAX_COUNT)}
        if self._country:
            params["country"] = self._country
        if self._freshness:
            params["freshness"] = self._freshness

        headers = {"X-Subscription-Token": self._api_key, "Accept": "application/json"}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(_BRAVE_ENDPOINT, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        # Brave wraps web hits under `web.results`; other keys (news, videos) are ignored.
        web_block = data.get("web", {}) if isinstance(data, dict) else {}
        raw_results = web_block.get("results") or [] if isinstance(web_block, dict) else []

        out: list[SearchResult] = []
        for r in raw_results[:limit]:
            if not isinstance(r, dict):
                continue
            url = r.get("url", "") or ""
            if not url:
                continue
            out.append(
                SearchResult(
                    title=r.get("title", "") or url,
                    url=url,
                    snippet=r.get("description", "") or "",
                    score=None,
                    raw=r,
                )
            )
        _logger.debug("brave search '%s' -> %d results", query, len(out))
        return out
