"""koboi/web/base.py -- provider ABCs for the web I/O layer.

Two capabilities, two ABCs (a provider may implement both -- e.g. Firecrawl does
search + scrape). Mirrors the one-ABC-per-stage shape of ``koboi/rag``.

Security contract: every fetch provider MUST validate each URL (and each redirect
hop) against a private-range SSRF guard before fetching. For in-process HTTP
clients that is ``koboi.tools.builtin.web._check_url_ssrf``; SaaS renderers
(Firecrawl, Playwright) must validate on the client side too -- defense in depth,
so the agent cannot use a provider as a probe for internal topology or a
metadata-endpoint oracle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from koboi.web.types import FetchResult, SearchResult


class BaseSearchProvider(ABC):
    """Query -> ranked list of ``SearchResult``."""

    @abstractmethod
    async def search(self, query: str, *, max_results: int = 10) -> list[SearchResult]:
        """Return up to ``max_results`` hits for ``query`` (most relevant first)."""
        ...


class BaseFetchProvider(ABC):
    """URL -> cleaned ``FetchResult`` (markdown/plain text, boilerplate removed)."""

    @abstractmethod
    async def fetch(self, url: str, *, render: str = "auto", timeout: int = 15) -> FetchResult:
        """Fetch and extract ``url``.

        ``render`` is a hint: ``never`` (static extract only), ``always`` (force a JS
        renderer), or ``auto`` (provider may escalate if static extract is thin/SPA).
        Implementations ignore hints they cannot honor.
        """
        ...
