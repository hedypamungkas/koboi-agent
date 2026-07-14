"""koboi/websearch/providers/readability.py -- default fetch provider (httpx + trafilatura).

Static fetch with readability extraction. JS rendering (``render: auto|always``) is a
Wave 1.5 concern (Playwright); this provider honors only static extraction in Wave 1 and
logs when a JS renderer is requested but unavailable.

Reuses the shared SSRF guard + fetch constants from ``koboi.tools.builtin.web`` (the SSRF
primitives must stay there -- they are imported by dotted path from MCP modules and patched
by path in several test suites). Those symbols are accessed lazily (``_web()``) rather than
imported at module top: this provider is imported during ``koboi.websearch`` package init, which
``koboi.tools.builtin.web`` itself triggers at import time, so a top-level back-import would
be circular. The fetch loop mirrors ``web_fetch``'s per-hop SSRF + redirect + retry semantics
but returns a structured ``FetchResult`` and applies readability.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from koboi.websearch.base import BaseFetchProvider
from koboi.websearch.registry import register_fetch_provider
from koboi.websearch.types import FetchResult

_logger = logging.getLogger(__name__)

# Optional readability extraction (boilerplate -> clean markdown/text). Absent in CI (no
# [web] extra); the _extract_html_content fallback keeps the provider functional offline.
try:  # pragma: no cover - dep guard
    import trafilatura  # type: ignore[import-not-found]

    _TRAFILATURA_AVAILABLE = True
except ImportError:  # pragma: no cover - dep guard
    trafilatura = None  # type: ignore[assignment]
    _TRAFILATURA_AVAILABLE = False

_MAX_REDIRECTS = 5


def _web():
    """Lazy accessor for ``koboi.tools.builtin.web`` symbols (SSRF guard + fetch constants).

    Imported lazily to avoid a circular import: this module is imported during
    ``koboi.websearch`` package init, which ``koboi.tools.builtin.web`` triggers at its own
    import time. ``import`` is cached after the first call, so this is a dict lookup.
    """
    import koboi.tools.builtin.web as _w  # noqa: PLC0415 - lazy to break a import cycle

    return _w


def _extract_with_readability(text: str) -> tuple[str, str]:
    """Return ``(content, content_type)``. trafilatura-first when available, else the regex extractor."""
    if _TRAFILATURA_AVAILABLE:
        try:
            extracted = trafilatura.extract(  # type: ignore[union-attr]
                text,
                output_format="markdown",
                include_comments=False,
                include_tables=True,
                with_metadata=False,
            )
            if extracted and extracted.strip():
                return extracted, "markdown"
        except Exception as exc:  # noqa: BLE001 - readability is best-effort; fall back below
            _logger.debug("trafilatura failed (%s); falling back to regex extractor", exc)
    w = _web()
    # Only run the regex extractor on HTML-looking payloads (mirrors web_fetch's gate).
    if "html" in text[:500].lower():
        return w._extract_html_content(text), "text"
    return text, "text"


@register_fetch_provider("httpx", description="httpx + trafilatura readability (default fetch)")
class ReadabilityFetchProvider(BaseFetchProvider):
    """Static fetch + readability extraction (the default fetch provider)."""

    def __init__(self, max_chars: int = 20000, timeout: int = 15, render: str = "never") -> None:
        self._max_chars = max_chars
        self._default_timeout = timeout
        self._default_render = render

    async def fetch(self, url: str, *, render: str = "auto", timeout: int = 15) -> FetchResult:
        w = _web()
        if render in ("auto", "always"):
            _logger.info("ReadabilityFetchProvider is static-only in Wave 1; render=%r treated as 'never'", render)

        if not url.startswith(("http://", "https://")):
            return FetchResult(url=url, content="", status=0, metadata={"error": "invalid scheme"})

        timeout_s = max(1, min(int(timeout or self._default_timeout), w.MAX_TIMEOUT))
        response: httpx.Response | None = None
        async with httpx.AsyncClient(
            timeout=timeout_s, follow_redirects=False, headers={"User-Agent": w.USER_AGENT}
        ) as client:
            current_url = url
            for _ in range(_MAX_REDIRECTS + 1):
                # SSRF guard on every hop (initial URL + each redirect target).
                try:
                    await asyncio.to_thread(w._check_url_ssrf, str(current_url))
                except ValueError as exc:
                    return FetchResult(url=str(current_url), content="", status=0, metadata={"error": str(exc)})
                except OSError:  # socket.gaierror etc.
                    return FetchResult(
                        url=str(current_url), content="", status=0, metadata={"error": "DNS resolution failed"}
                    )

                response, err = await self._fetch_with_retry(client, current_url)
                if err is not None:
                    return FetchResult(url=str(current_url), content="", status=0, metadata={"error": err})

                if response.status_code in (301, 302, 303, 307, 308):
                    loc = response.headers.get("location")
                    if not loc:
                        break
                    current_url = str(httpx.URL(current_url).join(loc))
                    continue
                break
            else:
                return FetchResult(url=url, content="", status=0, metadata={"error": "too many redirects"})

        if response is None:  # defensive: every loop path either returns or sets response
            return FetchResult(url=url, content="", status=0, metadata={"error": "no response"})

        final_url = str(current_url)
        raw = response.content[: w.MAX_RESPONSE_SIZE + 1]
        truncated = len(response.content) > w.MAX_RESPONSE_SIZE
        text = raw.decode("utf-8", errors="replace")
        content, content_type = _extract_with_readability(text)
        if len(content) > self._max_chars:
            content = content[: self._max_chars]
            truncated = True
        return FetchResult(
            url=final_url,
            content=content,
            content_type=content_type,
            status=response.status_code,
            truncated=truncated,
        )

    @staticmethod
    async def _fetch_with_retry(client: httpx.AsyncClient, url: str) -> tuple[httpx.Response | None, str | None]:
        """GET with bounded retry on transient status. Returns (response, error)."""
        w = _web()
        last_error = ""
        for attempt in range(w.MAX_RETRIES + 1):
            try:
                resp = await client.get(url)
            except httpx.ConnectError as exc:
                return None, f"connection failed: {exc}"
            except httpx.TimeoutException:
                return None, "request timed out"
            if resp.status_code < 400:
                return resp, None
            if resp.status_code in w.RETRYABLE_STATUS and attempt < w.MAX_RETRIES:
                last_error = f"HTTP {resp.status_code}"
                await asyncio.sleep(2**attempt)
                continue
            return None, f"HTTP {resp.status_code}"
        return None, f"max retries exceeded: {last_error}"
