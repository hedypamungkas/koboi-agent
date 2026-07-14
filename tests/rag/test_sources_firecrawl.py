"""Tests for koboi/rag/sources.py -- the ``source: firecrawl`` site-crawl loader."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import koboi.rag.sources as sources
from koboi.rag.sources import DocumentCache, _firecrawl_crawl, fetch_firecrawl_entry

_SSRF_SRC = "koboi.rag.sources._check_url_ssrf"


@pytest.fixture(autouse=True)
def _bypass_ssrf():
    # Default: bypass the SSRF guard so crawl tests with non-resolvable seed hosts don't
    # hit DNS. The SSRF-rejection test overrides this by patching the guard to raise.
    with patch(_SSRF_SRC, return_value=None):
        yield


def _pages(*pairs: tuple[str, str]) -> list[dict]:
    return [{"markdown": md, "metadata": {"sourceURL": url, "title": url}} for url, md in pairs]


class TestFetchFirecrawlEntry:
    def test_missing_api_key_skips(self, monkeypatch):
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        out = list(fetch_firecrawl_entry({"url": "https://docs.example.com"}, None))
        assert out == []

    def test_ssrf_blocks_metadata_seed(self):
        with patch(_SSRF_SRC, side_effect=ValueError("internal IP")):
            out = list(fetch_firecrawl_entry({"url": "http://169.254.169.254/latest", "api_key": "k"}, None))
        assert out == []

    def test_crawl_yields_pages(self, monkeypatch):
        monkeypatch.setattr(
            sources,
            "_firecrawl_crawl",
            lambda *a, **k: _pages(
                ("https://docs.example.com/a", "# A\n\nalpha"),
                ("https://docs.example.com/b", "# B\n\nbeta"),
            ),
        )
        out = list(fetch_firecrawl_entry({"url": "https://docs.example.com", "api_key": "k"}, None))
        assert len(out) == 2
        names = {name for name, _ in out}
        assert names == {"a", "b"}
        assert b"alpha" in out[0][1]

    def test_one_empty_page_is_skipped(self, monkeypatch):
        monkeypatch.setattr(
            sources,
            "_firecrawl_crawl",
            lambda *a, **k: [
                {"markdown": "alpha", "metadata": {"sourceURL": "https://x/a"}},
                {"markdown": "", "metadata": {"sourceURL": "https://x/b"}},
                {"markdown": "gamma", "metadata": {"sourceURL": "https://x/c"}},
            ],
        )
        out = list(fetch_firecrawl_entry({"url": "https://x", "api_key": "k"}, None))
        assert len(out) == 2  # the empty-markdown page is skipped, not fatal

    def test_crawl_failure_skips_silently(self, monkeypatch):
        def _boom(*_a, **_k):
            raise RuntimeError("api down")

        monkeypatch.setattr(sources, "_firecrawl_crawl", _boom)
        out = list(fetch_firecrawl_entry({"url": "https://x", "api_key": "k"}, None))
        assert out == []

    def test_per_page_cache_avoids_reput(self, monkeypatch, tmp_path):
        # The crawl itself re-runs each call (page URLs aren't known up front -- mirrors
        # fetch_s3_entry re-listing), but each page's bytes are cached so the 2nd call hits
        # the cache and does NOT re-put.
        monkeypatch.setattr(
            sources,
            "_firecrawl_crawl",
            lambda *a, **k: _pages(("https://docs.example.com/a", "# A\n\nalpha")),
        )
        cache = DocumentCache(str(tmp_path / "fc_cache"))
        put_calls: list[str] = []
        orig_put = cache.put

        def _counting_put(key: str, name: str, data: bytes):
            put_calls.append(key)
            return orig_put(key, name, data)

        cache.put = _counting_put  # type: ignore[method-assign]
        entry = {"url": "https://docs.example.com", "api_key": "k"}

        first = list(fetch_firecrawl_entry(entry, cache))
        second = list(fetch_firecrawl_entry(entry, cache))

        assert len(first) == 1 and len(second) == 1
        assert len(put_calls) == 1  # put ran once (1st call); 2nd call hit the cache
        assert first[0][1] == second[0][1]


class TestFirecrawlCrawlPollLoop:
    """Covers _firecrawl_crawl's POST -> poll -> completed path with mocked sync httpx."""

    class _Resp:
        def __init__(self, json_data: dict, status: int = 200) -> None:
            self._json = json_data
            self.status_code = status

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return self._json

    class _Client:
        def __init__(self, post_resp, get_resps) -> None:
            self._post = post_resp
            self._gets = list(get_resps)

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def post(self, *_a, **_k):
            return self._post

        def get(self, *_a, **_k):
            return self._gets.pop(0)

    def test_completed_returns_data(self):
        post_resp = self._Resp({"id": "job1"})
        get_resp = self._Resp({"status": "completed", "data": [{"markdown": "x"}]})
        client = self._Client(post_resp, [get_resp])
        with patch("httpx.Client", return_value=client):
            pages = _firecrawl_crawl("https://docs.example.com", "k", 10, None)
        assert pages == [{"markdown": "x"}]

    def test_sync_data_payload(self):
        post_resp = self._Resp({"data": [{"markdown": "sync"}]})  # no job id -> sync
        client = self._Client(post_resp, [])
        with patch("httpx.Client", return_value=client):
            pages = _firecrawl_crawl("https://docs.example.com", "k", 10, None)
        assert pages == [{"markdown": "sync"}]
