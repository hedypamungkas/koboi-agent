"""Tests for koboi/tools/builtin/ingest.py -- the ingest_url tool (W3). Offline (fake provider)."""

from __future__ import annotations

from koboi.rag.live import LiveCorpus
from koboi.tools.builtin.ingest import ingest_url
from koboi.web.types import FetchResult


class _FakeProvider:
    def __init__(
        self,
        content: str = "Article body about solid-state batteries.",
        title: str = "Batteries",
        error: str | None = None,
    ) -> None:
        self.content = content
        self.title = title
        self.error = error

    async def fetch(self, url: str, *, timeout: int = 15) -> FetchResult:
        if self.error:
            return FetchResult(url=url, content="", status=0, metadata={"error": self.error})
        return FetchResult(url=url, content=self.content, title=self.title, content_type="markdown")


class TestIngestUrl:
    async def test_ingests_into_corpus(self):
        corpus = LiveCorpus([])
        out = await ingest_url(
            "https://example.com/article",
            _deps={"fetch_provider": _FakeProvider(), "live_corpus": corpus},
        )
        assert "Ingested" in out
        assert len(corpus.chunks) >= 1
        assert corpus.dirty is True
        assert corpus.chunks[0].metadata.get("source") == "https://example.com/article"

    async def test_missing_fetch_provider(self):
        out = await ingest_url("https://example.com", _deps={"live_corpus": LiveCorpus([])})
        assert "Error" in out and "fetch_provider" in out

    async def test_missing_live_corpus(self):
        out = await ingest_url("https://example.com", _deps={"fetch_provider": _FakeProvider()})
        assert "Error" in out and "live_corpus" in out

    async def test_no_deps_at_all(self):
        out = await ingest_url("https://example.com")
        assert "Error" in out

    async def test_invalid_scheme(self):
        out = await ingest_url(
            "ftp://example.com",
            _deps={"fetch_provider": _FakeProvider(), "live_corpus": LiveCorpus([])},
        )
        assert "Error" in out and "http" in out

    async def test_empty_content(self):
        out = await ingest_url(
            "https://example.com",
            _deps={"fetch_provider": _FakeProvider(content="   "), "live_corpus": LiveCorpus([])},
        )
        assert "No ingestable content" in out

    async def test_fetch_error_propagates(self):
        out = await ingest_url(
            "https://example.com",
            _deps={"fetch_provider": _FakeProvider(error="internal IP"), "live_corpus": LiveCorpus([])},
        )
        assert "Error" in out and "internal IP" in out
