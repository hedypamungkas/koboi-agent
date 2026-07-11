"""Tests for koboi/rag/live.py -- LiveCorpus + LiveRetriever (W3)."""

from __future__ import annotations

from koboi.rag.live import LiveCorpus, LiveRetriever
from koboi.rag.types import Chunk


class TestLiveCorpus:
    def test_seed_sets_dirty(self):
        c = LiveCorpus([Chunk(id="c1", doc_id="d", content="x")])
        assert c.dirty is True
        assert len(c.chunks) == 1

    def test_empty_seed_not_dirty(self):
        assert LiveCorpus([]).dirty is False

    def test_add_chunks_sets_dirty(self):
        c = LiveCorpus([])
        assert c.dirty is False
        c.add_chunks([Chunk(id="c1", doc_id="d", content="x")])
        assert c.dirty is True
        assert len(c.chunks) == 1

    def test_add_empty_is_noop(self):
        c = LiveCorpus([])
        c.add_chunks([])
        assert c.dirty is False
        assert c.chunks == []

    def test_mark_clean(self):
        c = LiveCorpus([Chunk(id="c1", doc_id="d", content="x")])
        c.mark_clean()
        assert c.dirty is False


class TestLiveRetriever:
    async def test_sees_seed_chunks(self):
        corpus = LiveCorpus([Chunk(id="c1", doc_id="d", content="python programming language")])
        results = await LiveRetriever(corpus).retrieve("python", top_k=2)
        assert any("python" in r.chunk.content for r in results)

    async def test_sees_chunks_added_after_construction(self):
        corpus = LiveCorpus([Chunk(id="c1", doc_id="d", content="python programming")])
        ret = LiveRetriever(corpus)
        # First retrieval builds the delegate over the seed.
        await ret.retrieve("python")
        # Add a new chunk after construction -> dirty -> next retrieve rebuilds.
        corpus.add_chunks([Chunk(id="c2", doc_id="d", content="asyncio concurrency model")])
        results = await ret.retrieve("asyncio", top_k=2)
        assert any("asyncio" in r.chunk.content for r in results)

    async def test_empty_corpus_returns_empty(self):
        assert await LiveRetriever(LiveCorpus([])).retrieve("anything", top_k=2) == []

    async def test_reuses_delegate_when_not_dirty(self):
        corpus = LiveCorpus([Chunk(id="c1", doc_id="d", content="python")])
        ret = LiveRetriever(corpus)
        await ret.retrieve("python")
        delegate_before = ret._delegate
        await ret.retrieve("python")  # not dirty -> reuse
        assert ret._delegate is delegate_before
