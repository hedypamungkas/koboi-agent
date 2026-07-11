"""Tests for koboi/rag/live.py -- LiveCorpus + LiveRetriever (W3) + corpus-file loader (W5)."""

from __future__ import annotations

import json

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


class TestLiveCorpusFile:
    """W5 B2: load a LiveCorpus from a research run's persisted-findings jsonl."""

    def _write_jsonl(self, path, rows):
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    def test_from_corpus_file(self, tmp_path):
        path = tmp_path / "findings.jsonl"
        self._write_jsonl(
            path,
            [{"citation_id": 1, "node_id": "nA", "text": "alpha"}, {"citation_id": 2, "node_id": "nB", "text": "beta"}],
        )
        corpus = LiveCorpus.from_corpus_file(str(path))
        assert corpus is not None
        assert len(corpus.chunks) == 2
        assert corpus.chunks[0].content == "alpha"
        assert corpus.chunks[0].metadata["source"] == "nA"
        assert corpus.chunks[0].metadata["citation_id"] == 1
        assert corpus.dirty is True  # seed needs an initial delegate build

    def test_missing_file_returns_none(self, tmp_path):
        assert LiveCorpus.from_corpus_file(str(tmp_path / "nope.jsonl")) is None

    def test_empty_file_returns_none(self, tmp_path):
        (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")
        assert LiveCorpus.from_corpus_file(str(tmp_path / "empty.jsonl")) is None

    def test_malformed_rows_skipped(self, tmp_path):
        path = tmp_path / "mixed.jsonl"
        path.write_text(
            "not json\n" + json.dumps({"citation_id": 1, "node_id": "nA", "text": "alpha"}) + "\n", encoding="utf-8"
        )
        corpus = LiveCorpus.from_corpus_file(str(path))
        assert corpus is not None and len(corpus.chunks) == 1  # the malformed row skipped
