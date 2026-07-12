"""tests/test_rag_indo_nlp.py -- Indonesian stopwords + Sastrawi stemmer (opt-in, [indo-nlp] extra).

Stopword tests are stdlib-only (always run). Stemmer tests use the real Sastrawi when the
[indo-nlp] extra is installed (skip otherwise, like the tiktoken/[tokenizer] tests). The
stemmer fallback path (extra absent -> None + warning) is always tested via sys.modules blocking.
"""

from __future__ import annotations

import sys

import pytest

from koboi.rag.retriever import BM25Retriever, KeywordRetriever, _normalize_stemmer, _normalize_stopwords
from koboi.rag.types import Chunk

_HAS_SASTRAWI = True
try:
    import Sastrawi  # noqa: F401
except ImportError:
    _HAS_SASTRAWI = False


class TestStopwordLanguage:
    def test_id_set(self):
        s = _normalize_stopwords("id")
        assert s is not None and len(s) > 50
        assert "yang" in s and "dan" in s and "untuk" in s  # signature ID function words

    def test_en_set_and_backcompat_true(self):
        en = _normalize_stopwords("en")
        true_ = _normalize_stopwords(True)
        assert en == true_  # True = English (back-compat)
        assert "the" in en and "yang" not in en

    def test_none_and_false_off(self):
        assert _normalize_stopwords(None) is None
        assert _normalize_stopwords(False) is None

    def test_custom_set(self):
        assert _normalize_stopwords({"foo", "bar"}) == {"foo", "bar"}


class TestStemmerFallback:
    def test_id_without_extra_returns_none(self, monkeypatch):
        # Force the import to fail even if Sastrawi is installed in this venv.
        monkeypatch.setitem(sys.modules, "Sastrawi", None)
        monkeypatch.setitem(sys.modules, "Sastrawi.Stemmer", None)
        monkeypatch.setitem(sys.modules, "Sastrawi.Stemmer.StemmerFactory", None)
        assert _normalize_stemmer("id") is None  # graceful fallback, retrieval never breaks

    def test_none_and_false_off(self):
        assert _normalize_stemmer(None) is None
        assert _normalize_stemmer(False) is None

    def test_unknown_lang_returns_none(self):
        assert _normalize_stemmer("fr") is None


class TestStemmerID:
    pytestmark = pytest.mark.skipif(not _HAS_SASTRAWI, reason="needs [indo-nlp] extra (Sastrawi)")

    def test_stems_inflected_forms_to_root(self):
        st = _normalize_stemmer("id")
        assert st is not None
        # morphology: meN- / ber- / -kan / -i / -an / per-an / pe-an
        assert st("makanan") == "makan"
        assert st("memakan") == "makan"  # makanan + memakan -> same root -> they MATCH
        assert st("berjalan") == "jalan"
        assert st("pendidikan") == "didik"

    def test_bm25_stemmer_matches_inflected_query_to_doc(self):
        # Query uses an inflected form; doc uses the root -- they must match with stemming ON.
        chunks = [
            Chunk(id="root", doc_id="d", content="Karyawan makan di kantin."),
            Chunk(id="distract", doc_id="d", content="Cuaca hari ini cerah."),
        ]
        bm = BM25Retriever(chunks=chunks, stemmer="id")
        import asyncio

        res = asyncio.run(bm.retrieve("tempat memakan makanan", top_k=1))
        assert res and res[0].chunk.id == "root"  # "memakan"/"makanan" -> "makan" matches "makan"

    def test_default_off_preserves_behavior(self):
        # No stemmer -> inflected query does NOT stem -> weaker match (regression guard).
        chunks = [Chunk(id="root", doc_id="d", content="Karyawan makan di kantin.")]
        bm_off = BM25Retriever(chunks=chunks)  # stemmer=None (default)
        assert bm_off._stemmer is None

    def test_keyword_retriever_stemmer_wired(self):
        chunks = [Chunk(id="r", doc_id="d", content="bekerja sama")]
        kw = KeywordRetriever(chunks=chunks, stemmer="id")
        assert kw._stemmer is not None
        # token "bekerja" -> "kerja" in the index
        toks = kw._tokenize("bekerja")
        assert "kerja" in toks

    def test_stopwords_plus_stemmer_end_to_end(self):
        # ID query with function words + inflected form retrieves the right chunk.
        chunks = [
            Chunk(id="c", doc_id="d", content="Kebijakan cuti tahunan untuk karyawan adalah 12 hari."),
            Chunk(id="x", doc_id="d", content="Pembayaran gaji dilakukan bulanan."),
        ]
        bm = BM25Retriever(chunks=chunks, stopwords="id", stemmer="id")
        import asyncio

        res = asyncio.run(bm.retrieve("berapa hari cuti tahunan karyawan?", top_k=1))
        assert res and res[0].chunk.id == "c"
