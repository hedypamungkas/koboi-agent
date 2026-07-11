"""tests/test_rag_review_fixes.py -- Regression tests for PR #34 review findings.

T1 (C2 fix): dispatch_parser exception on binary data → skip, not mojibake.
T2 (W6 fix): query_rewrite without chat_client → warns.
T3 (strengthened): build_rag HTTP source actually retrieves the fetched content.
Plus: $lt operator (minor gap from test review).
"""

from __future__ import annotations

import logging

import koboi.rag.sources as sources
from koboi.rag.augmentation import OnTheFlyAugmentation
from koboi.rag.filters import matches_filter
from koboi.rag.parsers import BaseParser, dispatch_parser
from koboi.rag.registry import build_rag, parser_registry, register_parser
from koboi.rag.retriever import KeywordRetriever
from koboi.rag.types import Chunk


# ---- T1: C2 fix — corrupt PDF → skip, not mojibake ----
def test_dispatch_parser_exception_on_binary_skips():
    """When a parser raises on binary data, the fallback skips (empty), not mojibake."""

    class _CrashParser(BaseParser):
        def extract(self, name, data):
            raise RuntimeError("encrypted/corrupt PDF")

    register_parser("__test_crash")(_CrashParser)
    try:
        binary = b"%PDF-1.4\n\x00\x01\x02 binary junk\xff\xfe"
        text, meta = dispatch_parser("test.pdf", binary, format_hint="__test_crash")
        assert not text or not text.strip()  # empty → will be skipped by _load_documents
        assert meta.get("source_format") == "binary-parse-error"
    finally:
        parser_registry._entries.pop("__test_crash", None)


# ---- T2: W6 fix — query_rewrite without chat_client warns ----
def test_query_rewrite_without_chat_client_warns(caplog):
    chunks = [Chunk(id="c", doc_id="d", content="test", metadata={})]
    with caplog.at_level(logging.WARNING, logger="koboi.rag.augmentation"):
        OnTheFlyAugmentation(
            retriever=KeywordRetriever(chunks),
            top_k=2,
            query_rewrite=True,
            rewrite_client=None,
        )
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("no chat client" in m or "silently disabled" in m for m in msgs), f"expected warning, got {msgs}"


# ---- T3: strengthened — HTTP source actually retrieves ----
async def test_build_rag_http_source_retrieves(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "fetch_http", lambda *a, **k: b"refund window is 30 days")
    aug = build_rag(
        {
            "enabled": True,
            "retriever": "keyword",
            "top_k": 2,
            "documents": [{"source": "http", "url": "https://example.com/policy.md"}],
        }
    )
    assert aug is not None
    result = await aug.augment_for_memory("refund")
    assert "refund window is 30 days" in result  # the fetched content survived into retrieval


# ---- Minor gap: $lt operator ----
def test_lt_operator():
    assert matches_filter({"year": 2020}, {"year": {"$lt": 2021}})
    assert not matches_filter({"year": 2022}, {"year": {"$lt": 2021}})
