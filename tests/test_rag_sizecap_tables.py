"""tests/test_rag_sizecap_tables.py -- size-cap (#OOM guard) + per-document format
override + pdfplumber table-parser registration.

pdfplumber content extraction needs the optional ``[rag]`` extra + a real PDF fixture,
so only registration + the markdown helper are asserted here (the library is exercised
upstream, like pypdf/python-docx).
"""

from __future__ import annotations

import importlib.util

from koboi.rag.parsers import PdfTableParser, dispatch_parser
from koboi.rag.registry import _load_documents, parser_registry


# --------------------------------------------------------------------------- #
# Size cap (OOM guard)
# --------------------------------------------------------------------------- #
def test_size_cap_skips_over_limit(tmp_path):
    f = tmp_path / "big.md"
    f.write_text("x" * 200)
    # max_document_size_mb=0 -> max_bytes=0 -> any non-empty file is over-cap -> skipped
    _, chunks = _load_documents({"enabled": True, "max_document_size_mb": 0, "documents": [{"path": str(f)}]})
    assert chunks == []


def test_size_cap_default_loads_small_file(tmp_path):
    f = tmp_path / "ok.md"
    f.write_text("a small document")
    _, chunks = _load_documents({"enabled": True, "documents": [{"path": str(f)}]})
    assert len(chunks) >= 1  # default 10 MB cap -> loads


def test_size_cap_configurable_threshold(tmp_path):
    f = tmp_path / "mid.md"
    f.write_text("y" * 50)
    # 1 MB cap comfortably admits a 50-byte file
    _, chunks = _load_documents({"enabled": True, "max_document_size_mb": 1, "documents": [{"path": str(f)}]})
    assert len(chunks) >= 1


# --------------------------------------------------------------------------- #
# Per-document format override (table-extraction opt-in path)
# --------------------------------------------------------------------------- #
def test_dispatch_parser_uses_format_hint(monkeypatch):
    # format_hint must drive the registry lookup, not the extension.
    seen: list[str] = []
    real_get = parser_registry.get

    def spy(name):
        seen.append(name)
        return real_get(name)

    monkeypatch.setattr(parser_registry, "get", spy)
    dispatch_parser("report.pdf", b"hello world", format_hint="text")
    assert "text" in seen  # the hint was consulted (not the .pdf extension)


def test_load_documents_format_override(tmp_path):
    f = tmp_path / "data.unknownext"
    f.write_text("plain content here")
    _, chunks = _load_documents({"enabled": True, "documents": [{"path": str(f), "format": "text"}]})
    assert len(chunks) >= 1
    assert chunks[0].metadata.get("source_format") == "text"


# --------------------------------------------------------------------------- #
# PdfTableParser
# --------------------------------------------------------------------------- #
def test_table_to_markdown_helper():
    md = PdfTableParser._table_to_markdown([["Plan", "Price"], ["Pro", "$99"]])
    assert "| Plan | Price |" in md
    assert "| Pro | $99 |" in md
    assert "---" in md  # separator row after the header


def test_pdf_table_parser_registered_iff_pdfplumber():
    has = importlib.util.find_spec("pdfplumber") is not None
    assert ("pdf_table" in parser_registry.list_available()) == has
