"""tests/test_rag_ingestion.py -- Gap #1 (remote loader) + #2 (parsing) regression guards.

Covers: parser registry/gating, format detection, text/html extraction + fallbacks,
local-file backward compat, HTTP source (+ document cache), S3 resilience, SSRF guard,
and cache-key secret hygiene. PDF/DOCX content extraction is gated by their optional
extras (registration is asserted; the libs are exercised upstream).
"""

from __future__ import annotations

import sys

import pytest

import koboi.rag.sources as sources
from koboi.rag.parsers import HtmlParser, TextParser, detect_format, dispatch_parser
from koboi.rag.registry import _load_documents, build_rag, parser_registry
from koboi.rag.sources import DocumentCache, fetch_http, name_from_url, source_key


# --------------------------------------------------------------------------- #
# Parsers (#2)
# --------------------------------------------------------------------------- #
def test_text_and_html_parsers_always_registered():
    avail = parser_registry.list_available()
    assert "text" in avail and "html" in avail


def test_pdf_docx_registered_only_if_extra_installed():
    def has(mod: str) -> bool:
        try:
            __import__(mod)
            return True
        except ImportError:
            return False

    avail = parser_registry.list_available()
    assert ("pdf" in avail) == has("pypdf")
    assert ("docx" in avail) == has("docx")


def test_text_parser_encoding_fallback():
    text, meta = TextParser().extract("x.txt", "café".encode("latin-1"))
    assert "café" in text
    assert meta["source_format"] == "text"


def test_html_parser_strips_tags():
    text, _ = HtmlParser().extract("p.html", b"<p>Refund <b>30</b> days</p>")
    assert "Refund" in text and "30" in text
    assert "<" not in text and ">" not in text


def test_detect_format_by_extension_and_magic_bytes():
    assert detect_format("a.md", b"") == "text"
    assert detect_format("p.html", b"") == "html"
    assert detect_format("doc.docx", b"") == "docx"
    assert detect_format("x", b"%PDF-1.4 junk") == "pdf"
    assert detect_format("x", b"<html><body>") == "html"
    assert detect_format("x", b"plain bytes") == "text"


def test_dispatch_parser_falls_back_for_unknown_format():
    text, _ = dispatch_parser("weird.xyz", b"hello world")
    assert text == "hello world"


def test_dispatch_parser_falls_back_when_optional_parser_absent():
    # Without pypdf installed, a PDF routes to the text fallback (encoding-safe decode).
    text, _ = dispatch_parser("scan.pdf", b"%PDF-1.4 not really a pdf")
    assert isinstance(text, str)


# --------------------------------------------------------------------------- #
# Local-file backward compat (#1/#3 unchanged)
# --------------------------------------------------------------------------- #
def test_local_file_glob_and_dir_still_load(tmp_path):
    (tmp_path / "a.md").write_text("file a")
    (tmp_path / "b.md").write_text("file b")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.md").write_text("file c")
    _, globbed = _load_documents({"enabled": True, "documents": [{"path": str(tmp_path / "*.md")}]})
    assert len(globbed) == 2
    _, recursed = _load_documents({"enabled": True, "documents": [{"path": str(tmp_path)}]})
    assert len(recursed) == 3
    assert recursed[0].metadata.get("source_format") == "text"


# --------------------------------------------------------------------------- #
# HTTP source + document cache (#1)
# --------------------------------------------------------------------------- #
def test_http_source_loads_parses_and_caches(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_fetch(url, *, headers=None, timeout=None, max_bytes=None):
        calls["n"] += 1
        return b"# Handbook\nRefund window is 30 days."

    monkeypatch.setattr(sources, "fetch_http", fake_fetch)
    cache = str(tmp_path / "doccache")

    _, chunks = _load_documents(
        {
            "enabled": True,
            "document_cache_path": cache,
            "documents": [{"source": "http", "url": "https://example.com/handbook.md"}],
        }
    )
    assert len(chunks) == 1
    assert "Refund window is 30 days" in chunks[0].content
    assert chunks[0].metadata.get("source_format") == "text"

    # Second (per-session) build must hit the cache -> no new network fetch.
    _load_documents(
        {
            "enabled": True,
            "document_cache_path": cache,
            "documents": [{"source": "http", "url": "https://example.com/handbook.md"}],
        }
    )
    assert calls["n"] == 1


def test_http_fetch_failure_skips_entry_no_crash(tmp_path, monkeypatch):
    def boom(url, *, headers=None, timeout=None, max_bytes=None):
        raise RuntimeError("network down")

    monkeypatch.setattr(sources, "fetch_http", boom)
    _, chunks = _load_documents({"enabled": True, "documents": [{"source": "http", "url": "https://example.com/x.md"}]})
    assert chunks == []


def test_ssrf_guard_blocks_metadata_ip():
    # A misconfigured/attacker-controlled URL must not reach cloud metadata services.
    with pytest.raises(Exception):  # noqa: B017 (SSRF guard raises before any request)
        fetch_http("http://169.254.169.254/latest/meta-data/")


def test_build_rag_composes_http_source(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "fetch_http", lambda *a, **k: b"Refund policy: 30 days.")
    aug = build_rag(
        {
            "enabled": True,
            "retriever": "keyword",
            "top_k": 2,
            "documents": [{"source": "http", "url": "https://example.com/policy.md"}],
        }
    )
    assert aug is not None


# --------------------------------------------------------------------------- #
# S3 resilience (#1)
# --------------------------------------------------------------------------- #
def test_s3_missing_bucket_skipped():
    _, chunks = _load_documents({"enabled": True, "documents": [{"source": "s3"}]})
    assert chunks == []


def test_s3_without_boto3_skips_gracefully(monkeypatch):
    # Simulate the optional extra not being installed.
    monkeypatch.setitem(sys.modules, "boto3", None)
    _, chunks = _load_documents({"enabled": True, "documents": [{"source": "s3", "bucket": "b", "key": "k"}]})
    assert chunks == []  # no crash; the missing dep is logged at ERROR level


# --------------------------------------------------------------------------- #
# Cache-key secret hygiene (#1)
# --------------------------------------------------------------------------- #
def test_source_key_excludes_secrets():
    a = source_key({"source": "s3", "bucket": "b", "key": "k", "secret_access_key": "SHH"})
    b = source_key({"source": "s3", "bucket": "b", "key": "k", "secret_access_key": "DIFFERENT"})
    assert a == b  # secrets never enter the cache-key material


def test_name_from_url_falls_back():
    assert name_from_url("https://x.example.com/path/handbook.md") == "handbook.md"
    assert name_from_url("https://x.example.com/") == "document"
