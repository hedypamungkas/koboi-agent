"""tests/test_rag_chunk_cov.py -- branch coverage for 4 RAG ingestion modules.

Targets uncovered branches in:
- koboi/rag/chunker.py   (Fixed/Sentence/Paragraph/Semantic + resolve_chunker)
- koboi/rag/parsers.py   (text/html/pdf/docx/pdf_table + dispatch + detect_format)
- koboi/rag/sources.py   (fetch_http/fetch_http_entry/fetch_s3_entry + DocumentCache)
- koboi/rag/registry.py  (_load_documents source dispatch + build_rag error paths)

These are mostly pure-logic modules. Optional deps (pypdf/docx/pdfplumber/boto3)
are absent in the CI-faithful venv, so they are mocked/monkeypatched where needed.
"""

from __future__ import annotations

import copy
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from koboi.rag.chunker import (
    FixedSizeChunker,
    ParagraphChunker,
    SemanticChunker,
    SentenceChunker,
    resolve_chunker,
)
from koboi.rag.parsers import (
    DocxParser,
    HtmlParser,
    PdfParser,
    PdfTableParser,
    TextParser,
    _EXT_TO_FORMAT,
    detect_format,
    dispatch_parser,
)
from koboi.rag.registry import (
    _build_chunker,
    _build_retriever,
    _load_documents,
    augmentation_registry,
    build_rag,
    chunker_registry,
    parser_registry,
    retriever_registry,
)
from koboi.rag.sources import (
    DocumentCache,
    fetch_http,
    fetch_http_entry,
    fetch_s3_entry,
    name_from_url,
    source_key,
)
from koboi.rag.types import Document


# ---------------------------------------------------------------------------
# Shared fixtures: save/restore ALL module-level registries (incl. parser).
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registries():
    """Snapshot and restore every RAG registry so tests mutate only themselves."""
    saved = {
        "chunkers": copy.deepcopy(chunker_registry._entries),
        "retrievers": copy.deepcopy(retriever_registry._entries),
        "augmentations": copy.deepcopy(augmentation_registry._entries),
        "parsers": copy.deepcopy(parser_registry._entries),
    }
    yield
    chunker_registry._entries = saved["chunkers"]
    retriever_registry._entries = saved["retrievers"]
    augmentation_registry._entries = saved["augmentations"]
    parser_registry._entries = saved["parsers"]


def _doc(content: str, doc_id: str = "d1") -> Document:
    return Document(id=doc_id, title=doc_id, content=content)


# ---------------------------------------------------------------------------
# chunker.py
# ---------------------------------------------------------------------------


class TestFixedSizeChunker:
    def test_empty_returns_empty(self):
        assert FixedSizeChunker().chunk(_doc("   ")) == []

    def test_single_chunk_when_under_size(self):
        chunks = FixedSizeChunker(chunk_size=500).chunk(_doc("short text"))
        assert len(chunks) == 1
        assert chunks[0].content == "short text"
        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[0].id == "d1_c0"

    def test_snaps_to_sentence_boundary(self):
        # chunk_size lands mid-token; ". " snap should pull the end back to a period.
        text = "Hello world. Next sentence."
        chunks = FixedSizeChunker(chunk_size=18, overlap=0).chunk(_doc(text))
        # first window "Hello world. Nex" -> snap on ". " -> "Hello world."
        assert chunks[0].content.endswith("world.")

    def test_snaps_to_newline_when_no_period(self):
        # No ". " in window, but a newline exists -> snap to newline.
        text = "alpha bravo\ncharlie delta echo\nfoxtrot"
        chunks = FixedSizeChunker(chunk_size=20, overlap=0).chunk(_doc(text))
        assert len(chunks) >= 2
        # first chunk boundary should be at the newline
        assert "\n" not in chunks[0].content or chunks[0].content.endswith("bravo")

    def test_overlap_advances_start(self):
        # When next_start (end - overlap) > start, start advances by overlap not end.
        text = "a" * 40
        chunks = FixedSizeChunker(chunk_size=10, overlap=4).chunk(_doc(text))
        assert len(chunks) > 1

    def test_make_chunk_strips_content(self):
        chunk = FixedSizeChunker()._make_chunk("doc", 3, "  hi  ")
        assert chunk.content == "hi"
        assert chunk.id == "doc_c3"


class TestSentenceChunker:
    def test_empty_returns_empty(self):
        assert SentenceChunker().chunk(_doc("")) == []

    def test_groups_into_chunks_on_overflow(self):
        # Two sentences whose combined length exceeds a small max_chunk_size.
        s1 = "Sentence one is here."  # 22 chars
        s2 = "Sentence two follows now."  # 25 chars
        text = f"{s1} {s2}"
        chunks = SentenceChunker(max_chunk_size=30).chunk(_doc(text))
        # First sentence fills < 30; adding second exceeds -> flush first, then second.
        assert len(chunks) == 2
        assert s1 in chunks[0].content
        assert s2 in chunks[1].content

    def test_all_sentences_under_limit_one_chunk(self):
        text = "One. Two. Three."
        chunks = SentenceChunker(max_chunk_size=800).chunk(_doc(text))
        assert len(chunks) == 1
        assert "Three" in chunks[0].content


class TestParagraphChunker:
    def test_empty_returns_empty(self):
        assert ParagraphChunker().chunk(_doc("\n\n  ")) == []

    def test_merges_heading_with_following_content(self):
        text = "# Heading\n\nBody paragraph here."
        chunks = ParagraphChunker(max_chunk_size=1000).chunk(_doc(text))
        assert len(chunks) == 1
        assert "# Heading" in chunks[0].content
        assert "Body paragraph" in chunks[0].content

    def test_trailing_heading_appended(self):
        # A heading with no following content is appended as-is.
        text = "Intro paragraph.\n\n## Lonely Heading"
        chunks = ParagraphChunker(max_chunk_size=1000).chunk(_doc(text))
        contents = [c.content for c in chunks]
        assert any("## Lonely Heading" in c for c in contents)

    def test_consecutive_headings_merge(self):
        # Two headings in a row: first heading is flushed when second heading seen.
        text = "# H1\n\n# H2\n\nbody"
        chunks = ParagraphChunker(max_chunk_size=1000).chunk(_doc(text))
        contents = [c.content for c in chunks]
        assert any("# H1" in c for c in contents)
        assert any("# H2" in c and "body" in c for c in contents)

    def test_oversize_paragraph_falls_back_to_fixed(self):
        # Paragraph bigger than max_chunk_size -> FixedSizeChunker fallback.
        big = "x. " * 400  # ~1200 chars, many sentences
        text = big
        chunks = ParagraphChunker(max_chunk_size=100).chunk(_doc(text))
        assert len(chunks) > 1
        # every emitted chunk should carry a rewritten id with _c index
        assert all(c.id.startswith("d1_c") for c in chunks)

    def test_is_heading_helper(self):
        assert ParagraphChunker._is_heading("# Title")
        assert ParagraphChunker._is_heading("###### Deepest")
        assert not ParagraphChunker._is_heading("plain text")
        assert not ParagraphChunker._is_heading("####### too deep")
        assert not ParagraphChunker._is_heading("")


class TestSemanticChunker:
    def test_empty_returns_empty(self):
        assert SemanticChunker().chunk(_doc("")) == []

    def test_single_sentence_returns_one_chunk(self):
        chunks = SemanticChunker().chunk(_doc("Only one sentence."))
        assert len(chunks) == 1

    def test_falls_back_when_no_embeddings(self):
        # Multiple sentences but _get_embeddings_sync always returns None -> fallback.
        text = "First sentence. Second sentence. Third one here."
        chunks = SemanticChunker().chunk(_doc(text))
        # Falls back to SentenceChunker (still produces chunks).
        assert len(chunks) >= 1

    async def test_get_embeddings_sync_returns_none_in_running_loop(self):
        # Inside an async event loop, is_running() is True -> return None (line 180).
        result = await _run_sync_in_loop()
        assert result is None

    def test_get_embeddings_sync_returns_none_sync_context(self):
        # No running loop -> falls into the retriever_registry probe -> None (line 191).
        assert SemanticChunker()._get_embeddings_sync(["a"]) is None

    def test_get_embeddings_sync_probe_reached_when_loop_raises(self, monkeypatch):
        # Force get_event_loop to raise RuntimeError -> except pass -> registry probe
        # executes. With "semantic" absent, line 191 (`entry is None`) returns None.
        import asyncio

        def _raise():
            raise RuntimeError("no current event loop")

        monkeypatch.setattr(asyncio, "get_event_loop", _raise)
        retriever_registry._entries.pop("semantic", None)
        assert SemanticChunker()._get_embeddings_sync(["a"]) is None

    def test_get_embeddings_sync_swallows_registry_exception(self, monkeypatch):
        # retriever_registry.get raises -> except branch returns None (lines 194-195).
        def _boom(_name):
            raise RuntimeError("boom")

        monkeypatch.setattr(retriever_registry, "get", _boom)
        assert SemanticChunker()._get_embeddings_sync(["a"]) is None

    def test_split_by_similarity_splits_on_low_similarity(self):
        chunker = SemanticChunker(similarity_threshold=0.5, max_chunk_size=10000, min_chunk_size=1)
        sentences = ["alpha cat", "bravo dog", "charlie fish"]
        # embeddings: 0->1 orthogonal (sim 0 < 0.5 -> split), 1->2 identical (sim 1.0)
        embeddings = [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        chunks = chunker._split_by_similarity(_doc("alpha cat. bravo dog. charlie fish."), sentences, embeddings)
        # First split happens after sentence 0 (low sim + current_len >= min_chunk_size).
        assert len(chunks) >= 2

    def test_split_by_similarity_respects_min_chunk_size(self):
        # should_split True but current_len < min_chunk_size -> do NOT split yet.
        chunker = SemanticChunker(similarity_threshold=0.5, max_chunk_size=100000, min_chunk_size=1000)
        sentences = ["a", "b"]
        embeddings = [[1.0, 0.0], [0.0, 1.0]]  # sim 0 -> would split
        chunks = chunker._split_by_similarity(_doc("a. b."), sentences, embeddings)
        # No split because current_len (1) < min_chunk_size (1000) -> single chunk.
        assert len(chunks) == 1

    def test_cosine_similarity_zero_norm(self):
        assert SemanticChunker._cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0
        assert SemanticChunker._cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_chunk_uses_split_by_similarity_when_embeddings_available(self, monkeypatch):
        # Force _get_embeddings_sync to return real vectors -> exercises line 169 call site.
        text = "First sentence. Second sentence. Third sentence here."
        embeddings = [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        monkeypatch.setattr(SemanticChunker, "_get_embeddings_sync", lambda self, s: embeddings)
        chunks = SemanticChunker(similarity_threshold=0.5, min_chunk_size=1).chunk(_doc(text))
        assert len(chunks) >= 1


async def _run_sync_in_loop():
    """Call the sync helper from within a running loop (exercises is_running branch)."""
    return SemanticChunker()._get_embeddings_sync(["a"])


class TestResolveChunker:
    def test_resolves_default_paragraph(self):
        c = resolve_chunker({})
        assert isinstance(c, ParagraphChunker)

    def test_resolves_fixed_with_aliases(self):
        c = resolve_chunker({"chunker": "fixed", "chunk_size": 200, "overlap": 10})
        assert isinstance(c, FixedSizeChunker)
        assert c.chunk_size == 200
        assert c.overlap == 10

    def test_unknown_falls_back_to_paragraph(self):
        c = resolve_chunker({"chunker": "no-such-strategy"})
        assert isinstance(c, ParagraphChunker)

    def test_no_chunkers_registered_raises(self):
        chunker_registry.clear()
        with pytest.raises(ValueError, match="No chunkers registered"):
            resolve_chunker({"chunker": "unknown"})


# ---------------------------------------------------------------------------
# parsers.py
# ---------------------------------------------------------------------------


class TestTextParser:
    def test_utf8_text(self):
        text, meta = TextParser().extract("f.txt", b"hello world")
        assert text == "hello world"
        assert meta == {"source_format": "text"}

    def test_binary_returns_empty(self):
        text, meta = TextParser().extract("f.bin", b"\x00\x01\x02PK")
        assert text == ""
        assert meta == {"source_format": "binary"}

    def test_latin1_fallback(self):
        # 0x80 is invalid as utf-8 start but latin-1 decodes it -> never hits line 73.
        text, _ = TextParser().extract("f.txt", b"caf\xe9")
        assert "caf" in text


class TestHtmlParser:
    def test_strips_tags(self):
        html = b"<html><body><p>Hello <b>there</b></p></body></html>"
        text, meta = HtmlParser().extract("f.html", html)
        assert "Hello" in text and "there" in text
        assert meta == {"source_format": "html"}

    def test_feed_exception_falls_back_to_raw(self, monkeypatch):
        # Malformed-feed path (lines 97-99): stripper.feed raises -> raw decode used.
        def _boom(self, _data):
            raise ValueError("bad markup")

        monkeypatch.setattr("koboi.rag.parsers._TagStripper.feed", _boom)
        text, meta = HtmlParser().extract("f.html", b"<p>hi</p>")
        assert "<p>hi</p>" in text
        assert meta == {"source_format": "html"}


class TestPdfParser:
    def test_extract_uses_pdfreader(self, monkeypatch):
        # PdfReader is module-level (None when pypdf absent); patch with a fake.
        fake_page = MagicMock()
        fake_page.extract_text.return_value = "page text"
        fake_reader = MagicMock()
        fake_reader.pages = [fake_page, fake_page]

        import koboi.rag.parsers as parsers_mod

        monkeypatch.setattr(parsers_mod, "PdfReader", lambda stream: fake_reader)
        text, meta = PdfParser().extract("doc.pdf", b"%PDF-1.4 dummy")
        assert "page text" in text
        assert meta["source_format"] == "pdf"
        assert meta["page_count"] == 2


class TestPdfTableParser:
    def test_table_to_markdown_helper(self):
        md = PdfTableParser._table_to_markdown([["Plan", "Price"], ["Pro", "$9|99"]])
        # Pipe in cell is escaped to "/".
        assert "$9/99" in md
        # Separator row inserted after header.
        assert "---" in md

    def test_table_to_markdown_empty(self):
        assert PdfTableParser._table_to_markdown([]) == ""

    def test_extract_with_mocked_pdfplumber(self, monkeypatch):
        fake_page = MagicMock()
        fake_page.extract_text.return_value = "Page body"
        table = [["A", "B"], ["1", "2"]]
        fake_page.extract_tables.return_value = [table]
        fake_pdf = MagicMock()
        fake_pdf.__enter__ = MagicMock(return_value=fake_pdf)
        fake_pdf.__exit__ = MagicMock(return_value=False)
        fake_pdf.pages = [fake_page]

        fake_pdfplumber = MagicMock()
        fake_pdfplumber.open = MagicMock(return_value=fake_pdf)
        monkeypatch.setitem(sys.modules, "pdfplumber", fake_pdfplumber)

        text, meta = PdfTableParser().extract("t.pdf", b"bytes")
        assert "Page body" in text
        assert "| A | B |" in text
        assert meta["source_format"] == "pdf_table"
        assert meta["table_count"] == 1
        assert meta["page_count"] == 1


class TestDocxParser:
    def test_extract_with_mocked_docx(self, monkeypatch):
        # Note: CPython elides the function-level `import docx` and reads the
        # MODULE GLOBAL `docx` (None when python-docx is absent). Patch the global
        # directly rather than injecting into sys.modules.
        para_a = MagicMock()
        para_a.text = "First paragraph"
        para_b = MagicMock()
        para_b.text = "Second paragraph"
        fake_doc = MagicMock()
        fake_doc.paragraphs = [para_a, para_b]

        fake_docx = MagicMock()
        fake_docx.Document = MagicMock(return_value=fake_doc)

        import koboi.rag.parsers as parsers_mod

        monkeypatch.setattr(parsers_mod, "docx", fake_docx)
        text, meta = DocxParser().extract("file.docx", b"PK\x03\x04 stuff")
        assert text == "First paragraph\nSecond paragraph"
        assert meta == {"source_format": "docx"}


class TestDetectFormat:
    def test_extension_dispatch(self):
        assert detect_format("a.txt", b"x") == "text"
        assert detect_format("a.md", b"x") == "text"
        assert detect_format("a.html", b"x") == "html"
        assert detect_format("a.pdf", b"x") == "pdf"

    def test_pdf_magic_bytes(self):
        # Unknown extension but %PDF- magic -> pdf (line 199).
        assert detect_format("weird", b"%PDF-1.4 stuff") == "pdf"

    def test_html_magic_bytes(self):
        assert detect_format("noext", b"<!doctype html>") == "html"
        assert detect_format("noext", b"<html><body></body></html>") == "html"

    def test_defaults_to_text(self):
        assert detect_format("noext", b"just text") == "text"

    def test_ext_map_sanity(self):
        # Ensures _EXT_TO_FORMAT covers expected extensions.
        assert _EXT_TO_FORMAT[".markdown"] == "text"
        assert _EXT_TO_FORMAT[".htm"] == "html"
        assert _EXT_TO_FORMAT[".rst"] == "text"


class TestDispatchParser:
    def test_dispatches_by_format_hint(self):
        text, _ = dispatch_parser("f.unknown", b"hello", format_hint="text")
        assert text == "hello"

    def test_binary_input_to_text_returns_empty(self):
        text, meta = dispatch_parser("f.txt", b"\x00\x01binary")
        assert text == ""
        assert meta == {"source_format": "binary"}

    def test_text_not_registered_falls_back_to_decode(self):
        # Clear parsers -> both fmt and "text" miss -> direct utf-8 decode (line 220).
        parser_registry.clear()
        text, meta = dispatch_parser("f.txt", b"plain bytes")
        assert text == "plain bytes"
        assert meta == {}

    def test_parser_raises_on_binary_returns_empty(self, monkeypatch):
        # Register a parser that raises; dispatch binary data -> binary-parse-error (227).
        class _Boom(BaseParser if False else object):  # noqa: F841
            def extract(self, name, data):
                raise RuntimeError("corrupt")

        parser_registry.register("boomfmt", _Boom)
        text, meta = dispatch_parser("f.bin", b"\x00\x01data", format_hint="boomfmt")
        assert text == ""
        assert meta == {"source_format": "binary-parse-error"}

    def test_parser_raises_on_text_returns_decoded(self):
        class _Boom:
            def extract(self, name, data):
                raise RuntimeError("corrupt")

        parser_registry.register("boomfmt2", _Boom)
        text, meta = dispatch_parser("f.txt", b"plain", format_hint="boomfmt2")
        assert text == "plain"
        assert meta == {}


class TestRegisterBuiltinParsers:
    def test_registers_optional_formats_when_flags_forced(self, monkeypatch):
        # In the CI venv pypdf/docx/pdfplumber are absent, so the *_AVAILABLE flags
        # are False and lines 237/239/241 never run. Force them True and re-run.
        import koboi.rag.parsers as parsers_mod

        monkeypatch.setattr(parsers_mod, "_PYPDF_AVAILABLE", True)
        monkeypatch.setattr(parsers_mod, "_DOCX_AVAILABLE", True)
        monkeypatch.setattr(parsers_mod, "_PDFPLUMBER_AVAILABLE", True)
        parser_registry.clear()
        parsers_mod._register_builtins()
        assert parser_registry.get("pdf") is not None
        assert parser_registry.get("docx") is not None
        assert parser_registry.get("pdf_table") is not None


# ---------------------------------------------------------------------------
# sources.py
# ---------------------------------------------------------------------------


class TestSourceKeyAndName:
    def test_source_key_excludes_secrets(self):
        public = source_key({"source": "http", "url": "u", "token": "secret"})
        clean = source_key({"source": "http", "url": "u"})
        assert public == clean

    def test_source_key_stable(self):
        assert source_key({"a": 1, "b": 2}) == source_key({"b": 2, "a": 1})

    def test_name_from_url(self):
        assert name_from_url("https://x.io/path/file.md") == "file.md"
        assert name_from_url("https://x.io/") == "document"
        assert name_from_url("https://x.io") == "document"


class _FakeResponse:
    def __init__(self, status_code, content=b"", headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeClient:
    """Minimal httpx.Client stand-in: returns queued responses, supports ctx mgr."""

    def __init__(self, responses, responses_by_url=None, exc=None):
        self._responses = list(responses)
        self._by_url = responses_by_url or {}
        self._exc = exc
        self.get_calls = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, headers=None):
        self.get_calls.append(str(url))
        if self._exc is not None:
            raise self._exc
        if str(url) in self._by_url:
            return self._by_url[str(url)]
        return self._responses.pop(0) if self._responses else _FakeResponse(200, b"")


class _FakeURL:
    def __init__(self, url):
        self._url = url

    def join(self, loc):
        # Trivial join: if loc is absolute, use it; else append.
        if loc.startswith("http"):
            return _FakeURL(loc)
        return _FakeURL(self._url.rstrip("/") + "/" + loc.lstrip("/"))

    def __str__(self):
        return self._url


def _install_fake_httpx(monkeypatch, client_factory):
    """Inject a fake httpx module so fetch_http (which does `import httpx`) finds it."""
    fake = MagicMock()
    fake.Client = client_factory
    fake.HTTPError = type("HTTPError", (Exception,), {})
    fake.URL = _FakeURL
    monkeypatch.setitem(sys.modules, "httpx", fake)
    return fake


class TestFetchHttp:
    def test_happy_path_returns_content(self, monkeypatch):
        client = _FakeClient([_FakeResponse(200, b"DATA")])
        _install_fake_httpx(monkeypatch, lambda **kw: client)
        assert fetch_http("https://example.com/doc") == b"DATA"

    def test_retries_on_transient_then_succeeds(self, monkeypatch):
        # 429 (retryable) on attempt 0 -> retry; 200 on attempt 1.
        client = _FakeClient([_FakeResponse(429), _FakeResponse(200, b"OK")])
        _install_fake_httpx(monkeypatch, lambda **kw: client)
        assert fetch_http("https://example.com/doc") == b"OK"
        assert len(client.get_calls) == 2

    def test_transport_error_raised_after_max_retries(self, monkeypatch):
        fake = _install_fake_httpx(monkeypatch, lambda **kw: _FakeClient([], exc=None))
        # Make Client.get always raise HTTPError.
        client = _FakeClient([])
        client.get = lambda url, headers=None: (_ for _ in ()).throw(fake.HTTPError("net"))
        fake.Client = lambda **kw: client
        with pytest.raises(fake.HTTPError):
            fetch_http("https://example.com/doc")

    def test_follows_redirect(self, monkeypatch):
        by_url = {
            "https://example.com/doc": _FakeResponse(302, headers={"location": "https://example.com/final"}),
            "https://example.com/final": _FakeResponse(200, b"FINAL"),
        }
        client = _FakeClient([], responses_by_url=by_url)
        _install_fake_httpx(monkeypatch, lambda **kw: client)
        assert fetch_http("https://example.com/doc") == b"FINAL"

    def test_redirect_without_location_raises(self, monkeypatch):
        by_url = {"https://example.com/doc": _FakeResponse(302, headers={})}
        client = _FakeClient([], responses_by_url=by_url)
        _install_fake_httpx(monkeypatch, lambda **kw: client)
        with pytest.raises(RuntimeError, match="no Location"):
            fetch_http("https://example.com/doc")

    def test_too_many_redirects(self, monkeypatch):
        # Each hop redirects to itself (absolute Location) -> exhausts _MAX_REDIRECTS+1.
        by_url = {"https://example.com/loop": _FakeResponse(302, headers={"location": "https://example.com/loop"})}
        client = _FakeClient([], responses_by_url=by_url)
        _install_fake_httpx(monkeypatch, lambda **kw: client)
        with pytest.raises(RuntimeError, match="too many redirects"):
            fetch_http("https://example.com/loop")

    def test_timeout_clamped(self, monkeypatch):
        captured = {}

        def factory(**kw):
            captured["timeout"] = kw.get("timeout")
            return _FakeClient([_FakeResponse(200, b"X")])

        _install_fake_httpx(monkeypatch, factory)
        fetch_http("https://example.com/doc", timeout=999999)
        assert captured["timeout"] == 30  # MAX_TIMEOUT cap


class TestFetchHttpEntry:
    def test_empty_url_yields_nothing(self):
        assert list(fetch_http_entry({}, None)) == []

    def test_fetch_failure_yields_nothing(self, monkeypatch):
        def _fail(url, **kw):
            raise RuntimeError("network down")

        monkeypatch.setattr("koboi.rag.sources.fetch_http", _fail)
        assert list(fetch_http_entry({"url": "https://example.com/x"}, None)) == []

    def test_cache_hit_short_circuits(self, tmp_path):
        cache = DocumentCache(str(tmp_path))
        cache.put("k", "cached.txt", b"CACHED")
        # Build the same key the entry would build.
        key = source_key({"source": "http", "url": "https://example.com/c"})
        cache.put(key, "cached.txt", b"CACHED")
        out = list(fetch_http_entry({"url": "https://example.com/c"}, cache))
        assert out == [("cached.txt", b"CACHED")]

    def test_cache_write_failure_still_yields(self, monkeypatch, tmp_path):
        cache = DocumentCache(str(tmp_path))

        def _boom(*a, **kw):
            raise OSError("disk full")

        monkeypatch.setattr(cache, "put", _boom)
        monkeypatch.setattr("koboi.rag.sources.fetch_http", lambda url, **kw: b"DATA")
        out = list(fetch_http_entry({"url": "https://example.com/y"}, cache))
        assert out == [("y", b"DATA")]


class TestFetchS3Entry:
    def test_boto3_absent_yields_nothing(self):
        # boto3 is missing in the CI venv -> ImportError path -> no objects.
        assert list(fetch_s3_entry({"bucket": "b", "key": "k/"}, None)) == []

    def test_missing_bucket_warns_and_yields_nothing(self, monkeypatch):
        _install_fake_boto3(monkeypatch, client_factory=lambda *a, **kw: MagicMock())
        assert list(fetch_s3_entry({}, None)) == []

    def test_lists_and_downloads_objects(self, monkeypatch, tmp_path):
        cache = DocumentCache(str(tmp_path))
        fake_client = _make_fake_s3_client(
            pages=[{"Contents": [{"Key": "prefix/file1.txt"}, {"Key": "prefix/sub/"}]}],
            objects={"prefix/file1.txt": b"CONTENT1"},
        )
        _install_fake_boto3(monkeypatch, client_factory=lambda *a, **kw: fake_client)
        out = list(fetch_s3_entry({"bucket": "b", "key": "prefix/"}, cache))
        assert out == [("file1.txt", b"CONTENT1")]

    def test_cache_hit_skips_download(self, monkeypatch, tmp_path):
        cache = DocumentCache(str(tmp_path))
        fake_client = _make_fake_s3_client(
            pages=[{"Contents": [{"Key": "p/a.txt"}]}],
            objects={"p/a.txt": b"FRESH"},
        )
        _install_fake_boto3(monkeypatch, client_factory=lambda *a, **kw: fake_client)
        # Pre-seed cache with the exact key fetch_s3_entry computes.
        key = source_key({"source": "s3", "bucket": "b", "key": "p/a.txt", "endpoint_url": "", "region": "auto"})
        cache.put(key, "a.txt", b"CACHED")
        out = list(fetch_s3_entry({"bucket": "b", "key": "p/"}, cache))
        assert out == [("a.txt", b"CACHED")]
        # get_object should not have been called.
        fake_client.get_object.assert_not_called()

    def test_get_object_error_skips_object(self, monkeypatch):
        fake_client = _make_fake_s3_client(
            pages=[{"Contents": [{"Key": "p/bad"}, {"Key": "p/good"}]}],
            objects={"p/good": b"OK"},
            failing_keys={"p/bad"},
        )
        _install_fake_boto3(monkeypatch, client_factory=lambda *a, **kw: fake_client)
        out = list(fetch_s3_entry({"bucket": "b", "key": "p/"}, None))
        assert out == [("good", b"OK")]

    def test_no_objects_found_warns(self, monkeypatch):
        fake_client = _make_fake_s3_client(pages=[{"Contents": []}], objects={})
        _install_fake_boto3(monkeypatch, client_factory=lambda *a, **kw: fake_client)
        assert list(fetch_s3_entry({"bucket": "b", "key": "p/"}, None)) == []

    def test_outer_exception_yields_nothing(self, monkeypatch):
        def _factory(*a, **kw):
            raise RuntimeError("credentials error")

        _install_fake_boto3(monkeypatch, client_factory=_factory)
        assert list(fetch_s3_entry({"bucket": "b", "key": "p/"}, None)) == []

    def test_cache_write_failure_still_yields(self, monkeypatch, tmp_path):
        cache = DocumentCache(str(tmp_path))

        def _boom(*a, **kw):
            raise OSError("disk full")

        monkeypatch.setattr(cache, "put", _boom)
        fake_client = _make_fake_s3_client(pages=[{"Contents": [{"Key": "p/a.txt"}]}], objects={"p/a.txt": b"OK"})
        _install_fake_boto3(monkeypatch, client_factory=lambda *a, **kw: fake_client)
        out = list(fetch_s3_entry({"bucket": "b", "key": "p/"}, cache))
        assert out == [("a.txt", b"OK")]


def _make_fake_s3_client(pages, objects, failing_keys=None):
    failing_keys = failing_keys or set()
    client = MagicMock()
    paginator = MagicMock()
    paginator.paginate = MagicMock(return_value=pages)
    client.get_paginator = MagicMock(return_value=paginator)

    def _get_object(Bucket, Key):
        if Key in failing_keys:
            raise RuntimeError("object fetch failed")
        return {"Body": BytesIO(objects.get(Key, b""))}

    client.get_object = MagicMock(side_effect=_get_object)
    return client


def _install_fake_boto3(monkeypatch, client_factory):
    fake_boto3 = MagicMock()
    fake_boto3.client = MagicMock(side_effect=client_factory)
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    return fake_boto3


class TestDocumentCache:
    def test_put_and_get_roundtrip(self, tmp_path):
        cache = DocumentCache(str(tmp_path))
        assert cache.get("missing") is None
        cache.put("k1", "name.txt", b"DATA")
        assert cache.get("k1") == ("name.txt", b"DATA")

    def test_atomic_write_failure_cleans_temp_and_reraises(self, monkeypatch, tmp_path):
        # Force Path.replace to raise -> except branch unlinks temp + re-raises.
        cache = DocumentCache(str(tmp_path))

        def _bad_replace(self, target):
            raise OSError("cross-device")

        monkeypatch.setattr(Path, "replace", _bad_replace)
        with pytest.raises(OSError, match="cross-device"):
            cache.put("k2", "n.txt", b"x")
        # No leftover .tmp file written for the data path.
        leftovers = list(tmp_path.glob("*.tmp"))
        assert leftovers == []


# ---------------------------------------------------------------------------
# registry.py (_load_documents source dispatch + build_rag error paths)
# ---------------------------------------------------------------------------


class TestBuildChunkerRetrieverErrors:
    def test_build_chunker_raises_when_no_chunkers(self):
        chunker_registry.clear()
        with pytest.raises(ValueError, match="No chunkers registered"):
            _build_chunker({})

    def test_build_retriever_raises_when_no_retrievers(self):
        retriever_registry.clear()
        with pytest.raises(ValueError, match="No retrievers registered"):
            _build_retriever([], {})


class TestLoadDocumentsSources:
    def test_glob_no_match_warns(self, tmp_path):
        from koboi.rag.chunker import _register_builtins as reg_c

        reg_c()  # ensure paragraph chunker present
        rag_conf = {"documents": [str(tmp_path / "*.nope")]}
        chunker, chunks = _load_documents(rag_conf)
        assert chunks == []

    def test_empty_directory_warns(self, tmp_path):
        from koboi.rag.chunker import _register_builtins as reg_c

        reg_c()
        rag_conf = {"documents": [str(tmp_path)]}
        _, chunks = _load_documents(rag_conf)
        assert chunks == []

    def test_str_entry_unreadable_file_skipped(self, tmp_path):
        from koboi.rag.chunker import _register_builtins as reg_c

        reg_c()
        f = tmp_path / "doc.md"
        f.write_text("readable")
        rag_conf = {"documents": [str(f)]}

        def _boom(self):
            raise OSError("permission denied")

        monkeypatch_target = Path.read_bytes
        # Patch via monkeypatch on the class method used in _resolve_entry.

        original = Path.read_bytes

        def _patched(self):
            if self.name == "doc.md":
                raise OSError("permission denied")
            return original(self)

        # registry imports PathlibPath as local `Path as PathlibPath` inside func,
        # but it is the same pathlib.Path class, so patching the class works.
        import pathlib

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(pathlib.Path, "read_bytes", _patched)
        try:
            _, chunks = _load_documents(rag_conf)
            assert chunks == []
        finally:
            monkeypatch.undo()
            assert monkeypatch_target is Path.read_bytes  # sanity

    def test_non_dict_non_str_entry_skipped(self, tmp_path):
        from koboi.rag.chunker import _register_builtins as reg_c

        reg_c()
        rag_conf = {"documents": [42]}  # int -> not str, not dict -> skipped
        _, chunks = _load_documents(rag_conf)
        assert chunks == []

    def test_file_source_unreadable_skipped(self, tmp_path):
        from koboi.rag.chunker import _register_builtins as reg_c

        reg_c()
        f = tmp_path / "doc.md"
        f.write_text("readable")
        rag_conf = {"documents": [{"source": "file", "path": str(f)}]}

        import pathlib

        original = pathlib.Path.read_bytes

        def _patched(self):
            if self.name == "doc.md":
                raise OSError("io error")
            return original(self)

        mp = pytest.MonkeyPatch()
        mp.setattr(pathlib.Path, "read_bytes", _patched)
        try:
            _, chunks = _load_documents(rag_conf)
            assert chunks == []
        finally:
            mp.undo()

    def test_unknown_source_skipped(self, tmp_path):
        from koboi.rag.chunker import _register_builtins as reg_c

        reg_c()
        rag_conf = {"documents": [{"source": "ftp", "path": "x"}]}
        _, chunks = _load_documents(rag_conf)
        assert chunks == []

    def test_loads_text_file_into_chunks(self, tmp_path):
        from koboi.rag.chunker import _register_builtins as reg_c

        reg_c()
        f = tmp_path / "doc.md"
        f.write_text("# Title\n\nSome body text here.")
        rag_conf = {"documents": [str(f)]}
        chunker, chunks = _load_documents(rag_conf)
        assert len(chunks) >= 1
        assert chunks[0].metadata["source"] == "doc"
        assert chunks[0].metadata["source_format"] == "text"

    def test_max_document_size_skips_oversize(self, tmp_path):
        from koboi.rag.chunker import _register_builtins as reg_c

        reg_c()
        f = tmp_path / "big.txt"
        f.write_text("x" * 50)
        rag_conf = {"documents": [str(f)], "max_document_size_mb": 0}  # 0 MB cap
        _, chunks = _load_documents(rag_conf)
        assert chunks == []


class TestBuildRagErrorPaths:
    def test_no_augmentation_registered_raises(self, tmp_path):
        from koboi.rag.chunker import _register_builtins as reg_c
        from koboi.rag.retriever import _register_builtins as reg_r

        reg_c()
        reg_r()
        f = tmp_path / "doc.md"
        f.write_text("Some text to load.")
        augmentation_registry.clear()
        with pytest.raises(ValueError, match="No augmentation strategies registered"):
            build_rag({"enabled": True, "documents": [str(f)]})

    def test_disabled_returns_none(self):
        assert build_rag({"enabled": False}) is None
        assert build_rag({}) is None

    def test_no_documents_returns_none(self):
        from koboi.rag.chunker import _register_builtins as reg_c

        reg_c()
        assert build_rag({"enabled": True, "documents": []}) is None


class TestResolveKwargsExtras:
    def test_resolve_kwargs_passes_unknown_rag_keys_to_augmentation(self, tmp_path):
        # Line 495: for param_name in entry.parameters: if ... param_name in rag_conf.
        # Build a real pipeline and thread an extra config key that the augmentation
        # accepts as a constructor param.
        from koboi.rag.augmentation import InMemoryAugmentation
        from koboi.rag.chunker import _register_builtins as reg_c
        from koboi.rag.retriever import _register_builtins as reg_r

        reg_c()
        reg_r()
        augmentation_registry.clear()
        augmentation_registry.register("in_memory", InMemoryAugmentation)
        f = tmp_path / "doc.md"
        f.write_text("Some real content to embed.")
        rag_conf = {
            "enabled": True,
            "documents": [str(f)],
            "relevance_threshold": 0.25,  # an extra param on InMemoryAugmentation
        }
        aug = build_rag(rag_conf)
        assert aug is not None
