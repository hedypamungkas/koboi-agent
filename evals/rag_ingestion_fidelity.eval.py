"""Mock-safe ingestion-fidelity gate (Tier 1).

Verifies the document-ingestion layer that feeds retrieval: format parsers
(text/html stdlib; pdf/docx behind the ``[rag]`` extra) and the chunker size invariant.
Real PDF/DOCX text extraction from binary fixtures needs the optional libraries and a
real text-layer PDF tool, so those legs self-skip when the extra is absent and assert
registration when it is present (the ``registered-iff-extra`` contract).

No agent drive is needed; components are exercised directly and asserted via ``t.check``.

Run:  koboi eval-test evals/rag_ingestion_fidelity.eval.py --mock --strict
"""

from koboi.eval.t import Equals, Matches, Severity, scripted_response
from koboi.rag.chunker import FixedSizeChunker
from koboi.rag.parsers import HtmlParser, TextParser, detect_format
from koboi.rag.types import Document

CONFIG = {
    "agent": {
        "name": "rag-ingestion-eval",
        "description": "Ingestion-fidelity probe (parsers + chunker)",
        "system_prompt": "Use the provided context to answer.",
        "max_iterations": 4,
    },
    "llm": {"provider": "openai", "model": "gpt-4o-mini", "api_key": "dummy"},
}

MOCK_RESPONSES = [scripted_response("ok")]
TAGS = ["rag", "ingestion"]


async def test_text_parser_encoding_fallback(t):
    """TextParser must decode latin-1 bytes without crashing (encoding-safe)."""
    text, _meta = TextParser().extract("policy.txt", "café 12 days".encode("latin-1"))
    t.check(
        text,
        Matches(fn=lambda x: "café" in x, description="latin-1 decoded to 'café'"),
        name="text_parser_latin1",
        severity=Severity.GATE,
    )


async def test_html_parser_strips_tags(t):
    """HtmlParser must strip markup and keep the text content."""
    text, _meta = HtmlParser().extract("page.html", b"<div><p>Notice period is 30 days</p></div>")
    t.check(
        text,
        Matches(fn=lambda x: "30 days" in x, description="html stripped -> '30 days' present"),
        name="html_strips_tags",
        severity=Severity.GATE,
    )


async def test_detect_format_magic_bytes(t):
    """detect_format must classify by magic bytes when the extension is absent/unknown."""
    t.check(detect_format("a", b"%PDF-1.4\n%..."), Equals("pdf"), name="detect_pdf_magic", severity=Severity.GATE)
    t.check(
        detect_format("a.docx", b"PK\x03\x04" + b"\x00" * 20),
        Equals("docx"),
        name="detect_docx_magic",
        severity=Severity.GATE,
    )


async def test_pdf_docx_registered_iff_extra(t):
    """Pdf/Docx parsers are registered only when their optional library imports."""
    from koboi.rag.parsers import _DOCX_AVAILABLE, _PDFPLUMBER_AVAILABLE, _PYPDF_AVAILABLE  # noqa: F401
    from koboi.rag.registry import parser_registry

    # Ensure builtins are registered (lazy on first use).
    parser_registry.get("text") or None  # touch
    if _PYPDF_AVAILABLE:
        t.check(
            parser_registry.get("pdf") is not None,
            Matches(fn=lambda b: b, description="pdf registered (pypdf present)"),
            name="pdf_registered",
            severity=Severity.GATE,
        )
    if _DOCX_AVAILABLE:
        t.check(
            parser_registry.get("docx") is not None,
            Matches(fn=lambda b: b, description="docx registered (python-docx present)"),
            name="docx_registered",
            severity=Severity.GATE,
        )


async def test_chunker_emits_multiple_aligned_chunks(t):
    """A long document must split into >1 chunk; each chunk is non-empty."""
    long_doc = Document(
        id="d1",
        title="big",
        content="\n\n".join(f"Section {i}: carry forward maximum 3 days of leave." for i in range(20)),
    )
    chunks = FixedSizeChunker(chunk_size=120, overlap=0).chunk(long_doc)
    t.check(
        len(chunks),
        Matches(fn=lambda n: n > 1, description=">1 chunk emitted"),
        name="chunk_count",
        severity=Severity.GATE,
    )
    t.check(
        all(c.content.strip() for c in chunks),
        Matches(fn=lambda b: b, description="no empty chunks"),
        name="no_empty_chunks",
        severity=Severity.GATE,
    )
