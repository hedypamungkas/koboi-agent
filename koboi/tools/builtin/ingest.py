"""koboi/tools/builtin/ingest.py -- ``ingest_url`` tool (W3): fetch a URL into the live corpus.

Fetches via the injected fetch provider (Wave 1), chunks the content with the RAG chunker, and
appends the chunks to the injected ``live_corpus`` (a ``koboi.rag.live.LiveCorpus`` the agent's
``LiveRetriever`` reads from). Enables mid-conversation knowledge growth: ingest a URL, then
recall it next turn. Opt-in -- requires ``rag.live: true`` (the facade wires the LiveCorpus +
swaps the retriever) and ``ingest_url`` in ``tools.builtin``.
"""

from __future__ import annotations

from urllib.parse import urlparse

from koboi.tools.registry import tool
from koboi.types import RiskLevel


def _stem_from_url(url: str) -> str:
    path = urlparse(url).path
    name = path.rsplit("/", 1)[-1] if path else ""
    # Drop a trailing extension for a cleaner doc id; fall back to "document".
    if "." in name:
        name = name.rsplit(".", 1)[0]
    return name or "document"


@tool(
    name="ingest_url",
    group="web",
    description="Fetch a URL and add its content to the live knowledge corpus so it can be "
    "recalled in later turns. Use for sources worth remembering (not one-off lookups).",
    risk_level=RiskLevel.MODERATE,
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "HTTP(S) URL to fetch and ingest, e.g. 'https://example.com/article'",
            },
        },
        "required": ["url"],
    },
    deps=["fetch_provider", "live_corpus"],
)
async def ingest_url(url: str, _deps: dict | None = None) -> str:
    if not url.startswith(("http://", "https://")):
        return "Error: URL must start with http:// or https://"

    deps = _deps or {}
    provider = deps.get("fetch_provider")
    corpus = deps.get("live_corpus")
    if provider is None:
        return "Error: no fetch_provider configured"
    if corpus is None:
        return "Error: no live_corpus configured (enable rag.live to use ingest_url)"

    try:
        result = await provider.fetch(url)
    except Exception as e:  # noqa: BLE001 - boundary: any provider failure becomes an error string
        return f"Error: fetch failed — {e}"
    if result.metadata.get("error"):
        return f"Error: {result.metadata['error']}"
    if not (result.content or "").strip():
        return f"No ingestable content found at {url}"

    # Chunk + stamp the source URL so retrieved passages cite where they came from.
    from koboi.rag.chunker import ParagraphChunker
    from koboi.rag.types import Document

    doc = Document(id=_stem_from_url(url), title=result.title or _stem_from_url(url), content=result.content)
    chunks = ParagraphChunker().chunk(doc)
    for chunk in chunks:
        chunk.metadata["source"] = url

    corpus.add_chunks(chunks)
    return f"Ingested {len(chunks)} chunk(s) from {url} into the live corpus."
