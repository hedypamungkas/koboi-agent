"""koboi/websearch/types.py -- result dataclasses for search/fetch providers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """One hit from a search provider."""

    title: str
    url: str
    snippet: str = ""
    # Provider-specific relevance, when available (Brave score, Firecrawl rank, ...).
    # None when the provider does not surface one; not comparable across providers.
    score: float | None = None
    # Untouched provider payload for debug/eval (never surfaced to the LLM verbatim).
    raw: dict = field(default_factory=dict)


@dataclass
class FetchResult:
    """Cleaned content returned by a fetch provider."""

    url: str
    # Extracted main content (markdown or plain text -- boilerplate removed).
    content: str
    title: str = ""
    # markdown | text | html | pdf -- what the provider extracted.
    content_type: str = ""
    status: int = 200
    truncated: bool = False
    # Free-form provider metadata (page title, lang, word_count, published date, ...).
    metadata: dict = field(default_factory=dict)
