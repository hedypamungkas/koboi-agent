"""koboi/websearch/providers/mock.py -- offline hardcoded search index (default provider).

Migrated verbatim from ``koboi/tools/builtin/web.py`` (SEARCH_INDEX + _format_results +
_search_mock) so the suite stays offline; the legacy module re-exports these names for
back-compat. ``MockSearchProvider`` wraps the same matching logic for the registry path.
"""

from __future__ import annotations

from koboi.websearch.base import BaseSearchProvider
from koboi.websearch.registry import register_search_provider
from koboi.websearch.types import SearchResult

# Hardcoded offline index (kept identical to the pre-refactor web.py contents).
SEARCH_INDEX: dict[str, list[dict]] = {
    "python": [
        {
            "title": "Python Documentation",
            "url": "https://docs.python.org/3/",
            "snippet": "Official Python 3 documentation — tutorials, library reference, and guides.",
        },
        {
            "title": "Real Python",
            "url": "https://realpython.com/",
            "snippet": "Python tutorials, guides, and best practices for developers.",
        },
    ],
    "asyncio": [
        {
            "title": "Async IO in Python",
            "url": "https://docs.python.org/3/library/asyncio.html",
            "snippet": "Coroutines, event loops, tasks, and futures for async programming.",
        },
    ],
    "react": [
        {
            "title": "React Documentation",
            "url": "https://react.dev/",
            "snippet": "Learn React — components, hooks, state management.",
        },
    ],
    "typescript": [
        {
            "title": "TypeScript Handbook",
            "url": "https://www.typescriptlang.org/docs/handbook/",
            "snippet": "TypeScript type system, interfaces, generics, and modules.",
        },
    ],
    "fastapi": [
        {
            "title": "FastAPI Documentation",
            "url": "https://fastapi.tiangolo.com/",
            "snippet": "Modern Python web framework with automatic OpenAPI docs.",
        },
    ],
    "docker": [
        {
            "title": "Docker Documentation",
            "url": "https://docs.docker.com/",
            "snippet": "Container platform — build, ship, and run applications.",
        },
    ],
    "git": [
        {
            "title": "Git Documentation",
            "url": "https://git-scm.com/doc",
            "snippet": "Version control system — branching, merging, and collaboration.",
        },
    ],
    "ai": [
        {
            "title": "Anthropic API Docs",
            "url": "https://docs.anthropic.com/",
            "snippet": "Claude API — messages, tool use, and streaming.",
        },
        {
            "title": "OpenAI API Docs",
            "url": "https://platform.openai.com/docs",
            "snippet": "GPT models, embeddings, and fine-tuning.",
        },
    ],
    "agent": [
        {
            "title": "LangChain Docs",
            "url": "https://python.langchain.com/",
            "snippet": "Framework for building LLM-powered applications and agents.",
        },
        {"title": "CrewAI", "url": "https://docs.crewai.com/", "snippet": "Multi-agent orchestration framework."},
    ],
    "mcp": [
        {
            "title": "Model Context Protocol",
            "url": "https://modelcontextprotocol.io/",
            "snippet": "Open standard for connecting AI assistants to external tools and data.",
        },
    ],
}


def _match_mock_index(query: str) -> list[dict]:
    """Token-overlap match against ``SEARCH_INDEX``. Returns raw entry dicts."""
    q = query.lower().strip()
    query_tokens = set(q.split())
    results: list[dict] = []
    for key, entries in SEARCH_INDEX.items():
        key_tokens = set(key.split())
        if query_tokens & key_tokens:
            results.extend(entries)
    return results


def _format_results(query: str, results: list[dict]) -> str:
    """Format raw entry dicts as the legacy ``web_search`` output string."""
    lines = [f"Search results for '{query}':"]
    seen: set[str] = set()
    for r in results:
        if r["url"] not in seen:
            seen.add(r["url"])
            lines.append(f"  - {r['title']}: {r['url']}")
            lines.append(f"    {r['snippet']}")
    return "\n".join(lines)


def _search_mock(query: str) -> str:
    """Legacy offline search (string output). Kept for back-compat + offline tests."""
    results = _match_mock_index(query)
    if not results:
        available = ", ".join(sorted(SEARCH_INDEX.keys()))
        return f"No results found for '{query}'. Available topics: {available}"
    return _format_results(query, results)


@register_search_provider("mock", description="Hardcoded offline index (default; no network)")
class MockSearchProvider(BaseSearchProvider):
    """Registry-backed offline search over ``SEARCH_INDEX``."""

    def __init__(self, max_results: int = 10) -> None:
        self._max_results = max_results

    async def search(self, query: str, *, max_results: int = 10) -> list[SearchResult]:
        limit = max_results or self._max_results
        return [
            SearchResult(title=e["title"], url=e["url"], snippet=e.get("snippet", ""))
            for e in _match_mock_index(query)[:limit]
        ]
