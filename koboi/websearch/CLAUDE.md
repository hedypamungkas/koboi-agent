# koboi/websearch/ -- Web search + fetch provider registries

## What this is
Pluggable backends for the `web_search` / `web_fetch` builtin tools (`koboi/tools/builtin/web.py`).
**"websearch" = external-web data I/O providers -- NOT a web UI.** Two decorator-based registries
(mirroring `koboi/rag/`'s ComponentRegistry pattern): `@register_search_provider` and
`@register_fetch_provider`. Driven by the `websearch:` YAML section; `build_search_provider()` /
`build_fetch_provider()` resolve the configured provider. The deep_research orchestrator wraps each
in a `CountingProvider` to meter calls against a `ResearchBudget`.

## Key files
```
types.py          SearchResult, FetchResult dataclasses
base.py           BaseSearchProvider / BaseFetchProvider ABCs (SSRF mandate via _check_url_ssrf)
registry.py       ProviderRegistry + @register_search_provider/@register_fetch_provider,
                  build_search_provider()/build_fetch_provider(), load_custom_components()
providers/mock.py     Offline search provider (hardcoded SEARCH_INDEX; default, $0)
providers/ddg.py      DuckDuckGo HTML search (fallback; no key)
providers/brave.py    Brave Search API (X-Subscription-Token; search-only)
providers/firecrawl.py Firecrawl /v1/search (+ optional scrape_results fusion) + /v1/scrape (fetch)
providers/readability.py httpx + trafilatura fetch (default "httpx"; [websearch] extra for trafilatura)
providers/counting.py CountingSearchProvider/CountingFetchProvider -- budget-metering wrappers
__init__.py       Re-exports public API; registers built-ins at import
```

## Extension API
One ABC per stage, registered by name into a module-level registry; `build_*_provider()` resolves
from the `websearch.search` / `websearch.fetch` config.

- **Search provider**: subclass `BaseSearchProvider`, implement
  `async search(self, query: str, *, max_results: int = 10) -> list[SearchResult]`.
  Register: `@register_search_provider("name", description="...")`.
- **Fetch provider**: subclass `BaseFetchProvider`, implement
  `async fetch(self, url: str, *, render: str = "auto", timeout: int = 15) -> FetchResult`.
  Register: `@register_fetch_provider("name", description="...")`.

Register a custom provider two ways: (1) YAML-driven -- list dotted module paths under
`websearch.custom_modules`; `load_custom_components()` imports each so decorators fire on import;
(2) import the module in your `create_app()` / entrypoint before building the agent.

## Conventions
- **SSRF guard**: every fetch provider MUST call `koboi.tools.builtin.web._check_url_ssrf(url)`
  before fetching (defense-in-depth -- blocks internal/private IPs). Built-ins do this via a lazy
  import to avoid a circular dependency (`web.py` imports this package at init).
- **Provider config is nested**: a provider's kwargs live under `websearch.<stage>.<provider_name>`
  (e.g. `websearch.search.brave.api_key`), with shared knobs (`websearch.search.max_results`) as
  fallback. This keeps each provider's credentials isolated.
- **Offline-safe default**: no `websearch:` section -> `mock` search + `httpx` fetch (no API key,
  no network) so configs run in CI/tests.
- **Fail-soft**: unknown provider name -> `mock`/`httpx` fallback (logged).
- **CountingProvider**: deep_research wraps providers so each call charges the `ResearchBudget`
  (`used_searches`/`used_fetches`); a `$0` mock search keeps eval costs at zero.

## Config
```yaml
websearch:
  search:
    provider: firecrawl        # mock | ddg | brave | firecrawl
    max_results: 8
    firecrawl: { api_key: ${FIRECRAWL_API_KEY:}, scrape_results: true }
  fetch:
    provider: firecrawl        # httpx (readability, default) | firecrawl (JS rendering)
    firecrawl: { api_key: ${FIRECRAWL_API_KEY:}, only_main_content: true }
  custom_modules: [mycorp.websearch_providers.bing]
```

## Gotchas
- The `web_search`/`web_fetch` TOOLS keep their names (user-facing LLM tool names); the TOOL GROUP
  is `"web"` (MCP/tool-namespace). Neither changes with this package's `websearch` name.
- `web_fetch` catches provider exceptions + empty-result metadata and returns an error string
  (`koboi/tools/builtin/web.py`) -- a failing/empty fetch never crashes the agent loop or a
  deep_research run (covered by `tests/orchestration/test_deep_research_mechanics.py`).
- Firecrawl is the only provider registered for BOTH search and fetch; Brave/DDG/mock are
  search-only (they need a separate fetch provider).
- See `docs/architecture.md` ("Web Search/Fetch Providers") + `docs/deep-research-smoke.md`.
