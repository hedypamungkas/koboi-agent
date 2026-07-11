# Deep Research — Implementation Plan

> **North star:** ship an agent-driven, iterative, **cited web-research** capability on top of
> koboi's existing `execution.mode: dynamic` + DAG + durability. Everything else in this plan
> — fixing `web_search`, improving fetch, runtime RAG ingest — exists **only to serve that goal.**

**Status:** design / not yet implemented · **Verdict:** ✅ highly feasible, fully additive · **Date:** 2026-07-11

---

## TL;DR

- **Deep Research** = the agent plans a research task → searches the web → fetches & extracts
  sources → **assesses coverage** → drills deeper on gaps → synthesizes a **cited** report.
  It is a loop, not a single pass — that loop is what makes it "deep."
- koboi **already has ~70% of the substrate**: `plan_or_skip` (planner.py), DAG waves +
  durability (dag_scheduler.py), per-node `AgentDef.tools_config`, RAG chunker/retriever for
  passage ranking, orchestrator synthesis. The missing ~30% is the **research loop** itself.
- 4 waves. **W2 (Deep Research) is the payload.** W0/W1 are the I/O layers it calls; W3 is the
  persistence layer that lets findings accumulate. All are sketched in detail below.
- Architectural keystone: **deep research and RAG-over-URL share one primitive** —
  `fetch → extract → chunk → rank → cite` — built once, exposed two ways.

### Refined effort plan (deep-research-first)

| Wave | Role toward Deep Research | Scope | Effort | Depends on |
|---|---|---|---|---|
| **W0** | 🔧 I/O enabler — **search** | search-provider **registry** (`@register_search_provider`) + Brave + Firecrawl-search + `web:` config + `web_search` tool refactor | ~2-3 hari | — |
| **W1** | 🔧 I/O enabler — **fetch** | fetch-provider **registry** (`@register_fetch_provider`) + readability (trafilatura) + render escalation + Firecrawl-scrape + RAG `source: http/firecrawl` wiring + better HtmlParser | ~3-4 hari | — *(parallel to W0; shares the `koboi/web/` pkg)* |
| **W2** | 🎯 **THE PAYLOAD — Deep Research** | `execution.mode: deep_research` + `ResearchContext` + **coverage-gated loop** + `SourceStore` + **citation threading** + budget caps + `deep_research_demo.yaml` | ~6-8 hari | W0 + W1 *(can scaffold with mock providers in parallel)* |
| **W3** | 🔧 Persistence enabler | `ingest_url` tool + `LiveRetriever` (mutable corpus) + research-output persistence → "RAG not doc-only" + findings accumulate | ~3-4 hari | W1 + W2 |

**Total ~14-19 hari.** Critical path = `W0 → W2` (or `W1 → W2`). Each wave ships independently
and leaves the system better. W2's non-I/O pieces (SourceStore, CoverageEvaluator,
ResearchContext, budget) can start on day 1 using mock providers, then swap to real
Brave/Firecrawl when W0/W1 land.

---

## 1. What "Deep Research" means here

The canonical research loop (same shape as OpenAI Deep Research, Gemini Deep Research,
Perplexity Pro, GPT Researcher):

```
   ┌──────────────────────────────────────────────────────────┐
   │ 1. PLAN       decompose question → sub-questions +        │
   │               search queries        (plan_or_skip)        │
   │ 2. SEARCH     run queries           (search provider)     │
   │ 3. FETCH      top URLs → clean text (fetch provider)      │
   │ 4. EXTRACT    chunk + rank passages (RAG chunker/retriever)│
   │ 5. ASSESS     coverage sufficient?   (CoverageEvaluator)  │
   │ 6. ITERATE    gaps → follow-up queries → back to 2        │
   │ 7. SYNTHESIZE cited report          (orchestrator + cites)│
   └──────────────────────────────────────────────────────────┘
```

**What koboi already has vs. what's new:**

| Step | koboi today | Gap for deep research |
|---|---|---|
| 1 Plan | ✅ `plan_or_skip` (planner.py:101) emits step graph | planner prompt is generic; needs research-flavored decomposition into sub-questions + queries |
| 2 Search | 🔴 `mock` default + flaky DDG scrape (web.py:23,181) | reliable search (W0) |
| 3 Fetch | 🟡 regex extractor, no readability/JS (web.py:279) | clean extraction (W1) |
| 4 Extract/rank | ✅ RAG chunker + retriever (rag/) | reusable as-is — applied to fresh web content |
| 5 Assess | ❌ none | **CoverageEvaluator** (NEW — W2) |
| 6 Iterate | 🟡 `max_replans=0`, failure-only (orchestrator.py:132) | **coverage-gated deepening** (NEW — W2) |
| 7 Synthesize | ✅ orchestrator shared-client synthesis | needs SourceStore + citation injection (W2) |
| Tool access | 🔴 planned nodes get **no tools** (planner.py:23-28 → AgentDef has only system_prompt) | **research tool bundle on planned nodes** (NEW — W2) |
| Cite | 🟡 `[1][Source]` only inside RAG aug (augmentation.py:95) | generalize across web calls → SourceStore (W2) |
| Budget | 🟡 harness budget + jobs max_iterations exist | per-run research ledger (W2) |
| Resume | ✅ DAG durability (dag_scheduler → `steps` table) | extend to ResearchContext (W2) |

---

## 2. Architecture (the layered picture)

```
   ┌──────────────────────── Layer 3 — PERSISTENCE (W3) ────────────────────────┐
   │  ingest_url tool · LiveRetriever (mutable corpus) · findings persistence     │
   └──────────────────────────────────────┬─────────────────────────────────────┘
                                          │ consumes
   ┌────────────────────── Layer 2 — DEEP RESEARCH ENGINE (W2) ─────────────────┐
   │  execution.mode: deep_research                                              │
   │  ResearchContext (sub_questions · SourceStore · coverage_map · budget)      │
   │  coverage-gated loop (CoverageEvaluator + max_depth) · citation threading   │
   │  research tool bundle on planned nodes                                      │
   └──────────────────────────────────────┬─────────────────────────────────────┘
                                          │ calls
   ┌────────────────────────── Layer 1 — I/O (W0 + W1) ─────────────────────────┐
   │  koboi/web/  ·  search_provider_registry  (Brave · Firecrawl · ddg · mock)  │
   │               fetch_provider_registry   (httpx+readability · Firecrawl ·    │
   │                                          Playwright)                        │
   │  shared SSRF guard · shared primitive: fetch→extract→chunk→rank→cite        │
   └─────────────────────────────────────────────────────────────────────────────┘
```

**The keystone:** `fetch → extract → chunk → rank → cite` is one primitive. RAG-over-URL and
deep research are two callers of it. RAG ingests a static URL corpus at build; deep research
calls it live, iteratively, with citations. Build once.

---

## 3. Wave detail

### W0 — Search I/O (~2-3 days) · enabler

Fixes the reported "web_search doesn't work." Scope: a **search-provider registry** + Brave +
Firecrawl-search, with `web_search` refactored to a thin wrapper (names/signatures unchanged →
full back-compat). See §4 for the abstraction design.

### W1 — Fetch I/O (~3-4 days) · enabler

A **fetch-provider registry** + `trafilatura` readability (default) + Firecrawl-scrape +
optional Playwright escalation for JS SPAs. `web_fetch` refactored to a thin wrapper. RAG
`source: http` flows through the fetch provider (free quality upgrade); new `source: firecrawl`
crawls a whole docs site into the corpus. See §4.

### W2 — Deep Research Engine (~6-8 days) · 🎯 THE PAYLOAD

Detailed in **§5** — this is the deliverable the other waves exist for.

### W3 — Persistence / live corpus (~3-4 days) · enabler

`ingest_url` tool + `LiveRetriever` (mutable corpus) so agents add discovered URLs at runtime,
and so a finished research run **persists its findings** into a reusable corpus. Achieves the
original "RAG not doc-only" goal *and* lets research accumulate across sessions. See §6.

---

## 4. Layer 1 — Web Provider Abstraction (W0/W1)

Mirrors the RAG `ComponentRegistry` (proven 4× in-tree: chunker/retriever/augmentation/parser).
**Two registries** because search and fetch are distinct capabilities and **Firecrawl implements both.**

### 4.1 Package layout

```
koboi/web/                       NEW top-level pkg (sibling of rag/, sandbox/)
  __init__.py                    re-exports + _register_builtins() at import
  types.py                       SearchResult, FetchResult
  base.py                        BaseSearchProvider / BaseFetchProvider ABCs
  registry.py                    both registries, @register_*, build_search_provider/build_fetch_provider
  providers/
    mock.py  ddg.py              migrated from web.py (fallbacks)
    brave.py                     BraveSearchProvider   (search)
    firecrawl.py                 FirecrawlSearchProvider + FirecrawlFetchProvider (+ crawl source)
    readability.py               ReadabilityFetchProvider (httpx + trafilatura; default fetch)
    playwright.py  (W1.5)        PlaywrightFetchProvider (JS escalation)

koboi/tools/builtin/web.py       REFACTORED → thin tool wrappers over koboi/web/
koboi/rag/sources.py             fetch_http() → build_fetch_provider(); new fetch_firecrawl_entry()
```

### 4.2 Types & ABCs (sketch)

```python
# koboi/web/types.py
@dataclass
class SearchResult:
    title: str; url: str; snippet: str = ""
    score: float | None = None; raw: dict = field(default_factory=dict)

@dataclass
class FetchResult:
    url: str; content: str; title: str = ""; content_type: str = ""
    status: int = 200; truncated: bool = False; metadata: dict = field(default_factory=dict)

# koboi/web/base.py
class BaseSearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, *, max_results: int = 10) -> list[SearchResult]: ...

class BaseFetchProvider(ABC):
    @abstractmethod
    async def fetch(self, url: str, *, render: str = "auto", timeout: int = 15) -> FetchResult: ...
```

### 4.3 Registry + decorator

```python
# koboi/web/registry.py  (mirrors koboi/rag/registry.py)
search_provider_registry: dict[str, ProviderEntry] = {}
fetch_provider_registry:  dict[str, ProviderEntry] = {}

def register_search_provider(name, *, description="", inject=None, config_aliases=None): ...
def register_fetch_provider(name, *, description="", inject=None, config_aliases=None): ...
def build_search_provider(config: dict) -> BaseSearchProvider: ...   # web.search.provider → instance
def build_fetch_provider(config: dict) -> BaseFetchProvider: ...     # web.fetch.provider  → instance
```

`config_aliases` validated at register time (raises if a YAML key maps to a nonexistent
`__init__` param) — copied verbatim from RAG so the extension contract is identical.

### 4.4 Built-in providers

| Registry | Name | How | Auth / Deps |
|---|---|---|---|
| search | `mock` | hardcoded index (migrated) | — |
| search | `ddg` | DDG HTML scrape (migrated; fallback) | — |
| search | `brave` | `GET api.search.brave.com/res/v1/web/search` | `X-Subscription-Token: ${BRAVE_API_KEY}` · free 2000 q/mo |
| search | `firecrawl` | `POST api.firecrawl.dev/v1/search` | `Bearer ${FIRECRAWL_API_KEY}` · optional inline `scrapeOptions` |
| fetch | `httpx` *(default)* | httpx + **trafilatura** readability | `[web]` extra · guarded import, regex fallback |
| fetch | `firecrawl` | `POST api.firecrawl.dev/v1/scrape` `{formats:["markdown"], onlyMainContent:true}` | SaaS |
| fetch | `playwright` *(W1.5)* | headless Chromium, network-idle | `[browser]` extra · `sandbox.backend: restricted` |

`web.fetch.render: never|auto|always` controls JS escalation (`auto` = try httpx, escalate if
extracted text < N chars or SPA detected).

### 4.5 Config schema (new `web:` section)

```yaml
web:
  search:
    provider: brave            # brave | firecrawl | ddg | mock | <custom>
    max_results: 10
    brave: { api_key: ${BRAVE_API_KEY:}, country: "US", freshness: "pw" }
  fetch:
    provider: httpx            # httpx | firecrawl | playwright | <custom>
    render: never              # never | auto | always
    max_chars: 20000
    firecrawl: { api_key: ${FIRECRAWL_API_KEY:}, only_main_content: true }
  custom_modules: [mycorp.web_providers.bing]   # YAML-driven custom providers
```

### 4.6 DI wiring (facade builds once, injects — like `sandbox`/`tool_state`)

```python
# facade.py
registry.set_dep("search_provider", build_search_provider(config.get("web", {})))
registry.set_dep("fetch_provider",  build_fetch_provider(config.get("web", {})))

# koboi/tools/builtin/web.py (refactored — thin wrapper, contract unchanged)
@tool(name="web_search", ...)
async def web_search(query: str, _deps: dict, _tool_config: dict) -> str:
    provider = _deps.get("search_provider") or _default_search_provider()
    return _format_search_results(query, await provider.search(query, ...))

@tool(name="web_fetch", risk_level=RiskLevel.MODERATE, ...)
async def web_fetch(url: str, timeout: int = 15, _deps: dict, _tool_config: dict) -> str:
    provider = _deps.get("fetch_provider") or _default_fetch_provider()
    result = await provider.fetch(url, timeout=timeout)
    return _truncate(result.content, result.truncated)
```

The SSRF guard (`_check_url_ssrf`, web.py:270) stays a **shared utility every fetch provider
must call** — including Firecrawl/Playwright (defense in depth: the agent must not use a SaaS
renderer as an internal-topology / metadata-endpoint probe).

### 4.7 Custom provider = zero core changes

```python
# mycorp/web_providers/bing.py
@register_search_provider("bing", description="Bing Web Search", inject=["api_key"])
class BingSearchProvider(BaseSearchProvider):
    def __init__(self, api_key: str = "", *, endpoint: str = "https://api.bing.microsoft.com/v7.0/search"):
        ...
    async def search(self, query: str, *, max_results: int = 10) -> list[SearchResult]: ...
```
```yaml
web: { search: { provider: bing, bing: { api_key: ${BING_API_KEY:} } }, custom_modules: [mycorp.web_providers.bing] }
```

---

## 5. Layer 2 — Deep Research Engine (W2) · 🎯 the payload

This is the deliverable. Everything else exists to feed it.

### 5.1 New orchestration mode: `execution.mode: deep_research`

An **opinionated preset over `dynamic`** that turns the planner + DAG into a research loop.
Plain `dynamic` stays general-purpose. Config:

```yaml
orchestration:
  enabled: true
  execution:
    mode: deep_research
research:
  max_depth: 3                 # coverage-gated replan rounds
  max_searches: 20             # budget cap (hard stop)
  max_fetches: 30              # budget cap
  max_sources: 15              # keep top-N sources in the final report
  coverage_threshold: 0.7      # CoverageEvaluator score to stop iterating
  citations: numbered          # numbered | inline | none
  search_provider: brave       # ref into web.search (default: web.search.provider)
  fetch_provider: httpx        # ref into web.fetch
```

### 5.2 ResearchContext — shared state across the DAG (journaled for resume)

Today the orchestrator passes `query → answer` between nodes. Deep research needs richer
shared state. `ResearchContext` holds:

```python
@dataclass
class ResearchContext:
    sub_questions: list[str]                       # decomposed plan
    source_store: SourceStore                       # URL → content → chunks → citation id
    coverage_map: dict[str, Coverage]               # sub_q → {covered, evidence: [chunk_id]}
    budget: ResearchBudget                          # {searches, fetches, tokens} used/capped
    findings: list[Passage]                         # accumulated ranked evidence
    depth: int = 0
```

**Journaled to the `steps` table** (extends `dag_scheduler`'s `graph_plan`/`graph_node_complete`)
so `koboi run --resume` rehydrates `SourceStore` + `coverage_map` + `budget` → **no re-fetch,
no re-bill** after a crash/redeploy. Aligns with the existing P2-A journal work.

### 5.3 The loop (one depth-round)

1. **Plan** — `plan_or_skip` with a research-flavored prompt that emits, per step, a sub-question
   **and** its seed search queries. (Existing planner; extended prompt + optional `tools` field.)
2. **Research wave** (DAG, parallel across sub-questions; each node has the research tool bundle):
   `web_search` → rank results → `web_fetch` top-K → extract → chunk into `SourceStore` →
   retriever ranks passages by relevance to the sub-question → write `coverage_map`.
3. **Assess** — `CoverageEvaluator` (one LLM-judge call) scores overall coverage and emits
   follow-up queries for uncovered sub-questions.
4. **Iterate or synthesize** — if `coverage < threshold` AND `depth < max_depth` AND budget
   remains → replan with follow-up queries → loop. Else → synthesize.

### 5.4 The 4 deliverables (the W2 work, mapped to the gaps)

**(a) Research tool bundle on planned nodes** — closes the #1 blocker.
Today `PlanStep → AgentDef` carries only `system_prompt` (planner.py:23-28). In `deep_research`
mode every planned node auto-receives the bundle `[web_search, web_fetch, memory_store,
memory_recall]` via `AgentDef.tools_config`. The planner schema gains an optional `tools` field
so a step can narrow its own tools; absent → mode default applies.

**(b) Coverage-gated research loop** — the thing that makes it "deep."
`CoverageEvaluator(query, sub_questions, findings) → (score 0..1, follow_up_queries)`. Extends
`max_replans` semantics (today `orchestrator.py:132`, default 0, **failure-only**): in
`deep_research` the replan trigger is *low coverage*, not node failure, bounded by `max_depth`.
A node still failing still triggers a replan (existing path preserved).

**(c) SourceStore + citation threading** — generalizes `augmentation.py:95`.
`SourceStore` normalizes each fetched URL (dedup, strip tracking params), chunks it, assigns a
stable citation id `[n]`. Synthesis gets the store as a numbered "Sources" list + a prompt that
forces inline `[n]` markers resolving to URLs. Stamped to `RunResult.metadata['research_sources']`
for eval/UI. The same store feeds W3 persistence.

**(d) Budget caps** — mandatory (research is LLM-call-heavy and runaway-prone).
Per-run `ResearchBudget` enforces `max_searches`/`max_fetches`/`max_tokens`/`max_depth` hard
stops. Reuses the harness budget + jobs `max_iterations` machinery — no new concept.

### 5.5 Streaming + observability

Deep research runs minutes-long. Reuse `Orchestrator.run_stream` events and add research
subtypes for progress UI + eval trace: `SearchEvent`, `FetchEvent`, `SourceEvent`,
`CoverageEvent` (depth N, score, gaps). Surfaced in TUI + SSE.

### 5.6 Reuses (no rebuild)

`plan_or_skip` · DAG waves · `dag_scheduler` durability · RAG chunker/retriever (passage
ranking over fresh content) · orchestrator synthesis · harness budget. W2 is **orchestration
glue + 3 new classes** (ResearchContext, CoverageEvaluator, SourceStore), not a new engine.

---

## 6. Layer 3 — Persistence / live corpus (W3)

Two outcomes from one layer: **"RAG not doc-only"** (the original ask) **+ research findings
accumulate** (deep-research enhancer).

- **`ingest_url` tool** (MODERATE) — fetch via the W1 fetch provider → chunk via the RAG
  chunker → append to a `LiveCorpus`. Lets the agent grow its knowledge mid-conversation.
- **`LiveRetriever`** — today retrievers are construction-bound (`_chunks` fixed at `__init__`,
  retriever.py:41). Small contract addition: `add_chunks(chunks) -> None` on a wrapper, so a
  corpus can grow after build without re-instantiating the pipeline.
- **Research-output persistence** — a finished `deep_research` run dumps `SourceStore` to a
  named corpus file (`research.persist_findings: ./research_corpora/<topic>.jsonl`) → reusable
  across sessions, so follow-up questions don't re-research from scratch.

---

## 7. Security / SSRF / sandbox / license

- **SSRF** — `_check_url_ssrf` (private ranges + per-hop DNS re-check, web.py:253-267) is
  mandatory for every fetch provider, including SaaS ones. The agent must not use Firecrawl as
  a probe for internal topology or a metadata-endpoint oracle.
- **Sandbox** — httpx/Firecrawl run in-process (HTTP, not subprocess; subprocess rules don't
  apply). **Playwright** spawns Chromium → requires `sandbox.backend: restricted` + constrained
  profile (aligns with `docs/agentic-vs-autonomous-strategy.md` rec #10).
- **Secrets** — `${VAR:default}` interpolation only; never cache-key material or logged
  (mirrors `rag/sources.py:_SECRET_KEYS`).
- **Budget** — research caps are hard stops, not advisory (runaway = real $).
- **License** — `trafilatura` (Apache-2.0), `playwright` (Apache-2.0), Firecrawl/Brave/Tavily
  (HTTP only). No AGPL/FSL (same hygiene that excluded `pymupdf`). SaaS providers opt-in behind
  a key; core works with `mock`/`ddg` + regex-extractor fallback.

---

## 8. Tests + eval

- `tests/web/` — registry resolution, alias validation, unknown→fallback, custom-module import,
  each provider via mocked `httpx.MockTransport`, SSRF on all paths, render escalation.
- `tests/orchestration/test_deep_research.py` — `ResearchContext` round-trip, coverage gate
  triggers replan then stops at `max_depth`, budget hard-stop, SourceStore citation mapping,
  resume rehydrates context (no re-fetch), tool bundle attached to planned nodes.
- `tests/rag/test_sources_firecrawl.py` — `source: firecrawl` crawl loop + `DocumentCache`.
- Keep `mock` provider as the default in `conftest.py` → suite stays offline / CI-safe.
- **Empirical harness** `experiment_web_research_gaps.py` (pola `experiment_mcp_gaps.py`)
  asserting: default provider resolvable, Brave/Firecrawl register, custom decorator fires,
  SSRF enforced on Firecrawl path, planned nodes get tools in `deep_research`, coverage<threshold
  triggers replan, `max_depth` bounds it, `--resume` skips re-fetch.
- **Citation/faithfulness eval** — ties into the existing RAG eval-gap work (RAGASScorer shipped
  but never invoked — wire it here so deep-research grounding is evidenced, not claimed).

---

## 9. Migration / back-compat

| Before | After |
|---|---|
| `WEB_SEARCH_PROVIDER=duckduckgo` | still works (mapped to `web.search.provider: ddg`) |
| no env, no `web:` section | `mock` default (unchanged, now logged loudly) |
| `web_search` / `web_fetch` tool signatures | unchanged (LLM contract stable) |
| `rag.documents: [{source: http, url: ...}]` | unchanged output, better extraction (readability) |
| existing configs (`sales_agent.yaml`, `rag_agent.yaml`, …) | work as-is |
| `execution.mode: dynamic` | unchanged; `deep_research` is a new additive mode |

No breaking changes. Every wave ships independently.

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| Cost/runaway (research is LLM-heavy) | hard budget caps (W2-d) + reuse jobs `max_iterations` |
| CoverageEvaluator over/under-iterates | `max_depth` hard bound + threshold tunable + log every round |
| Citation hallucination (model cites wrong `[n]`) | synthesis prompt + post-check that every `[n]` resolves to a SourceStore id; eval |
| Playwright weight/sandbox escape | W1.5, optional `[browser]` extra, restricted sandbox |
| Provider key drift | `${VAR}` interpolation + clear "no provider configured" logging |
| Resume re-billing | ResearchContext journaled → `--resume` reuses SourceStore |

---

## 11. Open decisions (need your call before starting)

1. **Built-ins first**: Brave + Firecrawl (reco), or +Tavily too?
2. **`trafilatura`**: hard dep, or `[web]` extra + regex fallback? (reco: extra — pypdf precedent)
3. **`source: firecrawl` crawl**: W1 (reco — it's a static loader, fits `sources.py` today) or W3?
4. **`mock` silent default**: keep (back-compat, reco) or flip to a clear "no provider configured" error?
5. **W2 sequencing**: scaffold W2 on mock providers in parallel with W0/W1 (reco — fastest critical
   path), or strictly serialize W0→W1→W2?
6. **Citation style default**: `numbered` `[1]` (reco) or inline `(example.com)`?

---

## 12. Why this is low-risk

- **Proven pattern 4× in-tree** (`koboi/rag/registry.py`) for W0/W1 — no new architectural concept.
- **W2 is glue, not a new engine** — reuses planner, DAG, durability, RAG ranker, synthesis.
- **Fully additive** — new package, new mode, thin tool refactor, optional extras. Existing
  configs and tests unchanged.
- **Tool contract stable** — `web_search`/`web_fetch` keep names + params; only the engine swaps.
- **Each wave independently shippable** — W0 alone fixes the reported bug; W2 is unreachable
  without W0/W1 but its non-I/O classes can start immediately on mocks.
