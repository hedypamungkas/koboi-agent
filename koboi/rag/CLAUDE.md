# koboi/rag/ -- Retrieval-Augmented Generation pipeline

## What this is
Pluggable RAG pipeline: chunk documents, retrieve relevant chunks for a query, then
augment the user message or LLM call with retrieved context. Driven by the `rag:`
YAML section; uses the ComponentRegistry pattern (shared with sandbox/guardrails/context).
Three swappable stages -- chunker, retriever, augmentation -- each with built-ins and a
`@register_*` decorator for custom plugins. `build_rag()` composes all three from config.
See `docs/architecture.md` ("RAG Pipeline") for the end-to-end flow.

## Key files
```
types.py            Chunk, Document, RetrievalResult dataclasses
registry.py         ComponentRegistry + ComponentEntry, @register_* decorators, build_rag(), load_custom_components()
chunker.py          BaseChunker ABC + Fixed/Sentence/Paragraph/Semantic chunkers; resolve_chunker()
retriever.py        BaseRetriever ABC + Keyword/BM25/Semantic/Hybrid retrievers; embedding cache; Indonesian stopwords/stemmer (id); resolve_retriever()
augmentation.py     AugmentationStrategy ABC + InMemory/OnTheFly; heuristic RerankerRetriever wrapper
rerank.py           Cross-encoder rerank stage (PR #38): RerankBackend ABC + jina/cohere/local-BGE backends; CrossEncoderReranker wrapper; build_rerank_client() factory
rewrite.py          Query rewriting + HyDE backing module (#9); needs a chat client, output is ephemeral
filters.py          Metadata pre-filter operators (#10); NOT a security/ACL boundary
parsers.py          parser_registry -- text/html/pdf/docx document parsing ([rag] extra for PDF/DOCX);
                     HtmlParser prefers trafilatura (guarded) with a _TagStripper fallback
sources.py          _load_documents source loaders: file/http (httpx)/s3 (boto3, [rag-cloud] extra)/
                     firecrawl (site crawl -> corpus; `source: firecrawl`, [websearch] extra for scrape)
live.py             LiveCorpus (mutable chunk store w/ dirty flag) + LiveRetriever (lazy KeywordRetriever
                     delegate rebuild) -- the `rag.live` swap target for `ingest_url` (fetch -> chunk -> append)
sample_documents.py Loaders for data/sample/ docs (company policy, handbook, catalog)
__init__.py         Re-exports public API; calls _register_builtins() at import
```

## Extension API
One ABC per stage, registered by name into a module-level registry; `build_rag()` resolves each
from the `rag:` config dict.

- **Chunker**: subclass `BaseChunker`, implement `chunk(self, document: Document) -> list[Chunk]` (sync).
  Register: `@register_chunker("name", description="...", config_aliases={...})`.
- **Retriever**: subclass `BaseRetriever`, implement
  `async retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]`.
  Register: `@register_retriever("name", inject=["client"])` -- list `client` in `inject` to have
  the LLM client auto-passed (required for embedding retrievers).
- **Augmentation**: subclass `AugmentationStrategy`; override `async augment_for_memory(user_message)`
  and/or `async augment_for_llm(messages)`. Register: `@register_augmentation("name")`.

Register a custom plugin two ways: (1) YAML-driven -- list dotted module paths under
`rag.custom_modules`; `load_custom_components()` imports each so decorators fire on import;
(2) code-driven -- import your module before `build_rag()` runs (decorators mutate the module-level
registries directly). `ComponentRegistry.register()` introspects `__init__` to map YAML keys to
kwargs and validates `config_aliases` targets exist (raises `ValueError` otherwise).

## Conventions
- `rag:` YAML keys: `enabled`, `chunker` (default `paragraph`), `retriever` (default `keyword`),
  `augmentation` (default `in_memory`), `top_k` (default 3), `documents` (list of `{path: ...}`),
  `custom_modules`, plus per-stage kwargs matched by `__init__` param name.
- `build_rag(rag_conf, *, client=None, logger=None) -> AugmentationStrategy | None` (None when
  disabled or no documents load).
- Score semantics differ per method: keyword = TF-IDF cosine, semantic = embedding cosine,
  hybrid = Reciprocal Rank Fusion (k=60). `relevance_threshold` (on the augmentation) filters by
  raw score -- comparable only within keyword/semantic, not across hybrid's fused rank.
- `AugmentationStrategy.last_results` holds the most recent retrieval so AgentCore can stamp
  `RunResult.metadata['rag_results']` for eval assertions.
- Two injection points: `in_memory` augments the user message before storing; `on_the_fly`
  augments the last user message in-place before each LLM call.
- **Query rewriting / HyDE (#9)**: `rag.query_rewrite: true` rewrites the query (rule-based
  normalization always + an LLM call) before retrieval; `rag.hyde: true` generates a
  hypothetical answer for the semantic/hybrid leg. Needs a **chat** client, plumbed via
  `build_rag(..., chat_client=...)` (distinct from the embedding `client`). Output is
  ephemeral (retrieval query only, never stored); falls back to the raw query on error.
  `AugmentationStrategy.last_rewrite` is stamped to `RunResult.metadata['rag_rewrite']`.
- **Metadata filtering (#10)**: `rag.filter` constrains which chunks a retriever considers
  (relevance scoping -- freshness/source/type), e.g. `{year: {$gte: 2024}, source: {$in: [policy, handbook]}}`.
  Operators: scalar (equality), `$gte`/`$gt`/`$lte`/`$lt`, `$in`. Applied as a pre-filter in each
  retriever (so top_k isn't shrunk). **NOT a security/ACL boundary** -- see `koboi/rag/filters.py`.
- **Cross-encoder rerank (`rerank.py`, PR #38)**: `rag.rerank` is `bool | dict`. `true` (legacy)
  wraps the retriever in the heuristic keyword-overlap `RerankerRetriever`. A **dict**
  `{provider: jina|cohere|local, api_key, model, fetch_multiplier, score_threshold}` selects a
  true cross-encoder: `CrossEncoderReranker` over-fetches (`fetch_multiplier` × `top_k`, clamped
  to the provider batch cap — jina 2048 / cohere 100 / local 10000), re-scores, and stamps a
  distinctive `retrieval_method` (e.g. `rerank:jina(bm25)`) into `RunResult.metadata['rag_results']`
  so evals can detect the provider. Defaults: provider `jina`, model `jina-reranker-v3` /
  `rerank-multilingual-v3.0` (cohere) / `BAAI/bge-reranker-v2-m3` (local). HTTP backends need
  `api_key` (else warn + no rerank); `local`/BGE needs the `[rerank-local]` extra. **Fail-soft** —
  on any provider hiccup the wrapper returns the base retriever's results unchanged. Reuses
  `HttpTransport` + `BearerAuth` + the `LLMError` hierarchy; HTTP transport closed in
  `KoboiAgent.close()`. Unknown `provider` raises `LLMInvalidRequestError` at build time.
- **Indonesian NLP (`retriever.py`, PR #38)**: `rag.stopwords: true|en|id` (id = ~80 function
  words) and `rag.stemmer: id` (Sastrawi via the `[indo-nlp]` extra; **`True` is NOT valid for
  stemmer** — no English stemmer ships). Applied to BOTH index and query tokens on lexical
  retrievers (Keyword/BM25/Hybrid) so morphology (`memakan/makanan`→`makan`) normalizes.

## Gotchas
- **`SemanticChunker` is effectively `SentenceChunker`**: `_get_embeddings_sync()` has no access
  to the LLM client and always returns None, so it falls back every time. Use the *semantic
  retriever* (not the semantic chunker) for embedding-based behavior.
- **Two rerank paths**: `rag.rerank: true` (legacy bool) selects the heuristic keyword-overlap
  `RerankerRetriever` (in `augmentation.py`; also a manual wrapper, NOT registered by name).
  A **dict** (`{provider: jina|cohere|local, ...}`) selects the true cross-encoder in `rerank.py`
  (`CrossEncoderReranker`, wired into `build_rag` via `build_rerank_client`). See the Conventions
  bullet above for backend defaults and batch caps.
- **Document sources**: `_load_documents` supports `source: file` (default; local
  path/glob/dir), `source: http` (httpx -- presigned URLs work for R2/S3 public-ish
  objects; 0 new dep), and `source: s3` (boto3, the `[rag-cloud]` extra; Cloudflare R2
  via `endpoint_url`). Loaded bytes are parsed by format (text/html/pdf/docx) via
  `parser_registry` (`[rag]` extra for PDF/DOCX; HTML is stdlib). `document_cache_path`
  caches remote fetches across the per-session rebuilds in `koboi/server/pool.py`.
  `rag.max_document_size_mb` (default 10) bounds a single remote document; `http`/`s3`
  fetches (`koboi/rag/sources.py`) are streamed with an early Content-Length reject +
  bounded read (not fully buffered before the check) so an oversized/malicious
  response can't exhaust memory (CWE-400, issue #56).
- **Semantic/Hybrid retrievers degrade to keyword** when no `client` is injected or the embedding
  endpoint returns None (logged as a warning). Hybrid's RRF still fuses both legs.
- **Process-level embedding cache** (`_EMBEDDING_CACHE`) embeds a corpus once per process, keyed by
  content hash; assumes one embedding model per process (model change needs a restart). Call
  `clear_embedding_cache()` for test isolation.
- **`build_rag` returns None silently** when `enabled` is falsy or no document path resolves -- a
  typo'd/missing path yields zero chunks (warning only), so RAG silently does nothing.
- **`config_aliases` validated at register time**: mapping a YAML key to a nonexistent `__init__`
  param raises `ValueError` immediately, not at build time.
