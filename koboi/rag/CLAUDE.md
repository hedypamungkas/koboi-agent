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
retriever.py        BaseRetriever ABC + Keyword/Semantic/Hybrid retrievers; embedding cache; resolve_retriever()
augmentation.py     AugmentationStrategy ABC + InMemory/OnTheFly; RerankerRetriever wrapper
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

## Gotchas
- **`SemanticChunker` is effectively `SentenceChunker`**: `_get_embeddings_sync()` has no access
  to the LLM client and always returns None, so it falls back every time. Use the *semantic
  retriever* (not the semantic chunker) for embedding-based behavior.
- **`RerankerRetriever` is exported but NOT registered**: you cannot select it by name in YAML.
  It is a manual wrapper -- instantiate it around a base retriever in code.
- **Document sources**: `_load_documents` supports `source: file` (default; local
  path/glob/dir), `source: http` (httpx -- presigned URLs work for R2/S3 public-ish
  objects; 0 new dep), and `source: s3` (boto3, the `[rag-cloud]` extra; Cloudflare R2
  via `endpoint_url`). Loaded bytes are parsed by format (text/html/pdf/docx) via
  `parser_registry` (`[rag]` extra for PDF/DOCX; HTML is stdlib). `document_cache_path`
  caches remote fetches across the per-session rebuilds in `koboi/server/pool.py`.
- **Semantic/Hybrid retrievers degrade to keyword** when no `client` is injected or the embedding
  endpoint returns None (logged as a warning). Hybrid's RRF still fuses both legs.
- **Process-level embedding cache** (`_EMBEDDING_CACHE`) embeds a corpus once per process, keyed by
  content hash; assumes one embedding model per process (model change needs a restart). Call
  `clear_embedding_cache()` for test isolation.
- **`build_rag` returns None silently** when `enabled` is falsy or no document path resolves -- a
  typo'd/missing path yields zero chunks (warning only), so RAG silently does nothing.
- **`config_aliases` validated at register time**: mapping a YAML key to a nonexistent `__init__`
  param raises `ValueError` immediately, not at build time.
