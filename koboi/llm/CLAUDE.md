# koboi/llm/ -- Multi-provider LLM clients

## What this is
LLM HTTP clients for OpenAI-compatible, Anthropic, and Cloudflare (Workers AI) APIs. Pluggable
provider registry + adapter ABC; an in-memory failover `ProviderPool` for multi-provider setups.
The agent loop talks to a single `LLMClient` (a `RetryClient` or a `ProviderPool`) -- the pool is a
drop-in that implements the same interface. An optional `CachedClient` (`cache.py`) wraps the client
for `replay_mode: cache|replay` (workflow-export determinism).

## Key files
```
base.py              LLMClient ABC (complete/get_embeddings/complete_stream/close) + LLMError hierarchy
auth.py              AuthStrategy ABC + BearerAuth/APIKeyHeaderAuth/StaticHeaderAuth/CompositeAuth
http_transport.py    HttpTransport -- async httpx POST + SSE stream; status->LLMError mapping; retry 429/5xx
registry.py          ProviderDescriptor (frozen dataclass) + ProviderRegistry; register_builtin_providers()
factory.py           create_client() + build_embedding_client(); per-provider extra-param filtering
openai_adapter.py    OpenAIAdapter (LLMClient) -- OpenAI-compatible chat/embeddings/stream
anthropic_adapter.py AnthropicAdapter (LLMClient) -- Anthropic Messages API; no embeddings
resolve.py           resolve_llm_spec() -- normalize llm:/embedding: spec (inline dict | named providers: ref)
pool.py              ProviderPool (LLMClient) + FailoverPolicy + CircuitBreaker; ProviderPoolExhausted
cache.py             ResponseCache (SHA-256-keyed LLM response memo) + CachedClient (LLMClient wrapper for cache/replay determinism)
__init__.py          Re-exports public surface; calls register_builtin_providers() at import
```

## Extension API -- add a provider
1. Write a factory (called by `create_client` with these kwargs):
   `def _create_x(model, api_key, base_url, logger, timeout, max_tokens=None, auth_token="", temperature=None, embedding_model="text-embedding-3-small", api_version="2023-06-01", transport_retries=2, extra_params=None) -> LLMClient`
   Build an `HttpTransport` + `AuthStrategy`, return an `LLMClient` (a subclass, or reuse the OpenAI adapter for OpenAI-compatible APIs).
2. Register -- NO decorator, explicit call (mirrors `register_builtin_providers`):
   `ProviderRegistry.register(ProviderDescriptor(name="x", env_key_api="X_API_KEY", env_key_base_url="X_BASE_URL", env_key_model="X_MODEL", default_model="...", default_base_url="...", factory=_create_x))`
   (`extra_env={"key": "ENV_VAR"}` adds optional resolved env.)
3. Built-ins `openai`/`anthropic`/`cloudflare` are registered at import in `registry.py`.

## LLMClient ABC (base.py)
```python
class LLMClient(ABC):
    @property
    def model(self) -> str: ...                                  # default ""
    @abstractmethod
    async def complete(self, messages, tools=None) -> AgentResponse: ...
    @abstractmethod
    async def get_embeddings(self, text) -> list[float] | None: ...
    async def complete_stream(self, messages, tools=None) -> AsyncIterator[StreamEvent]: ...  # default: fall back to complete()
    async def close(self) -> None: ...                           # default no-op
```

## Conventions
- `Client = LLMClient` (alias in `client.py`) is the type threaded through the loop/orchestration/server.
- `RetryClient` (client.py, default) wraps a `create_client()` adapter and sets `self.provider`; `ProviderPool` is the multi-provider alternative.
- Providers register via `ProviderRegistry.register(ProviderDescriptor(...))` -- no decorator; `register_builtin_providers()` runs at import. `ProviderRegistry.clear()` exists for test isolation.
- `create_client` filters forward-as-is extra params per provider (`_PROVIDER_EXTRA_KEYS` allowlist + `_PROVIDER_PARAM_RENAMES`); unknown providers forward as-is; dropped keys log a warning. `""`/`"openai_compatible"` normalize to `"openai"`.
- `resolve_llm_spec(spec, config)` normalizes specs: inline dict -> as-is; string -> `providers:` map entry; empty -> None. The `{pool: name}` form is routed by the facade to `_build_pool_from_spec`; resolve itself raises `NotImplementedError` for it.
- Pool construction (Tier 2) lives in the facade (`_build_pool_from_spec`); only `FailoverPolicy` ships here -- `round_robin`/budget arrive later.

## Gotchas
- Adapters expose NO `provider` attribute; `ProviderPool._label` reads it via `getattr(client, 'provider', '?')`. Put `RetryClient`s in a pool, not bare adapters, or member labels show `?/<model>`.
- `AnthropicAdapter.get_embeddings()` always returns None -- an Anthropic-only client can't embed. Set a separate `embedding:` section (OpenAI) so `build_embedding_client` returns a real client; retrieval else falls back to keyword.
- Anthropic's API REQUIRES `max_tokens`; `AnthropicAdapter` falls back to 4096 when None. OpenAI/Cloudflare omit it when None (None-sentinel).
- OpenAI adapter DROPS `max_tokens` when `max_completion_tokens` is forwarded (o-series rejects both together).
- `ProviderPool` fails over only BEFORE the first stream byte; once yielding, a mid-stream error re-raises (mirrors `RetryClient`). Pre-first-byte exhaustion raises `ProviderPoolExhausted` with the full failure chain.
- `CircuitBreaker` is in-memory and per-pool-instance (shared across that pool's callers, NOT persisted). Opens after 3 failures, 30s cooldown (defaults); logs the closed->open transition.
- Pool narrows failover to `LLMError` only -- programming bugs (`TypeError`/`KeyError`) propagate immediately so they don't pollute the breaker.
- Cloudflare reuses the OpenAI adapter/factory (`_create_openai`) -- OpenAI-compatible subset, no o-series reasoning keys.
