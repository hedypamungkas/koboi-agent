"""koboi/llm/factory.py -- Factory function for creating LLM clients."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from koboi.llm.auth import (
    APIKeyHeaderAuth,
    AuthStrategy,
    BearerAuth,
    CompositeAuth,
    StaticHeaderAuth,
)
from koboi.llm.base import LLMClient, LLMInvalidRequestError
from koboi.llm.http_transport import HttpTransport

if TYPE_CHECKING:
    from koboi.logger import AgentLogger


_logger = logging.getLogger(__name__)

# Per-provider accepted generation keys (a subset of FORWARDABLE_LLM_KEYS in
# koboi/config.py). Providers do not agree on field names, so the shared
# allowlist alone would forward keys a provider rejects (Anthropic has no
# max_completion_tokens/logprobs/response_format/reasoning_effort; OpenAI has no
# top_k). Filtering here -- in create_client, the one place that knows the
# resolved provider -- turns a silent runtime 400 into a build-time drop + warn.
_PROVIDER_EXTRA_KEYS: dict[str, frozenset[str]] = {
    "openai": frozenset(
        {
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "seed",
            "response_format",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "max_completion_tokens",
            "reasoning_effort",
            "verbosity",
        }
    ),
    "cloudflare": frozenset(
        {  # Workers AI: OpenAI-compatible subset, no o-series reasoning
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "seed",
            "response_format",
        }
    ),
    "anthropic": frozenset({"top_p", "top_k", "thinking", "stop_sequences"}),
}

# 1:1 field renames so a user's key maps to the provider's equivalent instead of
# being dropped (OpenAI "stop" -> Anthropic "stop_sequences").
_PROVIDER_PARAM_RENAMES: dict[str, dict[str, str]] = {
    "anthropic": {"stop": "stop_sequences"},
}


def _filter_extra_params_for_provider(provider: str, extra_params: dict | None) -> dict | None:
    """Drop/translate forward-as-is generation keys the provider doesn't accept.

    Unknown/future providers (no entry in ``_PROVIDER_EXTRA_KEYS``) get the params
    forwarded as-is so newly-registered providers aren't broken. Returns ``None``
    when nothing remains so callers skip the body merge.
    """
    if not extra_params:
        return None
    allowed = _PROVIDER_EXTRA_KEYS.get(provider)
    if allowed is None:
        return extra_params
    renames = _PROVIDER_PARAM_RENAMES.get(provider, {})
    kept: dict = {}
    for key, value in extra_params.items():
        dest = renames.get(key, key)
        if dest in allowed:
            kept[dest] = value
        else:
            _logger.warning(
                "llm extra param %r is not supported by provider %r; dropping "
                "(not forwarded). See the provider's API docs.",
                key,
                provider,
            )
    return kept or None


def create_client(
    provider: str,
    model: str,
    api_key: str = "",
    base_url: str = "",
    logger: AgentLogger | None = None,
    timeout: float = 120.0,
    max_tokens: int | None = None,
    auth_token: str = "",
    temperature: float | None = None,
    embedding_model: str = "text-embedding-3-small",
    api_version: str = "2023-06-01",
    transport_retries: int = 2,
    extra_params: dict | None = None,
) -> LLMClient:
    """Create an LLM client for the given provider.

    Args:
        provider: "openai" (default), "cloudflare", or "anthropic".
        model: Model identifier string.
        api_key: API key for authentication.
        base_url: Provider base URL override.
        logger: Optional agent logger.
        timeout: HTTP request timeout in seconds.
        max_tokens: Max generation tokens. None = unset (OpenAI/Cloudflare omit it;
            Anthropic falls back to its 4096 default, which its API requires).
        auth_token: Optional secondary Bearer token (e.g. Anthropic OAuth).
        temperature: Optional temperature override. None = use provider default.
        embedding_model: Model name for embedding requests.
        api_version: Anthropic API version header.
        transport_retries: HTTP transport retry count.
    """
    from koboi.llm.registry import ProviderRegistry

    # Backward compat: normalize empty/alias strings
    normalized = provider if provider not in ("", "openai_compatible") else "openai"

    desc = ProviderRegistry.get(normalized)
    if desc is None:
        raise LLMInvalidRequestError(f"Unknown provider: '{provider}'. Available: {ProviderRegistry.list_available()}")

    extra_params = _filter_extra_params_for_provider(normalized, extra_params)

    return desc.factory(
        model=model,
        api_key=api_key,
        base_url=base_url,
        logger=logger,
        timeout=timeout,
        max_tokens=max_tokens,
        auth_token=auth_token,
        temperature=temperature,
        embedding_model=embedding_model,
        api_version=api_version,
        transport_retries=transport_retries,
        extra_params=extra_params,
    )


def build_embedding_client(embedding_config: dict | None, logger=None):
    """Build a dedicated embedding client from an ``embedding:`` config section.

    Decouples the embedding provider from the chat provider: when ``api_key`` is
    set, semantic retrieval routes here instead of the chat client (useful when
    the chat provider has no ``/embeddings`` endpoint). Returns ``None`` when the
    section is absent or has no ``api_key`` (e.g. ``EMBEDDING_API_KEY`` unset) so
    callers fall back to the chat client. Uses ``create_client`` so the
    configured ``embedding_model`` is honored.

    Shared by the single-agent facade and the orchestration factory.
    """
    emb = embedding_config or {}
    if not emb.get("api_key"):
        return None
    model = emb.get("model") or "text-embedding-3-small"
    return create_client(
        provider=emb.get("provider", "openai"),
        model=model,
        api_key=emb.get("api_key", ""),
        base_url=emb.get("base_url", ""),
        embedding_model=model,
        logger=logger,
    )


def _create_openai(
    model: str,
    api_key: str,
    base_url: str,
    logger: AgentLogger | None,
    timeout: float,
    max_tokens: int | None = None,
    auth_token: str = "",
    temperature: float | None = None,
    embedding_model: str = "text-embedding-3-small",
    api_version: str = "2023-06-01",
    transport_retries: int = 2,
    extra_params: dict | None = None,
) -> LLMClient:
    from koboi.llm.openai_adapter import OpenAIAdapter

    transport = HttpTransport(
        base_url=base_url or "https://api.openai.com/v1",
        auth=BearerAuth(api_key),
        timeout=timeout,
        max_retries=transport_retries,
    )
    return OpenAIAdapter(
        model=model,
        transport=transport,
        logger=logger,
        embedding_model=embedding_model,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_params=extra_params,
    )


def _create_anthropic(
    model: str,
    api_key: str,
    base_url: str,
    logger: AgentLogger | None,
    timeout: float,
    max_tokens: int | None = None,
    auth_token: str = "",
    temperature: float | None = None,
    embedding_model: str = "text-embedding-3-small",
    api_version: str = "2023-06-01",
    transport_retries: int = 2,
    extra_params: dict | None = None,
) -> LLMClient:
    from koboi.llm.anthropic_adapter import AnthropicAdapter

    strategies: list[AuthStrategy] = []

    if api_key:
        strategies.append(APIKeyHeaderAuth(api_key, header_name="x-api-key"))

    if auth_token:
        strategies.append(BearerAuth(auth_token))

    strategies.append(StaticHeaderAuth("anthropic-version", api_version))

    transport = HttpTransport(
        base_url=base_url or "https://api.anthropic.com/v1",
        auth=CompositeAuth(strategies),
        timeout=timeout,
        max_retries=transport_retries,
    )
    return AnthropicAdapter(
        model=model,
        transport=transport,
        max_tokens=max_tokens,
        logger=logger,
        temperature=temperature,
        extra_params=extra_params,
    )
