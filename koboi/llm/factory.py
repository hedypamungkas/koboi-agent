"""koboi/llm/factory.py -- Factory function for creating LLM clients."""

from __future__ import annotations

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


def create_client(
    provider: str,
    model: str,
    api_key: str = "",
    base_url: str = "",
    logger: AgentLogger | None = None,
    timeout: float = 120.0,
    max_tokens: int = 4096,
    auth_token: str = "",
    temperature: float | None = None,
    embedding_model: str = "text-embedding-3-small",
    api_version: str = "2023-06-01",
    transport_retries: int = 2,
) -> LLMClient:
    """Create an LLM client for the given provider.

    Args:
        provider: "openai" (default), "cloudflare", or "anthropic".
        model: Model identifier string.
        api_key: API key for authentication.
        base_url: Provider base URL override.
        logger: Optional agent logger.
        timeout: HTTP request timeout in seconds.
        max_tokens: Max tokens for generation (required by Anthropic).
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
    max_tokens: int = 4096,
    auth_token: str = "",
    temperature: float | None = None,
    embedding_model: str = "text-embedding-3-small",
    api_version: str = "2023-06-01",
    transport_retries: int = 2,
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
    )


def _create_anthropic(
    model: str,
    api_key: str,
    base_url: str,
    logger: AgentLogger | None,
    timeout: float,
    max_tokens: int = 4096,
    auth_token: str = "",
    temperature: float | None = None,
    embedding_model: str = "text-embedding-3-small",
    api_version: str = "2023-06-01",
    transport_retries: int = 2,
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
    )
