"""koboi.llm -- Multi-provider LLM client system.

Supports OpenAI-compatible and Anthropic APIs with pluggable providers.
"""

from koboi.llm.base import (
    LLMClient,
    LLMError,
    LLMConnectionError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMServerError,
    LLMInvalidRequestError,
    LLMResponseParseError,
)
from koboi.llm.auth import AuthStrategy, BearerAuth, APIKeyHeaderAuth, CompositeAuth, StaticHeaderAuth
from koboi.llm.factory import create_client
from koboi.llm.registry import ProviderRegistry, register_builtin_providers

# Register built-in providers at import time
register_builtin_providers()

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMConnectionError",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    "LLMServerError",
    "LLMInvalidRequestError",
    "LLMResponseParseError",
    "AuthStrategy",
    "BearerAuth",
    "APIKeyHeaderAuth",
    "CompositeAuth",
    "StaticHeaderAuth",
    "create_client",
    "ProviderRegistry",
]
