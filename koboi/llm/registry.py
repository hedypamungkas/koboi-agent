"""koboi/llm/registry.py -- Provider descriptor and registry for LLM providers."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from koboi.llm.base import LLMClient

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderDescriptor:
    """Declarative metadata for a single LLM provider."""

    name: str
    env_key_api: str
    env_key_base_url: str
    env_key_model: str
    default_model: str
    default_base_url: str
    factory: Callable[..., LLMClient]
    extra_env: dict[str, str] = field(default_factory=dict)


class ProviderRegistry:
    """Registry of named LLM provider descriptors."""

    _descriptors: dict[str, ProviderDescriptor] = {}

    @classmethod
    def register(cls, desc: ProviderDescriptor) -> None:
        cls._descriptors[desc.name] = desc

    @classmethod
    def get(cls, name: str) -> ProviderDescriptor | None:
        return cls._descriptors.get(name)

    @classmethod
    def list_available(cls) -> list[str]:
        return sorted(cls._descriptors.keys())

    @classmethod
    def clear(cls) -> None:
        """Remove all registered descriptors. Useful for test isolation."""
        cls._descriptors.clear()

    @classmethod
    def resolve_env(cls, provider_name: str, **overrides: str | None) -> dict[str, str]:
        """Resolve env vars for a provider.

        Values come from overrides first, then env, then defaults.
        Returns dict with keys: api_key, base_url, model, plus any extras.
        """
        desc = cls.get(provider_name)
        if desc is None:
            _logger.warning("Unknown provider '%s', falling back to openai", provider_name)
            desc = cls.get("openai")
            if desc is None:
                raise ValueError(f"No provider registered for '{provider_name}'")

        def _resolve(override: str | None, env_key: str, default: str) -> str:
            if override:
                return override
            return os.environ.get(env_key, default)

        result: dict[str, str] = {
            "api_key": _resolve(overrides.get("api_key"), desc.env_key_api, ""),
            "base_url": _resolve(overrides.get("base_url"), desc.env_key_base_url, ""),
            "model": _resolve(overrides.get("model"), desc.env_key_model, desc.default_model),
        }
        for key, env_var in desc.extra_env.items():
            result[key] = _resolve(overrides.get(key), env_var, "")
        return result


def register_builtin_providers() -> None:
    """Register built-in LLM providers. Called once at import time."""
    from koboi.llm.factory import _create_anthropic, _create_openai

    ProviderRegistry.register(
        ProviderDescriptor(
            name="openai",
            env_key_api="OPENAI_API_KEY",
            env_key_base_url="OPENAI_BASE_URL",
            env_key_model="OPENAI_MODEL",
            default_model="gpt-4o-mini",
            default_base_url="https://api.openai.com/v1",
            factory=_create_openai,
        )
    )

    ProviderRegistry.register(
        ProviderDescriptor(
            name="anthropic",
            env_key_api="ANTHROPIC_API_KEY",
            env_key_base_url="ANTHROPIC_BASE_URL",
            env_key_model="ANTHROPIC_MODEL",
            default_model="claude-sonnet-4-20250514",
            default_base_url="https://api.anthropic.com/v1",
            factory=_create_anthropic,
            extra_env={"auth_token": "ANTHROPIC_AUTH_TOKEN"},
        )
    )

    ProviderRegistry.register(
        ProviderDescriptor(
            name="cloudflare",
            env_key_api="CLOUDFLARE_API_TOKEN",
            env_key_base_url="CLOUDFLARE_BASE_URL",
            env_key_model="CLOUDFLARE_MODEL",
            default_model="@cf/meta/llama-3.1-70b-instruct",
            default_base_url="",
            factory=_create_openai,
            extra_env={"account_id": "CLOUDFLARE_ACCOUNT_ID"},
        )
    )
