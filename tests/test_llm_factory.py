"""Tests for koboi.llm.factory module."""
from __future__ import annotations

import pytest

from koboi.llm.anthropic_adapter import AnthropicAdapter
from koboi.llm.base import LLMClient, LLMInvalidRequestError
from koboi.llm.factory import create_client
from koboi.llm.openai_adapter import OpenAIAdapter


class TestCreateClient:
    def test_openai_provider_returns_openai_adapter(self):
        client = create_client(
            provider="openai",
            model="gpt-4o-mini",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
        )
        assert isinstance(client, OpenAIAdapter)
        assert isinstance(client, LLMClient)

    def test_empty_provider_defaults_to_openai(self):
        client = create_client(
            provider="",
            model="gpt-4o-mini",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
        )
        assert isinstance(client, OpenAIAdapter)

    def test_openai_compatible_alias(self):
        client = create_client(
            provider="openai_compatible",
            model="local-model",
            api_key="key",
            base_url="http://localhost:8080/v1",
        )
        assert isinstance(client, OpenAIAdapter)

    def test_anthropic_provider_returns_anthropic_adapter(self):
        client = create_client(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="sk-ant-test",
        )
        assert isinstance(client, AnthropicAdapter)
        assert isinstance(client, LLMClient)

    def test_unknown_provider_raises(self):
        with pytest.raises(LLMInvalidRequestError, match="Unknown provider"):
            create_client(provider="gemini", model="x", api_key="k")

    def test_anthropic_default_base_url(self):
        client = create_client(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="sk-ant-test",
        )
        assert "anthropic.com" in client._transport._base_url

    def test_anthropic_custom_base_url(self):
        client = create_client(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="sk-ant-test",
            base_url="https://custom-proxy.example.com/v1",
        )
        assert client._transport._base_url == "https://custom-proxy.example.com/v1"

    def test_anthropic_max_tokens_forwarded(self):
        client = create_client(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="sk-ant-test",
            max_tokens=8192,
        )
        assert client._max_tokens == 8192

    def test_anthropic_auth_token_forwarded(self):
        client = create_client(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="sk-ant-test",
            auth_token="oauth-token-here",
        )
        transport = client._transport
        headers = transport._auth.apply({})
        assert "x-api-key" in headers
        assert "Authorization" in headers

    def test_anthropic_x_api_key_only(self):
        client = create_client(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="sk-ant-key-only",
        )
        headers = client._transport._auth.apply({})
        assert headers.get("x-api-key") == "sk-ant-key-only"
        assert "anthropic-version" in headers

    def test_anthropic_version_header_always_present(self):
        client = create_client(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="sk-ant-test",
        )
        headers = client._transport._auth.apply({})
        assert headers["anthropic-version"] == "2023-06-01"

    def test_timeout_forwarded(self):
        client = create_client(
            provider="openai",
            model="gpt-4o-mini",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            timeout=60.0,
        )
        assert client._transport._timeout == 60.0
