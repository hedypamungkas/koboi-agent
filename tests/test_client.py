"""Tests for koboi.client module."""

from __future__ import annotations

import pytest

from koboi.client import Client, ClientError
from koboi.llm.anthropic_adapter import AnthropicAdapter
from koboi.llm.openai_adapter import OpenAIAdapter


class TestClientConfig:
    def test_placeholder_key_rejected(self):
        with pytest.raises(ClientError, match="API key"):
            Client(api_key="", base_url="http://localhost:8080/v1")

    def test_placeholder_key_sk_your_api_key(self):
        with pytest.raises(ClientError, match="API key"):
            Client(api_key="sk-your-api-key", base_url="http://localhost:8080/v1")

    def test_placeholder_key_sk_xxx(self):
        with pytest.raises(ClientError, match="API key"):
            Client(api_key="sk-xxx", base_url="http://localhost:8080/v1")

    def test_missing_base_url_uses_default(self):
        client = Client(api_key="sk-valid-key", base_url=None)
        assert client.api_key == "sk-valid-key"

    def test_anthropic_provider_resolves_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
        client = Client(provider="anthropic")
        assert client.api_key == "sk-ant-test-key"
        assert client.model == "claude-sonnet-4-20250514"

    def test_openai_provider_default_model(self, monkeypatch):
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        client = Client(api_key="sk-valid-key", base_url="http://localhost:8080/v1")
        assert client.model == "gpt-4o-mini"

    def test_anthropic_error_message_mentions_anthropic_key(self):
        with pytest.raises(ClientError, match="ANTHROPIC_API_KEY"):
            Client(provider="anthropic", api_key="")

    def test_openai_error_message_mentions_openai_key(self):
        with pytest.raises(ClientError, match="OPENAI_API_KEY"):
            Client(api_key="")

    def test_default_provider_is_openai(self):
        client = Client(api_key="sk-valid-key", base_url="http://localhost:8080/v1")
        assert client.provider == "openai"


class TestClientDelegation:
    def test_openai_creates_openai_adapter(self):
        client = Client(
            api_key="sk-test",
            base_url="http://localhost:8080/v1",
            provider="openai",
        )
        assert isinstance(client._impl, OpenAIAdapter)

    def test_anthropic_creates_anthropic_adapter(self):
        client = Client(
            api_key="sk-ant-test",
            base_url="https://api.anthropic.com/v1",
            provider="anthropic",
        )
        assert isinstance(client._impl, AnthropicAdapter)

    def test_timeout_forwarded(self):
        client = Client(
            api_key="sk-test",
            base_url="http://localhost:8080/v1",
            timeout=30.0,
        )
        assert client._impl._transport._timeout == 30.0

    def test_max_tokens_forwarded_anthropic(self):
        client = Client(
            api_key="sk-ant-test",
            base_url="https://api.anthropic.com/v1",
            provider="anthropic",
            max_tokens=8192,
        )
        assert client._impl._max_tokens == 8192

    def test_auth_token_forwarded_anthropic(self):
        client = Client(
            api_key="sk-ant-test",
            base_url="https://api.anthropic.com/v1",
            provider="anthropic",
            auth_token="my-oauth-token",
        )
        headers = client._impl._transport._auth.apply({})
        assert "Authorization" in headers


class TestClientCompleteDelegation:
    async def test_complete_delegates_to_impl(self):
        client = Client(api_key="sk-test", base_url="http://localhost:8080/v1")
        mock_response = type("R", (), {"content": "test", "tool_calls": [], "usage": None})()

        async def _complete(self, m, t):
            return mock_response

        async def _get_embeddings(self, t):
            return None

        client._impl = type(
            "MockImpl",
            (),
            {
                "complete": _complete,
                "get_embeddings": _get_embeddings,
            },
        )()
        result = await client.complete([{"role": "user", "content": "hi"}])
        assert result.content == "test"

    async def test_get_embeddings_delegates_to_impl(self):
        client = Client(api_key="sk-test", base_url="http://localhost:8080/v1")

        async def _complete(self, m, t):
            return None

        async def _get_embeddings(self, t):
            return [0.1, 0.2]

        client._impl = type(
            "MockImpl",
            (),
            {
                "complete": _complete,
                "get_embeddings": _get_embeddings,
            },
        )()
        result = await client.get_embeddings("hello")
        assert result == [0.1, 0.2]


class TestClientEnvResolution:
    def test_openai_env_vars(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://custom.openai.com/v1")
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        client = Client()
        assert client.api_key == "sk-from-env"
        assert client.base_url == "https://custom.openai.com/v1"
        assert client.model == "gpt-4o-mini"

    def test_anthropic_env_vars(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://custom.anthropic.com/v1")
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        client = Client(provider="anthropic")
        assert client.api_key == "sk-ant-from-env"
        assert client.base_url == "https://custom.anthropic.com/v1"
        assert client.model == "claude-haiku-4-5-20251001"

    def test_explicit_params_override_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "env-url")
        client = Client(api_key="explicit-key", base_url="explicit-url")
        assert client.api_key == "explicit-key"
        assert client.base_url == "explicit-url"


class TestOAuthTokenAuth:
    """Tests for auth_type='oauth_token' validation and wiring."""

    def test_oauth_token_passes_validation(self):
        client = Client(
            provider="anthropic",
            auth_type="oauth_token",
            auth_token="sk-ant-oat01-valid-token",
            base_url="https://api.anthropic.com/v1",
        )
        assert client._raw_auth_token == "sk-ant-oat01-valid-token"

    def test_oauth_token_empty_rejected(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        with pytest.raises(ClientError, match="OAuth token"):
            Client(provider="anthropic", auth_type="oauth_token", auth_token="")

    def test_oauth_token_placeholder_rejected(self):
        for placeholder in ("your-api-key-here", "sk-your-api-key", "sk-xxx"):
            with pytest.raises(ClientError, match="OAuth token"):
                Client(
                    provider="anthropic",
                    auth_type="oauth_token",
                    auth_token=placeholder,
                )

    def test_oauth_token_unresolved_env_rejected(self):
        with pytest.raises(ClientError, match="OAuth token"):
            Client(
                provider="anthropic",
                auth_type="oauth_token",
                auth_token="${ANTHROPIC_AUTH_TOKEN}",
            )

    def test_oauth_clears_api_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-should-be-ignored")
        client = Client(
            provider="anthropic",
            auth_type="oauth_token",
            auth_token="sk-ant-oat01-valid-token",
            base_url="https://api.anthropic.com/v1",
        )
        assert client.api_key == ""

    def test_oauth_sends_bearer_header(self):
        client = Client(
            provider="anthropic",
            auth_type="oauth_token",
            auth_token="sk-ant-oat01-valid-token",
            base_url="https://api.anthropic.com/v1",
        )
        headers = client._impl._transport._auth.apply({})
        assert headers["Authorization"] == "Bearer sk-ant-oat01-valid-token"
        assert "x-api-key" not in headers

    def test_oauth_token_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-ant-oat01-from-env")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        client = Client(
            provider="anthropic",
            auth_type="oauth_token",
            base_url="https://api.anthropic.com/v1",
        )
        assert client._raw_auth_token == "sk-ant-oat01-from-env"

    def test_oauth_explicit_token_overrides_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "env-token")
        client = Client(
            provider="anthropic",
            auth_type="oauth_token",
            auth_token="explicit-token",
            base_url="https://api.anthropic.com/v1",
        )
        assert client._raw_auth_token == "explicit-token"

    def test_oauth_creates_anthropic_adapter(self):
        client = Client(
            provider="anthropic",
            auth_type="oauth_token",
            auth_token="sk-ant-oat01-valid",
            base_url="https://api.anthropic.com/v1",
        )
        assert isinstance(client._impl, AnthropicAdapter)

    def test_api_key_still_required_when_auth_type_is_api_key(self):
        with pytest.raises(ClientError, match="API key"):
            Client(provider="anthropic", auth_type="api_key", api_key="")

    def test_auth_type_default_is_api_key(self):
        client = Client(
            api_key="sk-valid",
            base_url="http://localhost:8080/v1",
        )
        assert client.auth_type == "api_key"

    def test_api_key_mode_sends_x_api_key_header(self):
        client = Client(
            provider="anthropic",
            api_key="sk-ant-api03-test",
            base_url="https://api.anthropic.com/v1",
        )
        headers = client._impl._transport._auth.apply({})
        assert headers["x-api-key"] == "sk-ant-api03-test"
