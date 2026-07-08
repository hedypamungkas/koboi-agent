"""Tests for client.py -- retry logic, validation, edge cases."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from koboi.client import RetryClient, ClientError, PLACEHOLDER_KEYS
from koboi.llm.registry import ProviderRegistry
from koboi.llm.base import LLMServerError, LLMError
from koboi.types import AgentResponse


class TestClientValidation:
    def test_rejects_placeholder_key(self):
        with pytest.raises(ClientError, match="API key not configured"):
            RetryClient(api_key="your-api-key-here", provider="openai", base_url="http://test/v1")

    def test_rejects_empty_key(self):
        with pytest.raises(ClientError, match="API key not configured"):
            RetryClient(api_key="", provider="openai", base_url="http://test/v1")

    def test_rejects_unresolved_env(self):
        with pytest.raises(ClientError, match="API key not configured"):
            RetryClient(api_key="${OPENAI_API_KEY}", provider="openai", base_url="http://test/v1")

    def test_oauth_token_rejects_placeholder(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ClientError, match="OAuth token not configured"):
            RetryClient(provider="anthropic", auth_token="", auth_type="oauth_token", base_url="http://test/v1")


class TestClientRetry:
    async def test_retries_on_server_error(self):
        mock_impl = MagicMock()
        call_count = 0

        async def fail_then_succeed(messages, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise LLMServerError("server error")
            return AgentResponse(content="ok")

        mock_impl.complete = fail_then_succeed
        mock_impl._transport = MagicMock()

        with patch("koboi.client.create_client", return_value=mock_impl):
            client = RetryClient(api_key="sk-test", provider="openai", base_url="http://test/v1", max_retries=3)
            result = await client.complete([{"role": "user", "content": "hi"}])
            assert result.content == "ok"
            assert call_count == 3

    async def test_raises_after_max_retries(self):
        mock_impl = MagicMock()

        async def always_fail(messages, tools=None):
            raise LLMServerError("server error")

        mock_impl.complete = always_fail
        mock_impl._transport = MagicMock()

        with patch("koboi.client.create_client", return_value=mock_impl):
            client = RetryClient(api_key="sk-test", provider="openai", base_url="http://test/v1", max_retries=1)
            with pytest.raises(LLMServerError):
                await client.complete([{"role": "user", "content": "hi"}])

    async def test_non_retryable_error_propagates(self):
        mock_impl = MagicMock()

        async def auth_fail(messages, tools=None):
            raise LLMError("auth failed")

        mock_impl.complete = auth_fail
        mock_impl._transport = MagicMock()

        with patch("koboi.client.create_client", return_value=mock_impl):
            client = RetryClient(api_key="sk-test", provider="openai", base_url="http://test/v1")
            with pytest.raises(LLMError):
                await client.complete([{"role": "user", "content": "hi"}])

    async def test_unexpected_error_wrapped(self):
        mock_impl = MagicMock()

        async def unexpected_fail(messages, tools=None):
            raise ValueError("unexpected")

        mock_impl.complete = unexpected_fail
        mock_impl._transport = MagicMock()

        with patch("koboi.client.create_client", return_value=mock_impl):
            client = RetryClient(api_key="sk-test", provider="openai", base_url="http://test/v1")
            with pytest.raises(ClientError, match="Unexpected error"):
                await client.complete([{"role": "user", "content": "hi"}])


class TestClientEnvMap:
    def test_env_map_has_providers(self):
        available = ProviderRegistry.list_available()
        assert "openai" in available
        assert "anthropic" in available
        assert "cloudflare" in available

    def test_placeholder_keys(self):
        assert "" in PLACEHOLDER_KEYS
        assert "your-api-key-here" in PLACEHOLDER_KEYS
