"""Tests for koboi.llm.http_transport module."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from koboi.llm.auth import BearerAuth
from koboi.llm.base import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMInvalidRequestError,
    LLMRateLimitError,
    LLMResponseParseError,
    LLMServerError,
)
from koboi.llm.http_transport import HttpTransport


def _make_transport(base_url="https://api.test.com/v1", max_retries=2) -> HttpTransport:
    return HttpTransport(
        base_url=base_url,
        auth=BearerAuth("test-key"),
        max_retries=max_retries,
    )


class TestHttpTransportErrorMapping:
    async def test_401_raises_authentication_error(self):
        transport = _make_transport()
        mock_response = httpx.Response(
            status_code=401,
            json={"error": {"message": "invalid api key", "type": "authentication_error"}},
            request=httpx.Request("POST", "https://api.test.com/v1/chat/completions"),
        )
        with patch.object(transport._client, "post", return_value=mock_response):
            with pytest.raises(LLMAuthenticationError, match="Authentication failed"):
                await transport.post("/chat/completions", {"model": "test"})

    async def test_403_raises_authentication_error(self):
        transport = _make_transport()
        mock_response = httpx.Response(
            status_code=403,
            json={"error": {"message": "forbidden"}},
            request=httpx.Request("POST", "https://api.test.com/v1/chat/completions"),
        )
        with patch.object(transport._client, "post", return_value=mock_response):
            with pytest.raises(LLMAuthenticationError, match="HTTP 403"):
                await transport.post("/chat/completions", {})

    async def test_429_raises_rate_limit_error_with_retry_after(self):
        transport = _make_transport()
        mock_response = httpx.Response(
            status_code=429,
            headers={"retry-after": "5.0"},
            json={"error": {"message": "slow down"}},
            request=httpx.Request("POST", "https://api.test.com/v1/chat/completions"),
        )
        with patch.object(transport._client, "post", return_value=mock_response):
            with pytest.raises(LLMRateLimitError) as exc_info:
                await transport.post("/chat/completions", {})
            assert exc_info.value.retry_after == 5.0

    async def test_400_raises_invalid_request_error(self):
        transport = _make_transport()
        mock_response = httpx.Response(
            status_code=400,
            json={"error": {"message": "max_tokens is required"}},
            request=httpx.Request("POST", "https://api.test.com/v1/chat/completions"),
        )
        with patch.object(transport._client, "post", return_value=mock_response):
            with pytest.raises(LLMInvalidRequestError, match="max_tokens"):
                await transport.post("/chat/completions", {})

    async def test_500_raises_server_error(self):
        transport = _make_transport()
        mock_response = httpx.Response(
            status_code=500,
            json={"error": {"message": "internal error"}},
            request=httpx.Request("POST", "https://api.test.com/v1/chat/completions"),
        )
        with patch.object(transport._client, "post", return_value=mock_response):
            with pytest.raises(LLMServerError, match="HTTP 500"):
                await transport.post("/chat/completions", {})

    async def test_529_overloaded_raises_server_error(self):
        transport = _make_transport()
        mock_response = httpx.Response(
            status_code=529,
            json={"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
            request=httpx.Request("POST", "https://api.test.com/v1/messages"),
        )
        with patch.object(transport._client, "post", return_value=mock_response):
            with pytest.raises(LLMServerError, match="HTTP 529"):
                await transport.post("/messages", {})


class TestHttpTransportConnectionErrors:
    async def test_connect_error_raises_connection_error(self):
        transport = _make_transport()
        with patch.object(transport._client, "post", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(LLMConnectionError, match="Connection failed"):
                await transport.post("/chat/completions", {})

    async def test_timeout_raises_connection_error(self):
        transport = _make_transport()
        with patch.object(transport._client, "post", side_effect=httpx.TimeoutException("timed out")):
            with pytest.raises(LLMConnectionError, match="timed out"):
                await transport.post("/chat/completions", {})


class TestHttpTransportSuccess:
    async def test_returns_parsed_json(self):
        transport = _make_transport()
        expected = {"choices": [{"message": {"content": "Hello"}}]}
        mock_response = httpx.Response(
            status_code=200,
            json=expected,
            request=httpx.Request("POST", "https://api.test.com/v1/chat/completions"),
        )
        with patch.object(transport._client, "post", return_value=mock_response):
            result = await transport.post("/chat/completions", {"model": "test"})
        assert result == expected

    async def test_malformed_json_raises_parse_error(self):
        transport = _make_transport()
        mock_response = httpx.Response(
            status_code=200,
            text="not json at all",
            request=httpx.Request("POST", "https://api.test.com/v1/chat/completions"),
        )
        with patch.object(transport._client, "post", return_value=mock_response):
            with pytest.raises(LLMResponseParseError, match="Invalid JSON"):
                await transport.post("/chat/completions", {})


class TestHttpTransportRetry:
    async def test_retries_on_500_then_succeeds(self):
        transport = _make_transport(max_retries=2)
        error_response = httpx.Response(
            status_code=500,
            json={"error": {"message": "temp error"}},
            request=httpx.Request("POST", "https://api.test.com/v1/chat/completions"),
        )
        success_response = httpx.Response(
            status_code=200,
            json={"result": "ok"},
            request=httpx.Request("POST", "https://api.test.com/v1/chat/completions"),
        )
        with patch.object(transport._client, "post", side_effect=[error_response, success_response]):
            result = await transport.post("/chat/completions", {})
        assert result == {"result": "ok"}

    async def test_exhausts_retries_raises_server_error(self):
        transport = _make_transport(max_retries=1)
        error_response = httpx.Response(
            status_code=500,
            json={"error": {"message": "still broken"}},
            request=httpx.Request("POST", "https://api.test.com/v1/chat/completions"),
        )
        with patch.object(transport._client, "post", return_value=error_response):
            with pytest.raises(LLMServerError, match="HTTP 500"):
                await transport.post("/chat/completions", {})

    async def test_non_retryable_status_no_retry(self):
        transport = _make_transport(max_retries=2)
        mock_response = httpx.Response(
            status_code=400,
            json={"error": {"message": "bad request"}},
            request=httpx.Request("POST", "https://api.test.com/v1/chat/completions"),
        )
        call_count = 0

        def count_calls(*a, **kw):
            nonlocal call_count
            call_count += 1
            return mock_response

        with patch.object(transport._client, "post", side_effect=count_calls):
            with pytest.raises(LLMInvalidRequestError):
                await transport.post("/chat/completions", {})
        assert call_count == 1


class TestHttpTransportBaseURL:
    def test_trailing_slash_stripped(self):
        transport = _make_transport(base_url="https://api.test.com/v1/")
        assert transport._base_url == "https://api.test.com/v1"
