"""Tests for koboi.llm.openai_adapter module."""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch

import httpx
import pytest

from koboi.llm.base import LLMResponseParseError
from koboi.llm.http_transport import HttpTransport
from koboi.llm.openai_adapter import OpenAIAdapter
from koboi.types import AgentResponse


def _mock_transport(response_json, status_code=200):
    transport = HttpTransport(
        base_url="https://api.test.com/v1",
        auth=MagicMock(),
    )
    mock_response = httpx.Response(
        status_code=status_code,
        json=response_json,
        request=httpx.Request("POST", "https://api.test.com/v1/chat/completions"),
    )
    patcher = patch.object(transport._client, "post", return_value=mock_response)
    return transport, patcher


class TestOpenAIAdapterComplete:
    async def test_simple_text_response(self):
        adapter = OpenAIAdapter(model="gpt-4o-mini", transport=MagicMock())
        adapter._transport = MagicMock()
        adapter._transport.post = AsyncMock(
            return_value={
                "id": "chatcmpl-123",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )

        result = await adapter.complete(
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert result.content == "Hello!"
        assert result.tool_calls == []
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.is_complete

    async def test_tool_call_response(self):
        adapter = OpenAIAdapter(model="gpt-4o-mini", transport=MagicMock())
        adapter._transport = MagicMock()
        adapter._transport.post = AsyncMock(
            return_value={
                "id": "chatcmpl-456",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "Jakarta"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 15},
            }
        )

        result = await adapter.complete(
            messages=[{"role": "user", "content": "Weather in Jakarta?"}],
            tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
        )
        assert result.content is None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_abc"
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == '{"city": "Jakarta"}'
        assert not result.is_complete

    async def test_passes_tools_in_request(self):
        adapter = OpenAIAdapter(model="gpt-4o-mini", transport=MagicMock())
        adapter._transport = MagicMock()
        adapter._transport.post = AsyncMock(
            return_value={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {},
            }
        )

        tools = [{"type": "function", "function": {"name": "calc", "parameters": {}}}]
        await adapter.complete(messages=[{"role": "user", "content": "test"}], tools=tools)

        call_body = adapter._transport.post.call_args[0][1]
        assert call_body["tools"] == tools
        assert call_body["tool_choice"] == "auto"

    async def test_omits_tools_when_none(self):
        adapter = OpenAIAdapter(model="gpt-4o-mini", transport=MagicMock())
        adapter._transport = MagicMock()
        adapter._transport.post = AsyncMock(
            return_value={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {},
            }
        )

        await adapter.complete(messages=[{"role": "user", "content": "test"}])
        call_body = adapter._transport.post.call_args[0][1]
        assert "tools" not in call_body
        assert "tool_choice" not in call_body

    async def test_malformed_response_raises_parse_error(self):
        adapter = OpenAIAdapter(model="gpt-4o-mini", transport=MagicMock())
        adapter._transport = MagicMock()
        adapter._transport.post = AsyncMock(return_value={"no_choices_key": True})

        with pytest.raises(LLMResponseParseError, match="Unexpected"):
            await adapter.complete(messages=[{"role": "user", "content": "test"}])

    async def test_empty_choices_raises_parse_error(self):
        adapter = OpenAIAdapter(model="gpt-4o-mini", transport=MagicMock())
        adapter._transport = MagicMock()
        adapter._transport.post = AsyncMock(return_value={"choices": []})

        with pytest.raises(LLMResponseParseError):
            await adapter.complete(messages=[{"role": "user", "content": "test"}])

    async def test_multiple_tool_calls(self):
        adapter = OpenAIAdapter(model="gpt-4o-mini", transport=MagicMock())
        adapter._transport = MagicMock()
        adapter._transport.post = AsyncMock(
            return_value={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {"id": "call_1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                                {"id": "call_2", "type": "function", "function": {"name": "b", "arguments": '{"x":1}'}},
                            ],
                        },
                    }
                ],
                "usage": {},
            }
        )

        result = await adapter.complete(messages=[{"role": "user", "content": "go"}])
        assert len(result.tool_calls) == 2
        assert result.tool_calls[1].name == "b"


class TestOpenAIAdapterEmbeddings:
    async def test_get_embeddings_success(self):
        adapter = OpenAIAdapter(model="gpt-4o-mini", transport=MagicMock())
        adapter._transport = MagicMock()
        adapter._transport.post = AsyncMock(
            return_value={
                "data": [{"embedding": [0.1, 0.2, 0.3]}],
            }
        )

        result = await adapter.get_embeddings("hello")
        assert result == [0.1, 0.2, 0.3]

    async def test_get_embeddings_failure_returns_none(self):
        adapter = OpenAIAdapter(model="gpt-4o-mini", transport=MagicMock())
        adapter._transport = MagicMock()
        adapter._transport.post = AsyncMock(side_effect=Exception("network error"))

        result = await adapter.get_embeddings("hello")
        assert result is None


class TestOpenAIAdapterUsageParsing:
    async def test_no_usage_returns_none(self):
        adapter = OpenAIAdapter(model="gpt-4o-mini", transport=MagicMock())
        adapter._transport = MagicMock()
        adapter._transport.post = AsyncMock(
            return_value={
                "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            }
        )

        result = await adapter.complete(messages=[{"role": "user", "content": "hi"}])
        assert result.usage is None

    async def test_zero_tokens_parsed(self):
        adapter = OpenAIAdapter(model="gpt-4o-mini", transport=MagicMock())
        adapter._transport = MagicMock()
        adapter._transport.post = AsyncMock(
            return_value={
                "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": None, "completion_tokens": None},
            }
        )

        result = await adapter.complete(messages=[{"role": "user", "content": "hi"}])
        assert result.usage.prompt_tokens == 0
        assert result.usage.completion_tokens == 0
