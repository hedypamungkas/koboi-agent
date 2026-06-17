"""Tests for OpenAI adapter streaming and uncovered paths."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from koboi.llm.openai_adapter import OpenAIAdapter
from koboi.types import AgentResponse, ToolCall, TokenUsage


class MockTransport:
    def __init__(self, post_response=None, stream_lines=None):
        self._post_response = post_response or {}
        self._stream_lines = stream_lines or []
        self.last_post_url = None
        self.last_post_body = None

    async def post(self, url, body):
        self.last_post_url = url
        self.last_post_body = body
        return self._post_response

    async def post_stream(self, url, body):
        self.last_post_url = url
        self.last_post_body = body
        for line in self._stream_lines:
            yield line

    async def close(self):
        pass


def _sse_data(data: dict) -> bytes:
    return f"data: {json.dumps(data)}\n".encode()


class TestOpenAIAdapterComplete:
    async def test_basic_complete(self):
        transport = MockTransport(
            post_response={
                "choices": [{"message": {"content": "Hello!"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )
        adapter = OpenAIAdapter(model="gpt-4o", transport=transport)
        result = await adapter.complete(messages=[{"role": "user", "content": "Hi"}])
        assert result.content == "Hello!"
        assert result.usage.prompt_tokens == 10

    async def test_complete_with_tools(self):
        transport = MockTransport(
            post_response={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [{"id": "tc1", "function": {"name": "calc", "arguments": '{"x": 1}'}}],
                        }
                    }
                ],
            }
        )
        adapter = OpenAIAdapter(model="gpt-4o", transport=transport)
        result = await adapter.complete(
            messages=[{"role": "user", "content": "calc"}],
            tools=[{"type": "function", "function": {"name": "calc", "description": "calc", "parameters": {}}}],
        )
        assert len(result.tool_calls) == 1
        assert transport.last_post_body["tool_choice"] == "auto"

    async def test_complete_with_temperature(self):
        transport = MockTransport(
            post_response={
                "choices": [{"message": {"content": "ok"}}],
            }
        )
        adapter = OpenAIAdapter(model="gpt-4o", transport=transport, temperature=0.7)
        await adapter.complete(messages=[{"role": "user", "content": "Hi"}])
        assert transport.last_post_body["temperature"] == 0.7

    async def test_complete_with_logger(self):
        transport = MockTransport(
            post_response={
                "choices": [{"message": {"content": "ok"}}],
            }
        )
        logger = MagicMock()
        adapter = OpenAIAdapter(model="gpt-4o", transport=transport, logger=logger)
        await adapter.complete(messages=[{"role": "user", "content": "Hi"}])
        logger.log_llm_request.assert_called_once()
        logger.log_llm_response.assert_called_once()

    async def test_complete_bad_response_raises(self):
        from koboi.llm.base import LLMResponseParseError

        transport = MockTransport(post_response={"unexpected": "data"})
        adapter = OpenAIAdapter(model="gpt-4o", transport=transport)
        with pytest.raises(LLMResponseParseError):
            await adapter.complete(messages=[{"role": "user", "content": "Hi"}])


class TestOpenAIAdapterStreaming:
    async def test_text_streaming(self):
        lines = [
            _sse_data({"choices": [{"delta": {"content": "Hello"}}]}),
            _sse_data({"choices": [{"delta": {"content": " world"}}]}),
            _sse_data({"choices": [{}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}),
            b"data: [DONE]\n",
        ]
        transport = MockTransport(stream_lines=lines)
        adapter = OpenAIAdapter(model="gpt-4o", transport=transport)

        events = []
        async for event in adapter.complete_stream(messages=[{"role": "user", "content": "Hi"}]):
            events.append(event)

        from koboi.events import TextDeltaEvent, CompleteEvent

        text_events = [e for e in events if isinstance(e, TextDeltaEvent)]
        assert len(text_events) == 2
        assert text_events[0].content == "Hello"

        complete_events = [e for e in events if isinstance(e, CompleteEvent)]
        assert len(complete_events) == 1
        assert complete_events[0].response.content == "Hello world"

    async def test_tool_call_streaming(self):
        lines = [
            _sse_data(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "id": "tc1", "function": {"name": "calc"}},
                                ]
                            }
                        }
                    ]
                }
            ),
            _sse_data(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": '{"x":'}},
                                ]
                            }
                        }
                    ]
                }
            ),
            _sse_data(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": "1}"}},
                                ]
                            }
                        }
                    ]
                }
            ),
            b"data: [DONE]\n",
        ]
        transport = MockTransport(stream_lines=lines)
        adapter = OpenAIAdapter(model="gpt-4o", transport=transport)

        events = []
        async for event in adapter.complete_stream(messages=[{"role": "user", "content": "calc"}]):
            events.append(event)

        from koboi.events import ToolCallEvent

        tc_events = [e for e in events if isinstance(e, ToolCallEvent)]
        assert len(tc_events) == 1
        assert tc_events[0].tool_name == "calc"

    async def test_streaming_with_temperature(self):
        lines = [b"data: [DONE]\n"]
        transport = MockTransport(stream_lines=lines)
        adapter = OpenAIAdapter(model="gpt-4o", transport=transport, temperature=0.5)
        async for _ in adapter.complete_stream(messages=[{"role": "user", "content": "Hi"}]):
            pass
        assert transport.last_post_body["temperature"] == 0.5

    async def test_streaming_with_tools(self):
        lines = [b"data: [DONE]\n"]
        transport = MockTransport(stream_lines=lines)
        adapter = OpenAIAdapter(model="gpt-4o", transport=transport)
        async for _ in adapter.complete_stream(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[{"type": "function", "function": {"name": "t", "description": "t", "parameters": {}}}],
        ):
            pass
        assert transport.last_post_body["tool_choice"] == "auto"

    async def test_streaming_with_logger(self):
        lines = [
            _sse_data({"choices": [{"delta": {"content": "ok"}}]}),
            b"data: [DONE]\n",
        ]
        transport = MockTransport(stream_lines=lines)
        logger = MagicMock()
        adapter = OpenAIAdapter(model="gpt-4o", transport=transport, logger=logger)
        async for _ in adapter.complete_stream(messages=[{"role": "user", "content": "Hi"}]):
            pass
        logger.log_llm_request.assert_called_once()
        logger.log_llm_response.assert_called_once()

    async def test_streaming_skips_non_data_lines(self):
        lines = [
            b": comment\n",
            b"\n",
            _sse_data({"choices": [{"delta": {"content": "ok"}}]}),
            b"data: [DONE]\n",
        ]
        transport = MockTransport(stream_lines=lines)
        adapter = OpenAIAdapter(model="gpt-4o", transport=transport)
        events = []
        async for event in adapter.complete_stream(messages=[{"role": "user", "content": "Hi"}]):
            events.append(event)
        from koboi.events import TextDeltaEvent

        assert any(isinstance(e, TextDeltaEvent) for e in events)

    async def test_streaming_skips_invalid_json(self):
        lines = [
            b"data: {invalid json\n",
            _sse_data({"choices": [{"delta": {"content": "ok"}}]}),
            b"data: [DONE]\n",
        ]
        transport = MockTransport(stream_lines=lines)
        adapter = OpenAIAdapter(model="gpt-4o", transport=transport)
        events = []
        async for event in adapter.complete_stream(messages=[{"role": "user", "content": "Hi"}]):
            events.append(event)
        from koboi.events import TextDeltaEvent

        text_events = [e for e in events if isinstance(e, TextDeltaEvent)]
        assert len(text_events) == 1


class TestOpenAIAdapterHelpers:
    def test_parse_tool_calls(self):
        raw = [{"id": "tc1", "function": {"name": "calc", "arguments": '{"x": 1}'}}]
        result = OpenAIAdapter._parse_tool_calls(raw)
        assert len(result) == 1
        assert result[0].name == "calc"

    def test_parse_tool_calls_empty(self):
        assert OpenAIAdapter._parse_tool_calls(None) == []
        assert OpenAIAdapter._parse_tool_calls([]) == []

    def test_parse_usage(self):
        result = OpenAIAdapter._parse_usage({"prompt_tokens": 10, "completion_tokens": 5})
        assert result.prompt_tokens == 10

    def test_parse_usage_none(self):
        assert OpenAIAdapter._parse_usage(None) is None

    async def test_get_embeddings_success(self):
        transport = MockTransport(
            post_response={
                "data": [{"embedding": [0.1, 0.2, 0.3]}],
            }
        )
        adapter = OpenAIAdapter(model="gpt-4o", transport=transport)
        result = await adapter.get_embeddings("test text")
        assert result == [0.1, 0.2, 0.3]

    async def test_get_embeddings_failure(self):
        transport = MockTransport(post_response=None)
        transport.post = None  # Will cause error

        async def fail_post(url, body):
            raise Exception("405 error")

        transport.post = fail_post
        adapter = OpenAIAdapter(model="gpt-4o", transport=transport)
        result = await adapter.get_embeddings("test")
        assert result is None
