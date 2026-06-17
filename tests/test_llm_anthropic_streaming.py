"""Tests for Anthropic adapter streaming and uncovered paths."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from koboi.llm.anthropic_adapter import AnthropicAdapter
from koboi.types import AgentResponse, ToolCall, TokenUsage


class MockTransport:
    """Mock HttpTransport for testing adapters."""

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


def _sse_line(data: dict) -> bytes:
    return f"data: {json.dumps(data)}\n".encode()


class TestAnthropicAdapterComplete:
    async def test_basic_complete(self):
        transport = MockTransport(post_response={
            "content": [{"type": "text", "text": "Hello!"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })
        adapter = AnthropicAdapter(model="claude-sonnet-4-20250514", transport=transport)
        result = await adapter.complete(
            messages=[{"role": "user", "content": "Hi"}],
            tools=None,
        )
        assert result.content == "Hello!"
        assert result.usage.prompt_tokens == 10

    async def test_complete_with_tools(self):
        transport = MockTransport(post_response={
            "content": [{"type": "tool_use", "id": "tc1", "name": "calc", "input": {"expr": "1+1"}}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })
        adapter = AnthropicAdapter(model="claude-sonnet-4-20250514", transport=transport)
        result = await adapter.complete(
            messages=[{"role": "user", "content": "calc"}],
            tools=[{"type": "function", "function": {"name": "calc", "description": "calc", "parameters": {}}}],
        )
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "calc"

    async def test_complete_with_system_message(self):
        transport = MockTransport(post_response={
            "content": [{"type": "text", "text": "ok"}],
        })
        adapter = AnthropicAdapter(model="claude-sonnet-4-20250514", transport=transport)
        result = await adapter.complete(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
        )
        assert transport.last_post_body["system"] == "You are helpful."

    async def test_complete_with_temperature(self):
        transport = MockTransport(post_response={
            "content": [{"type": "text", "text": "ok"}],
        })
        adapter = AnthropicAdapter(model="claude-sonnet-4-20250514", transport=transport, temperature=0.5)
        await adapter.complete(messages=[{"role": "user", "content": "Hi"}])
        assert transport.last_post_body["temperature"] == 0.5

    async def test_complete_with_logger(self):
        transport = MockTransport(post_response={
            "content": [{"type": "text", "text": "ok"}],
        })
        logger = MagicMock()
        adapter = AnthropicAdapter(model="claude-sonnet-4-20250514", transport=transport, logger=logger)
        await adapter.complete(messages=[{"role": "user", "content": "Hi"}])
        logger.log_llm_request.assert_called_once()
        logger.log_llm_response.assert_called_once()

    async def test_complete_empty_response(self):
        transport = MockTransport(post_response={
            "content": [],
        })
        adapter = AnthropicAdapter(model="claude-sonnet-4-20250514", transport=transport)
        result = await adapter.complete(messages=[{"role": "user", "content": "Hi"}])
        assert result.content is None


class TestAnthropicAdapterStreaming:
    async def test_text_streaming(self):
        lines = [
            b"event: message_start\n",
            _sse_line({"type": "message_start", "message": {"usage": {"input_tokens": 10}}}),
            _sse_line({"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}),
            _sse_line({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}),
            _sse_line({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " world"}}),
            _sse_line({"type": "content_block_stop", "index": 0}),
            _sse_line({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}}),
            _sse_line({"type": "message_stop"}),
        ]
        transport = MockTransport(stream_lines=lines)
        adapter = AnthropicAdapter(model="claude-sonnet-4-20250514", transport=transport)

        events = []
        async for event in adapter.complete_stream(messages=[{"role": "user", "content": "Hi"}]):
            events.append(event)

        from koboi.events import TextDeltaEvent, CompleteEvent
        text_events = [e for e in events if isinstance(e, TextDeltaEvent)]
        assert len(text_events) == 2
        assert text_events[0].content == "Hello"
        assert text_events[1].content == " world"

        complete_events = [e for e in events if isinstance(e, CompleteEvent)]
        assert len(complete_events) == 1
        assert complete_events[0].response.content == "Hello world"

    async def test_tool_call_streaming(self):
        lines = [
            _sse_line({"type": "message_start", "message": {"usage": {"input_tokens": 10}}}),
            _sse_line({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "tc1", "name": "calc"}}),
            _sse_line({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"ex'}}),
            _sse_line({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": 'pr":"1+1"}'}}),
            _sse_line({"type": "content_block_stop", "index": 0}),
            _sse_line({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}}),
            _sse_line({"type": "message_stop"}),
        ]
        transport = MockTransport(stream_lines=lines)
        adapter = AnthropicAdapter(model="claude-sonnet-4-20250514", transport=transport)

        events = []
        async for event in adapter.complete_stream(messages=[{"role": "user", "content": "calc"}]):
            events.append(event)

        from koboi.events import ToolCallEvent
        tc_events = [e for e in events if isinstance(e, ToolCallEvent)]
        assert len(tc_events) == 1
        assert tc_events[0].tool_name == "calc"

    async def test_streaming_with_tools_param(self):
        lines = [
            _sse_line({"type": "message_start", "message": {"usage": {"input_tokens": 10}}}),
            _sse_line({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}}),
            _sse_line({"type": "message_stop"}),
        ]
        transport = MockTransport(stream_lines=lines)
        adapter = AnthropicAdapter(model="claude-sonnet-4-20250514", transport=transport, temperature=0.3)

        events = []
        async for event in adapter.complete_stream(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[{"type": "function", "function": {"name": "t", "description": "t", "parameters": {}}}],
        ):
            events.append(event)

        assert transport.last_post_body["stream"] is True
        assert transport.last_post_body["temperature"] == 0.3
        assert "tools" in transport.last_post_body

    async def test_streaming_with_system_message(self):
        lines = [
            _sse_line({"type": "message_start", "message": {"usage": {"input_tokens": 10}}}),
            _sse_line({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}}),
            _sse_line({"type": "message_stop"}),
        ]
        transport = MockTransport(stream_lines=lines)
        adapter = AnthropicAdapter(model="claude-sonnet-4-20250514", transport=transport)

        events = []
        async for event in adapter.complete_stream(
            messages=[
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hi"},
            ],
        ):
            events.append(event)

        assert transport.last_post_body["system"] == "Be helpful"

    async def test_streaming_with_logger(self):
        lines = [
            _sse_line({"type": "message_start", "message": {"usage": {"input_tokens": 10}}}),
            _sse_line({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}}),
            _sse_line({"type": "message_stop"}),
        ]
        transport = MockTransport(stream_lines=lines)
        logger = MagicMock()
        adapter = AnthropicAdapter(model="claude-sonnet-4-20250514", transport=transport, logger=logger)

        events = []
        async for event in adapter.complete_stream(messages=[{"role": "user", "content": "Hi"}]):
            events.append(event)

        logger.log_llm_request.assert_called_once()
        logger.log_llm_response.assert_called_once()


class TestAnthropicAdapterHelpers:
    def test_extract_system(self):
        messages = [
            {"role": "system", "content": "sys1"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "sys2"},
        ]
        system, remaining = AnthropicAdapter._extract_system(messages)
        assert system == "sys1\n\nsys2"
        assert len(remaining) == 1

    def test_translate_user_content_text(self):
        blocks = [{"type": "text", "text": "hello"}]
        result = AnthropicAdapter._translate_user_content(blocks)
        assert result == [{"type": "text", "text": "hello"}]

    def test_translate_user_content_image_url(self):
        blocks = [{"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}]
        result = AnthropicAdapter._translate_user_content(blocks)
        assert result[0]["type"] == "image"
        assert result[0]["source"]["type"] == "url"

    def test_translate_user_content_image_base64(self):
        blocks = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}}]
        result = AnthropicAdapter._translate_user_content(blocks)
        assert result[0]["source"]["type"] == "base64"
        assert result[0]["source"]["media_type"] == "image/png"

    def test_translate_user_content_non_dict(self):
        blocks = ["plain text"]
        result = AnthropicAdapter._translate_user_content(blocks)
        assert result[0]["type"] == "text"

    def test_translate_assistant_with_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": "thinking",
            "tool_calls": [
                {"id": "tc1", "function": {"name": "calc", "arguments": '{"x": 1}'}},
            ],
        }
        result = AnthropicAdapter._translate_assistant(msg)
        assert result["role"] == "assistant"
        assert len(result["content"]) == 2  # text + tool_use

    def test_translate_assistant_empty(self):
        msg = {"role": "assistant"}
        result = AnthropicAdapter._translate_assistant(msg)
        assert result["content"][0]["type"] == "text"

    def test_collect_tool_results(self):
        messages = [
            {"role": "tool", "tool_call_id": "tc1", "content": "result1"},
            {"role": "tool", "tool_call_id": "tc2", "content": "result2"},
            {"role": "user", "content": "next"},
        ]
        blocks, consumed = AnthropicAdapter._collect_tool_results(messages, 0)
        assert consumed == 2
        assert len(blocks) == 2

    def test_ensure_alternating_roles_merge(self):
        messages = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "c"},
        ]
        result = AnthropicAdapter._ensure_alternating_roles(messages)
        assert len(result) == 2
        assert result[0]["content"] == "a\nb"

    def test_ensure_alternating_starts_with_assistant(self):
        messages = [{"role": "assistant", "content": "hello"}]
        result = AnthropicAdapter._ensure_alternating_roles(messages)
        assert result[0]["role"] == "user"

    def test_ensure_alternating_empty(self):
        result = AnthropicAdapter._ensure_alternating_roles([])
        assert result == []

    def test_translate_tools(self):
        tools = [{"type": "function", "function": {"name": "calc", "description": "calc", "parameters": {"type": "object"}}}]
        result = AnthropicAdapter._translate_tools(tools)
        assert result[0]["name"] == "calc"
        assert result[0]["input_schema"]["type"] == "object"

    def test_parse_response_text_and_tools(self):
        data = {
            "content": [
                {"type": "text", "text": "I'll help"},
                {"type": "tool_use", "id": "tc1", "name": "calc", "input": {"x": 1}},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = AnthropicAdapter._parse_response(data)
        assert result.content == "I'll help"
        assert len(result.tool_calls) == 1
        assert result.usage.prompt_tokens == 10

    def test_parse_response_empty(self):
        data = {"content": []}
        result = AnthropicAdapter._parse_response(data)
        assert result.content is None

    async def test_get_embeddings_returns_none(self):
        transport = MockTransport()
        adapter = AnthropicAdapter(model="claude-sonnet-4-20250514", transport=transport)
        assert await adapter.get_embeddings("test") is None
