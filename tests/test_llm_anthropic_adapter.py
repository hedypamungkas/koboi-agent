"""Tests for koboi.llm.anthropic_adapter module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, AsyncMock


from koboi.llm.anthropic_adapter import AnthropicAdapter


def _make_adapter() -> AnthropicAdapter:
    adapter = AnthropicAdapter(model="claude-sonnet-4-20250514", transport=MagicMock())
    return adapter


class TestExtractSystem:
    def test_extracts_system_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        system, remaining = AnthropicAdapter._extract_system(messages)
        assert system == "You are helpful."
        assert remaining == [{"role": "user", "content": "Hello"}]

    def test_concats_multiple_system_messages(self):
        messages = [
            {"role": "system", "content": "Part 1"},
            {"role": "system", "content": "Part 2"},
            {"role": "user", "content": "Hi"},
        ]
        system, remaining = AnthropicAdapter._extract_system(messages)
        assert system == "Part 1\n\nPart 2"
        assert len(remaining) == 1

    def test_no_system_messages(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        system, remaining = AnthropicAdapter._extract_system(messages)
        assert system == ""
        assert len(remaining) == 2

    def test_empty_system_content_skipped(self):
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hello"},
        ]
        system, remaining = AnthropicAdapter._extract_system(messages)
        assert system == ""


class TestTranslateMessages:
    def test_simple_user_assistant(self):
        adapter = _make_adapter()
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."},
        ]
        result = adapter._translate_messages(messages)
        assert result[0] == {"role": "user", "content": "What is 2+2?"}
        assert result[1] == {"role": "assistant", "content": [{"type": "text", "text": "The answer is 4."}]}

    def test_assistant_with_tool_calls(self):
        adapter = _make_adapter()
        messages = [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"SF"}'},
                    },
                ],
            },
        ]
        result = adapter._translate_messages(messages)
        assert len(result) == 2
        assistant_msg = result[1]
        assert assistant_msg["role"] == "assistant"
        content_blocks = assistant_msg["content"]
        assert content_blocks[0] == {"type": "text", "text": "Let me check."}
        assert content_blocks[1]["type"] == "tool_use"
        assert content_blocks[1]["id"] == "call_1"
        assert content_blocks[1]["name"] == "get_weather"
        assert content_blocks[1]["input"] == {"city": "SF"}

    def test_tool_result_merged_into_user(self):
        adapter = _make_adapter()
        messages = [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "Checking.",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 25C"},
        ]
        result = adapter._translate_messages(messages)
        assert len(result) == 3
        tool_result_msg = result[2]
        assert tool_result_msg["role"] == "user"
        assert tool_result_msg["content"][0]["type"] == "tool_result"
        assert tool_result_msg["content"][0]["tool_use_id"] == "call_1"
        assert tool_result_msg["content"][0]["content"] == "Sunny, 25C"

    def test_consecutive_tool_results_merged(self):
        adapter = _make_adapter()
        messages = [
            {"role": "user", "content": "Compare"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                    {"id": "call_2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Result A"},
            {"role": "tool", "tool_call_id": "call_2", "content": "Result B"},
        ]
        result = adapter._translate_messages(messages)
        tool_result_msg = result[2]
        assert tool_result_msg["role"] == "user"
        assert len(tool_result_msg["content"]) == 2
        assert tool_result_msg["content"][0]["tool_use_id"] == "call_1"
        assert tool_result_msg["content"][1]["tool_use_id"] == "call_2"

    def test_ensures_alternating_roles(self):
        adapter = _make_adapter()
        messages = [
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Hello"},
        ]
        result = adapter._translate_messages(messages)
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "."
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "user"

    def test_merges_consecutive_same_role_strings(self):
        adapter = _make_adapter()
        messages = [
            {"role": "user", "content": "Part 1"},
            {"role": "user", "content": "Part 2"},
        ]
        result = adapter._translate_messages(messages)
        assert len(result) == 1
        assert "Part 1" in result[0]["content"]
        assert "Part 2" in result[0]["content"]

    def test_empty_messages(self):
        adapter = _make_adapter()
        result = adapter._translate_messages([])
        assert result == []

    def test_assistant_no_content_no_tool_calls(self):
        adapter = _make_adapter()
        messages = [{"role": "assistant", "content": None}]
        result = adapter._translate_messages(messages)
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == [{"type": "text", "text": ""}]

    def test_invalid_arguments_json_handled(self):
        adapter = _make_adapter()
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "x", "arguments": "not-valid-json"}},
                ],
            }
        ]
        result = adapter._translate_messages(messages)
        assert result[0]["role"] == "user"
        tool_use_block = result[1]["content"][0]
        assert tool_use_block["input"] == {}


class TestTranslateTools:
    def test_converts_openai_to_anthropic_format(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Do math",
                    "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}},
                },
            },
        ]
        result = AnthropicAdapter._translate_tools(tools)
        assert len(result) == 1
        assert result[0]["name"] == "calculator"
        assert result[0]["description"] == "Do math"
        assert result[0]["input_schema"] == {"type": "object", "properties": {"expr": {"type": "string"}}}

    def test_missing_function_fields_get_defaults(self):
        tools = [{"type": "function", "function": {}}]
        result = AnthropicAdapter._translate_tools(tools)
        assert result[0]["name"] == ""
        assert result[0]["description"] == ""
        assert "input_schema" in result[0]


class TestParseResponse:
    def test_text_only_response(self):
        adapter = _make_adapter()
        data = {
            "id": "msg_123",
            "content": [{"type": "text", "text": "Hello from Claude!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 15, "output_tokens": 10},
        }
        result = adapter._parse_response(data)
        assert result.content == "Hello from Claude!"
        assert result.tool_calls == []
        assert result.usage.prompt_tokens == 15
        assert result.usage.completion_tokens == 10
        assert result.is_complete

    def test_tool_use_response(self):
        adapter = _make_adapter()
        data = {
            "id": "msg_456",
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_abc", "name": "search", "input": {"query": "koboi"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 30, "output_tokens": 20},
        }
        result = adapter._parse_response(data)
        assert "Let me check." in result.content
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "toolu_abc"
        assert result.tool_calls[0].name == "search"
        assert json.loads(result.tool_calls[0].arguments) == {"query": "koboi"}
        assert not result.is_complete

    def test_multiple_tool_use_blocks(self):
        adapter = _make_adapter()
        data = {
            "content": [
                {"type": "tool_use", "id": "t1", "name": "a", "input": {"x": 1}},
                {"type": "tool_use", "id": "t2", "name": "b", "input": {"y": 2}},
            ],
            "usage": {},
        }
        result = adapter._parse_response(data)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "a"
        assert result.tool_calls[1].name == "b"

    def test_empty_content(self):
        adapter = _make_adapter()
        data = {"content": [], "usage": {}}
        result = adapter._parse_response(data)
        assert result.content is None
        assert result.tool_calls == []

    def test_no_usage(self):
        adapter = _make_adapter()
        data = {"content": [{"type": "text", "text": "Hi"}]}
        result = adapter._parse_response(data)
        assert result.usage is None

    def test_multiple_text_blocks_concatenated(self):
        adapter = _make_adapter()
        data = {
            "content": [
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": "Part 2"},
            ],
            "usage": {},
        }
        result = adapter._parse_response(data)
        assert result.content == "Part 1\nPart 2"


class TestAnthropicAdapterComplete:
    async def test_full_flow_builds_correct_request(self):
        adapter = _make_adapter()
        adapter._transport = MagicMock()
        adapter._transport.post = AsyncMock(
            return_value={
                "content": [{"type": "text", "text": "Response"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        )

        result = await adapter.complete(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "calc", "description": "math", "parameters": {"type": "object"}},
                }
            ],
        )

        call_args = adapter._transport.post.call_args
        path = call_args[0][0]
        body = call_args[0][1]

        assert path == "/messages"
        assert body["model"] == "claude-sonnet-4-20250514"
        assert body["max_tokens"] == 4096
        assert body["system"] == "You are helpful."
        assert len(body["messages"]) == 1
        assert body["messages"][0] == {"role": "user", "content": "Hello"}
        assert body["tools"][0]["name"] == "calc"
        assert body["tools"][0]["input_schema"] == {"type": "object"}
        assert body["tool_choice"] == {"type": "auto"}

        assert result.content == "Response"
        assert result.is_complete

    async def test_no_system_no_tools(self):
        adapter = _make_adapter()
        adapter._transport = MagicMock()
        adapter._transport.post = AsyncMock(
            return_value={
                "content": [{"type": "text", "text": "Hi"}],
                "usage": {},
            }
        )

        await adapter.complete(messages=[{"role": "user", "content": "Hello"}])

        body = adapter._transport.post.call_args[0][1]
        assert "system" not in body
        assert "tools" not in body


class TestAnthropicAdapterEmbeddings:
    async def test_returns_none(self):
        adapter = _make_adapter()
        assert await adapter.get_embeddings("text") is None
