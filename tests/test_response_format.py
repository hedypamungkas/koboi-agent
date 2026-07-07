"""tests/test_response_format.py -- structured output (response_format) threading.

Verifies the provider-agnostic ``response_format`` (a JSON Schema dict) is:
  - wrapped natively by OpenAIAdapter (``body['response_format']``),
  - emulated by AnthropicAdapter via a forced ``tool_use`` collapsed to content,
  - threaded through RetryClient to the underlying impl,
  - passed by the QualityEvaluator and applied by the AgentCore loop on
    tool-less iterations.

asyncio_mode="auto" -> bare ``async def test_*`` (no decorator).
"""

from __future__ import annotations

import json

from koboi.client import RetryClient
from koboi.llm.anthropic_adapter import AnthropicAdapter, _STRUCTURED_TOOL_NAME
from koboi.llm.base import LLMClient
from koboi.llm.openai_adapter import OpenAIAdapter
from koboi.loop import AgentCore
from koboi.memory import ConversationMemory
from koboi.orchestration.orchestrator import QualityEvaluator, _QUALITY_SCHEMA
from koboi.types import AgentResponse

SCHEMA: dict = {
    "type": "object",
    "properties": {"score": {"type": "number"}, "note": {"type": "string"}},
    "required": ["score"],
}


class _FakeTransport:
    """Records the last request body and returns canned response data."""

    def __init__(self, data, stream_lines=None):
        self.data = data
        self.stream_lines = stream_lines or []
        self.last_body = None
        self.base_url = "http://fake"

    async def post(self, path, body):
        self.last_body = body
        return self.data

    async def post_stream(self, path, body):
        self.last_body = body
        for line in self.stream_lines:
            yield line.encode() if isinstance(line, str) else line

    async def close(self):
        pass


class _Recorder(LLMClient):
    """LLMClient that records the response_format it received."""

    def __init__(self):
        self.rf = None

    async def complete(self, messages, tools=None, response_format=None):
        self.rf = response_format
        return AgentResponse(content="ok")

    async def get_embeddings(self, text):
        return None


async def test_openai_adapter_sets_response_format_body():
    data = {"choices": [{"message": {"content": '{"score": 0.9}'}}], "usage": {}}
    transport = _FakeTransport(data)
    adapter = OpenAIAdapter("m", transport)
    await adapter.complete(messages=[{"role": "user", "content": "hi"}], response_format=SCHEMA)

    rf = transport.last_body["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"] == SCHEMA
    assert rf["json_schema"]["name"] == "structured_output"


async def test_openai_adapter_omits_response_format_when_none():
    transport = _FakeTransport({"choices": [{"message": {"content": "x"}}], "usage": {}})
    adapter = OpenAIAdapter("m", transport)
    await adapter.complete(messages=[{"role": "user", "content": "hi"}])
    assert "response_format" not in transport.last_body


async def test_anthropic_adapter_forces_structured_tool_and_collapses():
    # Anthropic returns a tool_use for the forced structured-output tool.
    data = {
        "content": [
            {
                "type": "tool_use",
                "id": "t1",
                "name": _STRUCTURED_TOOL_NAME,
                "input": {"score": 0.9, "note": "good"},
            }
        ],
        "usage": {"input_tokens": 5, "output_tokens": 7},
    }
    transport = _FakeTransport(data)
    adapter = AnthropicAdapter("m", transport)
    result = await adapter.complete(messages=[{"role": "user", "content": "hi"}], response_format=SCHEMA)

    # forced tool_choice to the synthetic structured tool
    assert transport.last_body["tool_choice"] == {"type": "tool", "name": _STRUCTURED_TOOL_NAME}
    # the structured tool is registered with the schema as its input_schema
    structured = [t for t in transport.last_body["tools"] if t["name"] == _STRUCTURED_TOOL_NAME]
    assert structured and structured[0]["input_schema"] == SCHEMA
    # collapsed: content carries the JSON, no leftover tool_calls
    assert json.loads(result.content) == {"score": 0.9, "note": "good"}
    assert result.tool_calls == []


async def test_anthropic_adapter_no_force_when_no_response_format():
    data = {
        "content": [{"type": "text", "text": "hello"}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    transport = _FakeTransport(data)
    adapter = AnthropicAdapter("m", transport)
    result = await adapter.complete(messages=[{"role": "user", "content": "hi"}], tools=None)
    assert "tool_choice" not in transport.last_body
    assert result.content == "hello"


async def test_retry_client_threads_response_format():
    rc = RetryClient.__new__(RetryClient)
    rc._impl = _Recorder()
    rc.max_retries = 0
    rc.retry_backoff_base = 2.0
    await rc.complete(messages=[{"role": "user", "content": "hi"}], response_format=SCHEMA)
    assert rc._impl.rf is SCHEMA


async def test_quality_evaluator_passes_response_format(mock_client):
    client = mock_client(responses=[AgentResponse(content='{"score": 0.8, "feedback": "ok", "needs_revision": false}')])
    evaluator = QualityEvaluator(client)
    score, _feedback, _needs = await evaluator.evaluate("q", "a")
    assert client.last_response_format == _QUALITY_SCHEMA
    assert score == 0.8


async def test_agentcore_loop_applies_response_format_toolless(mock_client):
    client = mock_client(responses=[AgentResponse(content='{"score": 1}')])
    core = AgentCore(
        client=client,
        memory=ConversationMemory(),
        output_schema=SCHEMA,
        max_iterations=1,
    )
    # no tools registered -> tool_defs is None -> response_format is applied
    await core.run("hi")
    assert client.last_response_format == SCHEMA
