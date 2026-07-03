"""koboi/eval/t/mock.py -- Scripted LLM client for deterministic, API-key-free eval tests."""

from __future__ import annotations

import json

from koboi.llm.base import LLMClient
from koboi.types import AgentResponse, TokenUsage, ToolCall


class ScriptedClient(LLMClient):
    """Deterministic LLM client that serves scripted ``AgentResponse`` objects in order.

    Mirrors ``tests/conftest.py:MockClient`` so ``.eval.py`` tests run without an
    API key. Once the script is exhausted it returns a terminal empty response so
    the agent loop exits cleanly instead of hanging.
    """

    def __init__(self, responses: list[AgentResponse], *, model: str = "scripted"):
        self._responses = list(responses)
        self._index = 0
        self._model = model
        self.call_count = 0

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, messages: list[dict], tools: list[dict] | None = None) -> AgentResponse:
        self.call_count += 1
        if self._index < len(self._responses):
            response = self._responses[self._index]
            self._index += 1
            return response
        return AgentResponse(content="No more scripted responses", tool_calls=[])

    async def get_embeddings(self, text: str) -> list[float] | None:
        return None


def scripted_response(
    content: str | None = None,
    tool_calls: list[ToolCall] | None = None,
    *,
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> AgentResponse:
    """Build a scripted :class:`~koboi.types.AgentResponse` for ``MOCK_RESPONSES``."""
    return AgentResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage=TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def scripted_tool_call(name: str, arguments: dict | None = None) -> ToolCall:
    """Build a scripted :class:`~koboi.types.ToolCall` (arguments JSON-encoded)."""
    return ToolCall(id=f"tc_{name}", name=name, arguments=json.dumps(arguments or {}))
