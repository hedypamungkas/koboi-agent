"""Shared test fixtures for koboi-agent tests."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from koboi.types import AgentResponse, ToolCall, TokenUsage, ToolDefinition, RiskLevel, RunResult
from koboi.llm.base import LLMClient
from koboi.memory import ConversationMemory
from koboi.tools.registry import ToolRegistry
from koboi.config import Config


class MockClient(LLMClient):
    """Test double for Client that returns predetermined responses."""

    def __init__(self, responses: list[AgentResponse] | None = None):
        self.responses = responses or []
        self._index = 0
        self.call_count = 0
        self.last_messages = None
        self.last_tools = None
        self._model = "mock-model"

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        self._model = value

    async def complete(self, messages: list[dict], tools: list[dict] | None = None) -> AgentResponse:
        self.call_count += 1
        self.last_messages = messages
        self.last_tools = tools
        if self._index < len(self.responses):
            resp = self.responses[self._index]
            self._index += 1
            return resp
        return AgentResponse(content="No more responses", tool_calls=[])

    async def complete_stream(self, messages: list[dict], tools: list[dict] | None = None):
        from koboi.events import TextDeltaEvent, CompleteEvent

        resp = await self.complete(messages, tools)
        if resp.content:
            yield TextDeltaEvent(content=resp.content)
        yield CompleteEvent(response=resp, content=resp.content or "")

    async def get_embeddings(self, text: str) -> list[float] | None:
        return None

    async def close(self):
        pass


class MockToolCall:
    """Test double for ToolCall."""

    def __init__(self, id: str = "tc_1", name: str = "test_tool", arguments: str = "{}"):
        self.id = id
        self.name = name
        self.arguments = arguments


def make_mock_response(content: str | None = None, tool_calls: list[ToolCall] | None = None) -> AgentResponse:
    return AgentResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
    )


def make_mock_tool_call(name: str = "test_tool", arguments: dict | None = None) -> ToolCall:
    return ToolCall(
        id=f"tc_{name}",
        name=name,
        arguments=json.dumps(arguments or {}),
    )


def make_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        name="get_weather",
        description="Get weather for a city",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        fn=lambda city: f"Weather in {city}: Sunny, 28°C",
    )
    registry.register(
        name="calculate",
        description="Calculate a math expression",
        parameters={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        fn=lambda expression: str(eval(expression)),
    )
    return registry


@pytest.fixture
def mock_client():
    return MockClient


@pytest.fixture
def tool_registry():
    return make_tool_registry()


@pytest.fixture
def memory():
    return ConversationMemory()


@pytest.fixture
def simple_config(tmp_path):
    config_data = {
        "agent": {"name": "test-agent", "max_iterations": 5, "system_prompt": "You are helpful."},
        "llm": {"model": "gpt-4o-mini", "api_key": "test-key", "base_url": "http://localhost:8080/v1"},
    }
    config_path = tmp_path / "test_config.yaml"
    import yaml

    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return Config.from_yaml(config_path)
