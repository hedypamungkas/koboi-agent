"""Tests for koboi.loop module."""

from __future__ import annotations


import pytest

from koboi.memory import ConversationMemory
from koboi.tools.registry import ToolRegistry
from koboi.types import ToolCall
from koboi.exceptions import AgentMaxIterationsError
from tests.conftest import MockClient, make_mock_response, make_mock_tool_call, make_tool_registry


class TestAgentCoreSimpleRun:
    async def test_simple_response(self):
        from koboi.loop import AgentCore

        client = MockClient([make_mock_response("Hello!")])
        agent = AgentCore(client=client, memory=ConversationMemory(), tools=ToolRegistry(), max_iterations=5)
        result = await agent.run("Hi")
        assert result.content == "Hello!"
        assert client.call_count == 1

    async def test_tool_call_then_response(self):
        from koboi.loop import AgentCore

        registry = make_tool_registry()
        tc = make_mock_tool_call("get_weather", {"city": "Jakarta"})
        client = MockClient(
            [
                make_mock_response(None, [tc]),
                make_mock_response("Weather in Jakarta is sunny!"),
            ]
        )
        agent = AgentCore(client=client, memory=ConversationMemory(), tools=registry, max_iterations=5)
        result = await agent.run("What's the weather in Jakarta?")
        assert "Jakarta" in result.content
        assert client.call_count == 2

    async def test_max_iterations(self):
        from koboi.loop import AgentCore

        tc = make_mock_tool_call("get_weather", {"city": "X"})
        client = MockClient([make_mock_response(None, [tc])] * 20)
        agent = AgentCore(client=client, memory=ConversationMemory(), tools=make_tool_registry(), max_iterations=3)
        with pytest.raises(AgentMaxIterationsError):
            await agent.run("test")

    async def test_reset(self):
        from koboi.loop import AgentCore

        client = MockClient([make_mock_response("Hi")])
        agent = AgentCore(client=client, tools=ToolRegistry(), max_iterations=5)
        await agent.run("Hello")
        assert len(agent.memory) > 0
        agent.reset()
        assert len(agent.memory) == 0


class TestAgentCoreWithTools:
    async def test_multiple_tool_calls(self):
        from koboi.loop import AgentCore

        registry = make_tool_registry()
        tc1 = make_mock_tool_call("get_weather", {"city": "Jakarta"})
        tc2 = make_mock_tool_call("calculate", {"expression": "2+2"})
        client = MockClient(
            [
                make_mock_response(None, [tc1, tc2]),
                make_mock_response("Done!"),
            ]
        )
        agent = AgentCore(client=client, memory=ConversationMemory(), tools=registry, max_iterations=5)
        result = await agent.run("Check weather and calculate")
        assert "Done" in result.content

    async def test_tool_not_found(self):
        from koboi.loop import AgentCore

        tc = make_mock_tool_call("nonexistent_tool", {})
        client = MockClient(
            [
                make_mock_response(None, [tc]),
                make_mock_response("OK"),
            ]
        )
        agent = AgentCore(client=client, memory=ConversationMemory(), tools=ToolRegistry(), max_iterations=5)
        result = await agent.run("test")
        assert "OK" in result.content


class TestRunResultEnrichment:
    async def test_elapsed_seconds_is_positive(self):
        from koboi.loop import AgentCore

        client = MockClient([make_mock_response("Hello!")])
        agent = AgentCore(client=client, memory=ConversationMemory(), tools=ToolRegistry(), max_iterations=5)
        result = await agent.run("Hi")
        assert result.elapsed_seconds > 0.0

    async def test_elapsed_seconds_zero_for_instant(self):
        """Even instant runs should have elapsed_seconds >= 0."""
        from koboi.loop import AgentCore

        client = MockClient([make_mock_response("OK")])
        agent = AgentCore(client=client, memory=ConversationMemory(), tools=ToolRegistry(), max_iterations=5)
        result = await agent.run("test")
        assert result.elapsed_seconds >= 0.0

    async def test_tools_used_empty_when_no_tools(self):
        from koboi.loop import AgentCore

        client = MockClient([make_mock_response("No tools used")])
        agent = AgentCore(client=client, memory=ConversationMemory(), tools=ToolRegistry(), max_iterations=5)
        result = await agent.run("test")
        assert result.tools_used == []

    async def test_tools_used_with_single_tool(self):
        from koboi.loop import AgentCore

        registry = make_tool_registry()
        tc = make_mock_tool_call("get_weather", {"city": "Jakarta"})
        client = MockClient(
            [
                make_mock_response(None, [tc]),
                make_mock_response("Weather is sunny!"),
            ]
        )
        agent = AgentCore(client=client, memory=ConversationMemory(), tools=registry, max_iterations=5)
        result = await agent.run("Weather?")
        assert result.tools_used == ["get_weather"]

    async def test_tools_used_deduplicated(self):
        from koboi.loop import AgentCore

        registry = make_tool_registry()
        tc1 = make_mock_tool_call("get_weather", {"city": "Jakarta"})
        tc2 = make_mock_tool_call("get_weather", {"city": "Bandung"})
        client = MockClient(
            [
                make_mock_response(None, [tc1]),
                make_mock_response(None, [tc2]),
                make_mock_response("Done"),
            ]
        )
        agent = AgentCore(client=client, memory=ConversationMemory(), tools=registry, max_iterations=5)
        result = await agent.run("Check both cities")
        assert result.tools_used == ["get_weather"]

    async def test_tools_used_multiple_distinct(self):
        from koboi.loop import AgentCore

        registry = make_tool_registry()
        tc1 = make_mock_tool_call("get_weather", {"city": "Jakarta"})
        tc2 = make_mock_tool_call("calculate", {"expression": "2+2"})
        client = MockClient(
            [
                make_mock_response(None, [tc1, tc2]),
                make_mock_response("Done"),
            ]
        )
        agent = AgentCore(client=client, memory=ConversationMemory(), tools=registry, max_iterations=5)
        result = await agent.run("Weather and calc")
        assert set(result.tools_used) == {"get_weather", "calculate"}

    async def test_model_in_metadata(self):
        from koboi.loop import AgentCore

        client = MockClient([make_mock_response("OK")])
        client.model = "gpt-4o"
        agent = AgentCore(client=client, memory=ConversationMemory(), tools=ToolRegistry(), max_iterations=5)
        result = await agent.run("test")
        assert result.metadata.get("model") == "gpt-4o"

    async def test_token_usage_accumulated(self):
        from koboi.loop import AgentCore

        registry = make_tool_registry()
        tc = make_mock_tool_call("get_weather", {"city": "X"})
        client = MockClient(
            [
                make_mock_response(None, [tc]),
                make_mock_response("Done"),
            ]
        )
        agent = AgentCore(client=client, memory=ConversationMemory(), tools=registry, max_iterations=5)
        result = await agent.run("test")
        assert result.token_usage is not None
        assert result.token_usage.prompt_tokens > 0
        assert result.token_usage.completion_tokens > 0

    async def test_iterations_used(self):
        from koboi.loop import AgentCore

        client = MockClient([make_mock_response("One shot")])
        agent = AgentCore(client=client, memory=ConversationMemory(), tools=ToolRegistry(), max_iterations=5)
        result = await agent.run("test")
        assert result.iterations_used == 1

    async def test_iterations_used_with_tool(self):
        from koboi.loop import AgentCore

        registry = make_tool_registry()
        tc = make_mock_tool_call("get_weather", {"city": "X"})
        client = MockClient(
            [
                make_mock_response(None, [tc]),
                make_mock_response("Done"),
            ]
        )
        agent = AgentCore(client=client, memory=ConversationMemory(), tools=registry, max_iterations=5)
        result = await agent.run("test")
        assert result.iterations_used == 2


class TestRunResultDirect:
    """Test RunResult dataclass properties directly."""

    def test_tools_used_empty(self):
        from koboi.types import RunResult

        r = RunResult(content="hi")
        assert r.tools_used == []

    def test_tools_used_deduplication(self):
        from koboi.types import RunResult

        tc1 = ToolCall(id="1", name="web_search", arguments="{}")
        tc2 = ToolCall(id="2", name="calculator", arguments="{}")
        tc3 = ToolCall(id="3", name="web_search", arguments="{}")
        r = RunResult(content="hi", tool_calls_made=[tc1, tc2, tc3])
        assert r.tools_used == ["web_search", "calculator"]

    def test_model_from_metadata(self):
        from koboi.types import RunResult

        r = RunResult(content="hi", metadata={"model": "gpt-4o"})
        assert r.model == "gpt-4o"

    def test_model_empty_when_no_metadata(self):
        from koboi.types import RunResult

        r = RunResult(content="hi")
        assert r.model == ""

    def test_elapsed_seconds_default(self):
        from koboi.types import RunResult

        r = RunResult(content="hi")
        assert r.elapsed_seconds == 0.0

    def test_elapsed_seconds_set(self):
        from koboi.types import RunResult

        r = RunResult(content="hi", elapsed_seconds=3.14)
        assert r.elapsed_seconds == 3.14

    def test_str_returns_content(self):
        from koboi.types import RunResult

        r = RunResult(content="hello world")
        assert str(r) == "hello world"
