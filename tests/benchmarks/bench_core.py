"""Core performance benchmarks for koboi-agent."""
import tempfile
from pathlib import Path

import pytest
import yaml

from koboi.config import Config
from koboi.memory import ConversationMemory
from koboi.tools.registry import ToolRegistry
from koboi.tokens import estimate_tokens, estimate_single
from koboi.context.manager import TruncationManager


def test_config_loading(benchmark, sample_config_dict):
    """Benchmark loading a YAML config file."""

    def load_config():
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            yaml.dump(sample_config_dict, f)
            config_path = f.name
        return Config.from_yaml(config_path)

    result = benchmark(load_config)
    assert result.agent_name == "benchmark-agent"


def test_facade_creation(benchmark, sample_config_dict):
    """Benchmark creating KoboiAgent.from_config()."""
    from koboi.facade import KoboiAgent

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        yaml.dump(sample_config_dict, f)
        config_path = f.name

    def create_agent():
        return KoboiAgent.from_config(config_path, verbose=False)

    result = benchmark(create_agent)
    assert result.config.agent_name == "benchmark-agent"


def test_tool_registration(benchmark):
    """Benchmark registering 10 tools."""

    def register_tools():
        registry = ToolRegistry()
        for i in range(10):
            registry.register(
                name=f"tool_{i}",
                description=f"Tool number {i}",
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
                fn=lambda x: f"Result: {x}",
            )
        return registry

    result = benchmark(register_tools)
    assert len(result._tools) == 10


def test_memory_add(benchmark, sample_messages):
    """Benchmark adding messages to ConversationMemory."""

    def add_messages():
        memory = ConversationMemory()
        for msg in sample_messages:
            if msg["role"] == "user":
                memory.add_user_message(msg.get("content", ""))
            elif msg["role"] == "assistant":
                memory.add_assistant_message(msg.get("content", ""))
            elif msg["role"] == "system":
                memory.add_context_message(msg.get("content", ""))
        return memory

    result = benchmark(add_messages)
    assert len(result.get_messages()) == len(sample_messages)


def test_memory_get(benchmark, sample_messages):
    """Benchmark getting messages from populated memory."""
    memory = ConversationMemory()
    for msg in sample_messages:
        if msg["role"] == "user":
            memory.add_user_message(msg.get("content", ""))
        elif msg["role"] == "assistant":
            memory.add_assistant_message(msg.get("content", ""))
        elif msg["role"] == "system":
            memory.add_context_message(msg.get("content", ""))

    def get_messages():
        return memory.get_messages()

    result = benchmark(get_messages)
    assert len(result) == len(sample_messages)


def test_token_estimation_single(benchmark):
    """Benchmark estimate_tokens for a single message."""
    message = {
        "role": "user",
        "content": "This is a test message for token estimation. " * 10,
    }

    result = benchmark(estimate_single, message)
    assert result > 0


def test_token_estimation_multiple(benchmark, sample_messages):
    """Benchmark estimate_tokens for multiple messages."""
    result = benchmark(estimate_tokens, sample_messages)
    assert result > 0


def test_token_estimation_large(benchmark):
    """Benchmark estimate_tokens for large text."""
    large_message = {
        "role": "user",
        "content": "Test message. " * 1000,
    }

    result = benchmark(estimate_single, large_message)
    assert result > 0


def test_context_truncation(benchmark, sample_messages):
    """Benchmark TruncationManager with many messages."""
    many_messages = []
    for i in range(50):
        many_messages.append({"role": "user", "content": f"Message {i}: " + "test " * 20})
        many_messages.append({"role": "assistant", "content": f"Response {i}: " + "reply " * 20})

    import asyncio

    def run_truncation():
        manager = TruncationManager(keep_last=6)
        return asyncio.run(manager.manage(many_messages, max_tokens=2000))

    result = benchmark(run_truncation)
    assert len(result) < len(many_messages)


def test_context_smart_truncation(benchmark, sample_messages):
    """Benchmark SmartTruncationManager."""
    from koboi.context.manager import SmartTruncationManager

    many_messages = []
    for i in range(50):
        many_messages.append({"role": "user", "content": f"Message {i}: " + "test " * 20})
        many_messages.append({"role": "assistant", "content": f"Response {i}: " + "reply " * 20})

    import asyncio

    def run_truncation():
        manager = SmartTruncationManager(keep_last=6)
        return asyncio.run(manager.manage(many_messages, max_tokens=2000))

    result = benchmark(run_truncation)
    assert len(result) < len(many_messages)


def test_context_key_facts(benchmark, sample_messages):
    """Benchmark KeyFactsManager."""
    from koboi.context.manager import KeyFactsManager

    many_messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What's the weather?"},
    ]

    for i in range(20):
        tool_call = {
            "id": f"tc_{i}",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}
        }
        many_messages.append({
            "role": "assistant",
            "content": f"Let me check {i}.",
            "tool_calls": [tool_call]
        })
        many_messages.append({
            "role": "tool",
            "tool_call_id": f"tc_{i}",
            "content": f"Weather in Paris: Sunny {i}C"
        })

    import asyncio

    def run_manage():
        manager = KeyFactsManager(keep_last=4)
        return asyncio.run(manager.manage(many_messages, max_tokens=1000))

    result = benchmark(run_manage)
    assert len(result) < len(many_messages)


def test_ensure_tool_integrity(benchmark):
    """Benchmark ensure_tool_integrity with complex message sequences."""
    from koboi.context.manager import ensure_tool_integrity

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Calculate 2+2"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "tc_1", "type": "function", "function": {"name": "calculator", "arguments": '{"expr": "2+2"}'}}
        ]},
        {"role": "tool", "tool_call_id": "tc_1", "content": "Result: 4"},
        {"role": "user", "content": "Now calculate 3+3"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "tc_2", "type": "function", "function": {"name": "calculator", "arguments": '{"expr": "3+3"}'}}
        ]},
    ]

    # Add some orphaned tool results
    messages.append({"role": "tool", "tool_call_id": "orphan_tc", "content": "Orphaned result"})

    result = benchmark(ensure_tool_integrity, messages)
    # Orphan should be removed
    assert all(m.get("tool_call_id") != "orphan_tc" for m in result)
