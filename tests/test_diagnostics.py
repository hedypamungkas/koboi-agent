"""Tests for koboi/diagnostics.py -- session diagnostics bundle generator."""

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from unittest.mock import MagicMock, patch

from koboi.diagnostics import (
    collect_diagnostics,
    _sanitize_config,
    _redact_nested,
    _get_version,
)


def _make_mock_agent():
    """Create a mock KoboiAgent with all needed attributes."""
    agent = MagicMock()
    agent.config.agent_name = "test-agent"
    agent.config.provider = "openai"
    agent.config.model = "gpt-4o"
    agent.config.max_iterations = 10
    agent.config.rag_enabled = False
    agent.config.mode = "chat"
    agent.config._data = {
        "agent": {"name": "test"},
        "llm": {"api_key": "sk-secret123", "model": "gpt-4o"},
    }

    agent.core.memory.get_messages.return_value = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    agent.core.memory._session_id = "sess_abc123"

    # Mock hooks
    mock_hook_chain = MagicMock()
    mock_hook_chain.find_hook.return_value = None
    mock_hook_chain.list_hooks.return_value = [
        {"name": "LoggingHook", "events": ["session_start"]},
    ]
    agent.core.hooks = mock_hook_chain

    # Mock tools
    mock_registry = MagicMock()
    mock_registry._tools = {
        "calculator": MagicMock(
            risk_level=MagicMock(value="safe"),
            description="Calculate math expressions",
        ),
    }
    agent.core.tools = mock_registry

    return agent


class TestCollectDiagnostics:
    def test_returns_valid_zip(self):
        agent = _make_mock_agent()
        data = collect_diagnostics(agent)
        assert isinstance(data, bytes)
        buf = BytesIO(data)
        with zipfile.ZipFile(buf) as zf:
            names = zf.namelist()
            assert "metadata.json" in names
            assert "messages.json" in names

    def test_metadata_contains_expected_fields(self):
        agent = _make_mock_agent()
        data = collect_diagnostics(agent)
        buf = BytesIO(data)
        with zipfile.ZipFile(buf) as zf:
            meta = json.loads(zf.read("metadata.json"))
            assert meta["agent_name"] == "test-agent"
            assert meta["model"] == "openai/gpt-4o"
            assert meta["max_iterations"] == 10
            assert meta["session_id"] == "sess_abc123"
            assert "python_version" in meta
            assert "platform" in meta

    def test_messages_included(self):
        agent = _make_mock_agent()
        data = collect_diagnostics(agent)
        buf = BytesIO(data)
        with zipfile.ZipFile(buf) as zf:
            messages = json.loads(zf.read("messages.json"))
            assert len(messages) == 2
            assert messages[0]["role"] == "user"

    def test_config_sanitized(self):
        agent = _make_mock_agent()
        data = collect_diagnostics(agent)
        buf = BytesIO(data)
        with zipfile.ZipFile(buf) as zf:
            config = json.loads(zf.read("config.json"))
            assert config["llm"]["api_key"] == "***REDACTED***"

    def test_tools_included(self):
        agent = _make_mock_agent()
        data = collect_diagnostics(agent)
        buf = BytesIO(data)
        with zipfile.ZipFile(buf) as zf:
            tools = json.loads(zf.read("tools.json"))
            assert "calculator" in tools

    def test_hooks_included(self):
        agent = _make_mock_agent()
        data = collect_diagnostics(agent)
        buf = BytesIO(data)
        with zipfile.ZipFile(buf) as zf:
            hooks = json.loads(zf.read("hooks.json"))
            assert len(hooks) == 1
            assert hooks[0]["name"] == "LoggingHook"

    def test_telemetry_included_when_present(self):
        agent = _make_mock_agent()
        mock_telemetry = MagicMock()
        mock_telemetry.report.return_value = {"total_tokens": 100}
        mock_hook = MagicMock()
        mock_hook._telemetry = mock_telemetry

        def find_hook(predicate):
            if predicate.__name__ == "<lambda>":
                return mock_hook
            return None

        agent.core.hooks.find_hook = find_hook
        data = collect_diagnostics(agent)
        buf = BytesIO(data)
        with zipfile.ZipFile(buf) as zf:
            telemetry = json.loads(zf.read("telemetry.json"))
            assert telemetry["total_tokens"] == 100

    def test_carryover_included_when_present(self):
        agent = _make_mock_agent()
        mock_carryover = MagicMock()
        mock_carryover._state.summary.return_value = "carryover data"
        mock_hook = MagicMock()
        mock_hook._state = mock_carryover._state

        call_count = [0]

        def find_hook(predicate):
            call_count[0] += 1
            if call_count[0] == 1:
                return None  # No telemetry
            return mock_hook

        agent.core.hooks.find_hook = find_hook
        data = collect_diagnostics(agent)
        buf = BytesIO(data)
        with zipfile.ZipFile(buf) as zf:
            assert zf.read("carryover.txt").decode() == "carryover data"

    def test_handles_exceptions_gracefully(self):
        agent = MagicMock()
        agent.config._data = {}
        agent.config.agent_name = "test"
        agent.config.provider = "openai"
        agent.config.model = "gpt-4o"
        agent.config.max_iterations = 5
        agent.config.rag_enabled = False
        agent.config.mode = "chat"
        agent.core.memory.get_messages.side_effect = Exception("fail")
        agent.core.memory._session_id = "test"
        agent.core.hooks.find_hook.return_value = None
        agent.core.hooks.list_hooks.return_value = []
        agent.core.tools._tools = {}
        # Should not raise
        data = collect_diagnostics(agent)
        assert isinstance(data, bytes)


class TestSanitizeConfig:
    def test_redacts_api_key(self):
        data = {"api_key": "sk-secret", "model": "gpt-4o"}
        result = _sanitize_config(data)
        assert result["api_key"] == "***REDACTED***"
        assert result["model"] == "gpt-4o"

    def test_redacts_nested(self):
        data = {"llm": {"api_key": "sk-secret", "password": "hunter2"}}
        result = _sanitize_config(data)
        assert result["llm"]["api_key"] == "***REDACTED***"
        assert result["llm"]["password"] == "***REDACTED***"

    def test_preserves_non_sensitive(self):
        data = {"model": "gpt-4o", "max_tokens": 100}
        result = _sanitize_config(data)
        assert result["model"] == "gpt-4o"
        assert result["max_tokens"] == 100


class TestRedactNested:
    def test_redacts_matching_keys(self):
        d = {"api_key": "secret", "token": "abc"}
        _redact_nested(d, {"api_key", "token"})
        assert d["api_key"] == "***REDACTED***"
        assert d["token"] == "***REDACTED***"

    def test_ignores_empty_values(self):
        d = {"api_key": ""}
        _redact_nested(d, {"api_key"})
        assert d["api_key"] == ""

    def test_ignores_non_string_values(self):
        d = {"api_key": 123}
        _redact_nested(d, {"api_key"})
        assert d["api_key"] == 123


class TestGetVersion:
    def test_returns_string(self):
        version = _get_version()
        assert isinstance(version, str)
