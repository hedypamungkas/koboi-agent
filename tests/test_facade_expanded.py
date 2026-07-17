"""Tests for koboi/facade.py — KoboiAgent facade expanded coverage."""

from __future__ import annotations

import yaml
from unittest.mock import MagicMock, AsyncMock

from koboi.facade import (
    KoboiAgent,
    _build_client,
    _build_tools,
    _build_context,
    _build_guardrails,
    _build_approval,
    _build_skills,
    _build_hooks,
    _build_policy,
)
from koboi.config import Config
from koboi.logger import AgentLogger


def _write_config(tmp_path, config_data):
    p = tmp_path / "test_config.yaml"
    with open(p, "w") as f:
        yaml.dump(config_data, f)
    return p


def _base_config():
    return {
        "agent": {"name": "test-agent", "max_iterations": 5, "system_prompt": "You are helpful."},
        "llm": {"model": "gpt-4o-mini", "api_key": "test-key", "base_url": "http://localhost:8080/v1"},
    }


class TestBuildClient:
    def test_openai_provider(self, tmp_path):
        config = Config.from_yaml(_write_config(tmp_path, _base_config()))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))
        client = _build_client(config, logger)
        assert client is not None

    def test_anthropic_provider(self, tmp_path):
        cfg = _base_config()
        cfg["llm"]["provider"] = "anthropic"
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))
        client = _build_client(config, logger)
        assert client is not None

    def test_cloudflare_provider(self, tmp_path):
        cfg = _base_config()
        cfg["llm"]["provider"] = "cloudflare"
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))
        client = _build_client(config, logger)
        assert client is not None


class TestBuildTools:
    def test_empty_config(self, tmp_path):
        config = Config.from_yaml(_write_config(tmp_path, _base_config()))
        registry = _build_tools(config)
        assert len(registry._tools) == 0

    def test_builtin_tools(self, tmp_path):
        cfg = _base_config()
        cfg["tools"] = {"builtin": ["calculate", "list_files", "run_shell"]}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        registry = _build_tools(config)
        assert "calculate" in registry._tools
        assert "list_files" in registry._tools
        assert "run_shell" in registry._tools

    def test_custom_tool_import_failure(self, tmp_path):
        cfg = _base_config()
        cfg["tools"] = {"custom": [{"module": "nonexistent.module", "function": "foo"}]}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        registry = _build_tools(config)
        assert len(registry._tools) == 0  # import fails gracefully

    def test_tool_overrides(self, tmp_path):
        cfg = _base_config()
        cfg["tools"] = {"defaults": {"timeout": 30}, "overrides": {}}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        registry = _build_tools(config)
        assert len(registry._tools) == 0  # no tools, just config

    def test_disabled_tools_via_config(self, tmp_path):
        """P3g: tools.disabled removes a tool from the built registry end-to-end."""
        cfg = _base_config()
        cfg["tools"] = {"builtin": ["calculate", "run_shell", "read_file"], "disabled": ["run_shell"]}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        registry = _build_tools(config)
        names = {d["function"]["name"] for d in registry.get_definitions()}
        assert "run_shell" not in names
        assert {"calculate", "read_file"} <= names

    def test_disabled_alias_via_config(self, tmp_path):
        """P3g: the 'shell' alias resolves in tools.disabled too."""
        cfg = _base_config()
        cfg["tools"] = {"builtin": ["calculate", "run_shell"], "disabled": ["shell"]}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        registry = _build_tools(config)
        names = {d["function"]["name"] for d in registry.get_definitions()}
        assert "run_shell" not in names
        assert "calculate" in names

    def test_groups_via_config(self, tmp_path):
        """P3g: tools.groups hides non-matching groups from the LLM view."""
        cfg = _base_config()
        cfg["tools"] = {"builtin": ["calculate", "web_search"], "groups": ["math"]}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        registry = _build_tools(config)
        names = {d["function"]["name"] for d in registry.get_definitions()}
        assert names == {"calculate"}  # web_search (group "web") hidden


class TestBuildContext:
    def test_noop_strategy(self, tmp_path):
        config = Config.from_yaml(_write_config(tmp_path, _base_config()))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))
        result = _build_context(config, logger)
        assert result is None

    def test_truncation_strategy(self, tmp_path):
        cfg = _base_config()
        cfg["context"] = {"strategy": "truncation", "keep_last": 10}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))
        result = _build_context(config, logger)
        assert result is not None

    def test_smart_truncation_strategy(self, tmp_path):
        cfg = _base_config()
        cfg["context"] = {"strategy": "smart_truncation"}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))
        result = _build_context(config, logger)
        assert result is not None

    def test_unknown_strategy(self, tmp_path):
        cfg = _base_config()
        cfg["context"] = {"strategy": "unknown"}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))
        result = _build_context(config, logger)
        assert result is None


class TestBuildGuardrails:
    def test_no_guardrails(self, tmp_path):
        config = Config.from_yaml(_write_config(tmp_path, _base_config()))
        inp, out, rl, audit = _build_guardrails(config)
        assert inp == []
        assert out == []
        assert rl is None
        assert audit is None

    def test_input_guardrail(self, tmp_path):
        cfg = _base_config()
        cfg["guardrails"] = {"input": {"max_length": 500}}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        inp, out, rl, audit = _build_guardrails(config)
        assert len(inp) > 0
        assert out == []

    def test_output_guardrail(self, tmp_path):
        cfg = _base_config()
        cfg["guardrails"] = {"output": {"enabled": True}}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        inp, out, rl, audit = _build_guardrails(config)
        assert len(out) > 0

    def test_rate_limiter(self, tmp_path):
        cfg = _base_config()
        cfg["guardrails"] = {"rate_limit": {"max_calls_per_session": 50}}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        inp, out, rl, audit = _build_guardrails(config)
        assert rl is not None


class TestBuildApproval:
    def test_auto_approval(self, tmp_path):
        config = Config.from_yaml(_write_config(tmp_path, _base_config()))
        result = _build_approval(config)
        assert result is None

    def test_callback_approval(self, tmp_path):
        cfg = _base_config()
        cfg["guardrails"] = {"approval": {"handler": "callback", "callback": "lambda x: True"}}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        result = _build_approval(config)
        assert result is not None


class TestBuildSkills:
    def test_no_skills(self, tmp_path):
        config = Config.from_yaml(_write_config(tmp_path, _base_config()))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))
        result = _build_skills(config, logger)
        assert result is None


class TestBuildHooks:
    def test_basic_hooks(self, tmp_path):
        config = Config.from_yaml(_write_config(tmp_path, _base_config()))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))
        chain = _build_hooks(config, logger, audit_trail=None)
        assert chain is not None
        assert len(chain._hooks) >= 1  # at least LoggingHook

    def test_hooks_with_telemetry(self, tmp_path):
        cfg = _base_config()
        cfg["harness"] = {"telemetry": True}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))
        chain = _build_hooks(config, logger, audit_trail=None)
        assert len(chain._hooks) >= 2

    def test_hooks_with_carryover(self, tmp_path):
        cfg = _base_config()
        cfg["harness"] = {"carryover": True}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))
        chain = _build_hooks(config, logger, audit_trail=None)
        assert len(chain._hooks) >= 2

    def test_hooks_with_doom_loop(self, tmp_path):
        cfg = _base_config()
        cfg["harness"] = {"doom_loop": {"consecutive_identical_threshold": 3}}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))
        chain = _build_hooks(config, logger, audit_trail=None)
        assert len(chain._hooks) >= 2


class TestBuildPolicy:
    def test_no_policy(self, tmp_path):
        config = Config.from_yaml(_write_config(tmp_path, _base_config()))
        result = _build_policy(config)
        assert result is not None  # Always returns engine (hardcoded deny-list)

    def test_with_rules(self, tmp_path):
        cfg = _base_config()
        cfg["policy"] = {
            "rules": [
                {"tool": "shell", "pattern": "rm -rf", "action": "deny"},
                {"tool": "*", "action": "allow"},
            ]
        }
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        result = _build_policy(config)
        assert result is not None


class TestKoboiAgent:
    async def test_run_delegates(self, tmp_path):
        config = Config.from_yaml(_write_config(tmp_path, _base_config()))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))

        core = MagicMock()
        from koboi.types import RunResult

        core.run = AsyncMock(return_value=RunResult(content="ok", iterations_used=1, tool_calls_made=[], success=True))

        agent = KoboiAgent(core=core, config=config, logger=logger)
        result = await agent.run("test")
        assert result.content == "ok"
        core.run.assert_called_once_with("test")

    async def test_run_stream_delegates(self, tmp_path):
        config = Config.from_yaml(_write_config(tmp_path, _base_config()))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))

        core = MagicMock()
        from koboi.events import TextDeltaEvent

        async def mock_stream(msg):
            yield TextDeltaEvent(content="hi")

        core.run_stream = mock_stream

        agent = KoboiAgent(core=core, config=config, logger=logger)
        events = []
        async for e in agent.run_stream("test"):
            events.append(e)
        assert len(events) == 1

    def test_reset(self, tmp_path):
        config = Config.from_yaml(_write_config(tmp_path, _base_config()))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))

        core = MagicMock()
        agent = KoboiAgent(core=core, config=config, logger=logger)
        agent.reset()
        core.reset.assert_called_once()

    def test_add_tool(self, tmp_path):
        config = Config.from_yaml(_write_config(tmp_path, _base_config()))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))

        core = MagicMock()
        agent = KoboiAgent(core=core, config=config, logger=logger)
        agent.add_tool("my_tool", lambda x: x, "A tool", {"type": "object"})
        core.tools.register.assert_called_once()

    def test_config_property(self, tmp_path):
        config = Config.from_yaml(_write_config(tmp_path, _base_config()))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))

        core = MagicMock()
        agent = KoboiAgent(core=core, config=config, logger=logger)
        assert agent.config is config
        assert agent.core is core

    def test_run_sync(self, tmp_path):
        config = Config.from_yaml(_write_config(tmp_path, _base_config()))
        logger = AgentLogger(log_dir=str(tmp_path / "logs"))

        core = MagicMock()
        from koboi.types import RunResult

        core.run = AsyncMock(
            return_value=RunResult(content="sync ok", iterations_used=1, tool_calls_made=[], success=True)
        )

        agent = KoboiAgent(core=core, config=config, logger=logger)
        result = agent.run_sync("test")
        assert result.content == "sync ok"
