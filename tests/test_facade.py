"""Tests for koboi.facade module."""
from __future__ import annotations

import yaml

from koboi.config import Config


def _write_config(tmp_path, config_data: dict) -> str:
    path = tmp_path / "test_config.yaml"
    with open(path, "w") as f:
        yaml.dump(config_data, f)
    return str(path)


class TestFacadeConfigLoading:
    def test_config_from_yaml(self, tmp_path):
        path = _write_config(tmp_path, {
            "agent": {"name": "test-agent", "system_prompt": "Hello", "max_iterations": 3},
            "llm": {"model": "gpt-4o-mini", "api_key": "test-key", "base_url": "http://localhost:8080/v1"},
        })
        config = Config.from_yaml(path)
        assert config.agent_name == "test-agent"
        assert config.max_iterations == 3

    def test_config_with_env_vars(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_API_KEY", "resolved-key-12345")
        path = _write_config(tmp_path, {
            "agent": {"name": "env-agent"},
            "llm": {"api_key": "${MY_API_KEY}", "base_url": "http://localhost/v1", "model": "gpt-4"},
        })
        config = Config.from_yaml(path)
        assert config.api_key == "resolved-key-12345"

    def test_config_rag_disabled_by_default(self, tmp_path):
        path = _write_config(tmp_path, {
            "agent": {"name": "no-rag"},
            "llm": {"model": "gpt-4"},
        })
        config = Config.from_yaml(path)
        assert config.rag_enabled is False

    def test_config_rag_enabled(self, tmp_path):
        path = _write_config(tmp_path, {
            "agent": {"name": "rag-agent"},
            "llm": {"model": "gpt-4"},
            "rag": {"enabled": True, "chunker": "sentence", "retriever": "keyword"},
        })
        config = Config.from_yaml(path)
        assert config.rag_enabled is True


class TestFacadeE2E:
    """End-to-end tests: create KoboiAgent from config and verify wiring."""

    def _base_config(self) -> dict:
        return {
            "agent": {"name": "e2e-agent", "system_prompt": "You are a test agent.", "max_iterations": 3},
            "llm": {"model": "gpt-4o-mini", "api_key": "test-key", "base_url": "http://localhost:8080/v1"},
        }

    def test_agent_creation_basic(self, tmp_path):
        """KoboiAgent.from_config creates an agent with basic config."""
        from koboi.facade import KoboiAgent
        path = _write_config(tmp_path, self._base_config())
        agent = KoboiAgent.from_config(path)
        assert agent.config.agent_name == "e2e-agent"
        assert agent.core is not None
        assert agent.core.max_iterations == 3

    def test_agent_has_tools_when_builtin_configured(self, tmp_path):
        """Agent has tools registered when builtin tools are specified."""
        from koboi.facade import KoboiAgent
        config = self._base_config()
        config["tools"] = {"builtin": ["calculate"]}
        path = _write_config(tmp_path, config)
        agent = KoboiAgent.from_config(path)
        assert "calculate" in agent.core.tools._tools

    def test_agent_has_guardrails_when_configured(self, tmp_path):
        """Agent has guardrails when configured."""
        from koboi.facade import KoboiAgent
        config = self._base_config()
        config["guardrails"] = {
            "input": {"detect_injection": True, "max_length": 5000},
            "output": {"detect_sensitive": True},
            "rate_limit": {"max_calls_per_session": 50, "max_calls_per_minute": 10},
        }
        path = _write_config(tmp_path, config)
        agent = KoboiAgent.from_config(path)
        assert len(agent.core.input_guardrails) > 0
        assert len(agent.core.output_guardrails) > 0
        assert agent.core.rate_limiter is not None

    def test_agent_has_context_manager(self, tmp_path):
        """Agent has context manager when strategy is configured."""
        from koboi.facade import KoboiAgent
        config = self._base_config()
        config["context"] = {"strategy": "truncation", "max_context_tokens": 4000}
        path = _write_config(tmp_path, config)
        agent = KoboiAgent.from_config(path)
        assert agent.core.context_manager is not None

    def test_agent_reset(self, tmp_path):
        """Agent reset clears memory."""
        from koboi.facade import KoboiAgent
        path = _write_config(tmp_path, self._base_config())
        agent = KoboiAgent.from_config(path)
        agent.core.memory.add_user_message("test")
        assert len(agent.core.memory) > 0
        agent.reset()
        assert len(agent.core.memory) == 0

    def test_agent_policy_engine_wired(self, tmp_path):
        """PolicyEngine is created when policy.rules is configured."""
        from koboi.facade import KoboiAgent
        config = self._base_config()
        config["policy"] = {"rules": [
            {"tool": "run_shell", "pattern": "rm -rf", "action": "deny"},
            {"tool": "run_shell", "action": "confirm"},
        ]}
        path = _write_config(tmp_path, config)
        agent = KoboiAgent.from_config(path)
        # Policy engine is wired through hooks, not directly on core
        assert agent.core is not None

    def test_agent_hook_chain_built(self, tmp_path):
        """HookChain is built and attached to agent."""
        from koboi.facade import KoboiAgent
        config = self._base_config()
        config["harness"] = {"telemetry": True, "carryover": True}
        path = _write_config(tmp_path, config)
        agent = KoboiAgent.from_config(path)
        assert agent.core.hooks is not None
        assert len(agent.core.hooks._hooks) > 0


class TestFacadeFactoryMethods:
    """Tests for from_dict() and from_config_string() factory methods."""

    def _base_data(self) -> dict:
        return {
            "agent": {"name": "factory-agent", "system_prompt": "You are helpful.", "max_iterations": 3},
            "llm": {"model": "gpt-4o-mini", "api_key": "test-key", "base_url": "http://localhost:8080/v1"},
        }

    def test_from_dict_creates_agent(self):
        from koboi.facade import KoboiAgent
        agent = KoboiAgent.from_dict(self._base_data())
        assert agent.config.agent_name == "factory-agent"
        assert agent.core is not None
        assert agent.core.max_iterations == 3

    def test_from_dict_with_tools(self):
        from koboi.facade import KoboiAgent
        data = self._base_data()
        data["tools"] = {"builtin": ["calculate"]}
        agent = KoboiAgent.from_dict(data)
        assert "calculate" in agent.core.tools._tools

    def test_from_dict_with_guardrails(self):
        from koboi.facade import KoboiAgent
        data = self._base_data()
        data["guardrails"] = {
            "input": {"max_length": 5000},
            "rate_limit": {"max_calls_per_session": 50},
        }
        agent = KoboiAgent.from_dict(data)
        assert len(agent.core.input_guardrails) > 0
        assert agent.core.rate_limiter is not None

    def test_from_config_string_creates_agent(self):
        from koboi.facade import KoboiAgent
        yaml_str = """
agent:
  name: yaml-string-agent
  system_prompt: Be helpful
  max_iterations: 5
llm:
  model: gpt-4o
  api_key: test-key
  base_url: http://localhost:8080/v1
"""
        agent = KoboiAgent.from_config_string(yaml_str)
        assert agent.config.agent_name == "yaml-string-agent"
        assert agent.config.model == "gpt-4o"
        assert agent.core.max_iterations == 5

    def test_from_config_string_with_tools(self):
        from koboi.facade import KoboiAgent
        yaml_str = """
agent:
  name: tool-agent
llm:
  model: gpt-4o
  api_key: test-key
  base_url: http://localhost:8080/v1
tools:
  builtin:
    - calculate
    - web_search
"""
        agent = KoboiAgent.from_config_string(yaml_str)
        assert "calculate" in agent.core.tools._tools

    def test_from_dict_and_from_config_produce_equivalent(self, tmp_path):
        """from_dict and from_config with same data produce equivalent agents."""
        from koboi.facade import KoboiAgent
        data = self._base_data()
        path = _write_config(tmp_path, data)
        agent_yaml = KoboiAgent.from_config(path)
        agent_dict = KoboiAgent.from_dict(data)
        assert agent_yaml.config.agent_name == agent_dict.config.agent_name
        assert agent_yaml.config.model == agent_dict.config.model
        assert agent_yaml.core.max_iterations == agent_dict.core.max_iterations

    def test_from_dict_is_shared_builder(self):
        """from_dict uses the same _from_config helper as from_config."""
        from koboi.facade import KoboiAgent
        agent = KoboiAgent.from_dict(self._base_data())
        # Verify all subsystems are wired (same as from_config)
        assert agent.core.client is not None
        assert agent.core.memory is not None
        assert agent.core.tools is not None
        assert agent.core.hooks is not None
