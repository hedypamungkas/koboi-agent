"""Tests for koboi.config module."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from koboi.config import Config, _resolve_env, _walk_resolve


class TestEnvResolution:
    def test_resolve_existing_env(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY", "my-api-key")
        assert _resolve_env("${TEST_KEY}") == "my-api-key"

    def test_resolve_missing_env_with_default(self):
        assert _resolve_env("${NONEXISTENT_KEY:fallback}") == "fallback"

    def test_resolve_missing_env_no_default(self):
        assert _resolve_env("${NONEXISTENT_KEY_XYZ}") == "${NONEXISTENT_KEY_XYZ}"

    def test_walk_resolve_dict(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "resolved")
        data = {"key": "${MY_KEY}", "nested": {"inner": "${MY_KEY}"}}
        result = _walk_resolve(data)
        assert result["key"] == "resolved"
        assert result["nested"]["inner"] == "resolved"

    def test_walk_resolve_list(self, monkeypatch):
        monkeypatch.setenv("ITEM", "value")
        result = _walk_resolve(["${ITEM}", "plain"])
        assert result == ["value", "plain"]

    def test_walk_resolve_passthrough(self):
        assert _walk_resolve(42) == 42
        assert _walk_resolve(True) is True


class TestConfig:
    def test_from_yaml(self, tmp_path):
        config_data = {
            "agent": {"name": "test", "max_iterations": 5},
            "llm": {"model": "gpt-4", "api_key": "key"},
        }
        path = tmp_path / "test.yaml"
        with open(path, "w") as f:
            yaml.dump(config_data, f)

        config = Config.from_yaml(path)
        assert config.agent_name == "test"
        assert config.max_iterations == 5
        assert config.model == "gpt-4"
        assert config.api_key == "key"

    def test_from_yaml_missing_file(self):
        with pytest.raises(FileNotFoundError):
            Config.from_yaml("/nonexistent/path.yaml")

    def test_get_nested(self, tmp_path):
        config = Config({"a": {"b": {"c": 42}}})
        assert config.get("a", "b", "c") == 42
        assert config.get("a", "b", "missing", default="fallback") == "fallback"

    def test_convenience_accessors(self):
        config = Config(
            {
                "agent": {"name": "my-agent", "system_prompt": "Hello"},
                "llm": {"model": "gpt-4o", "api_key": "key", "base_url": "http://api"},
                "rag": {"enabled": True},
            }
        )
        assert config.agent_name == "my-agent"
        assert config.system_prompt == "Hello"
        assert config.model == "gpt-4o"
        assert config.api_key == "key"
        assert config.base_url == "http://api"
        assert config.rag_enabled is True

    def test_defaults(self):
        config = Config({})
        assert config.agent_name == "koboi-agent"
        assert config.max_iterations == 10
        assert config.model == "gpt-4o-mini"
        assert config.rag_enabled is False
        assert config.provider == "openai"
        assert config.llm_timeout == 120.0
        assert config.llm_max_tokens is None  # unset -> omitted from OpenAI body
        assert config.llm_auth_token == ""

    def test_provider_property(self):
        config = Config({"llm": {"provider": "anthropic"}})
        assert config.provider == "anthropic"

    def test_anthropic_llm_properties(self):
        config = Config(
            {
                "llm": {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-20250514",
                    "api_key": "sk-ant-test",
                    "base_url": "https://api.anthropic.com/v1",
                    "timeout": 60.0,
                    "max_tokens": 8192,
                    "auth_token": "oauth-tok",
                },
            }
        )
        assert config.provider == "anthropic"
        assert config.model == "claude-sonnet-4-20250514"
        assert config.llm_timeout == 60.0
        assert config.llm_max_tokens == 8192
        assert config.llm_auth_token == "oauth-tok"


class TestConfigFromDict:
    def test_from_dict_basic(self):
        config = Config.from_dict(
            {
                "agent": {"name": "dict-agent", "max_iterations": 7},
                "llm": {"model": "gpt-4o", "api_key": "sk-test"},
            }
        )
        assert config.agent_name == "dict-agent"
        assert config.max_iterations == 7
        assert config.model == "gpt-4o"
        assert config.api_key == "sk-test"

    def test_from_dict_minimal(self):
        config = Config.from_dict({"agent": {"name": "minimal"}, "llm": {"model": "m"}})
        assert config.agent_name == "minimal"
        assert config.model == "m"

    def test_from_dict_empty(self):
        config = Config.from_dict({}, validate=False)
        assert config.agent_name == "koboi-agent"  # default

    def test_from_dict_with_env_vars(self, monkeypatch):
        monkeypatch.setenv("DICT_TEST_KEY", "resolved")
        config = Config.from_dict(
            {"agent": {"name": "env-agent"}, "llm": {"api_key": "${DICT_TEST_KEY}", "model": "x"}}
        )
        assert config.api_key == "resolved"

    def test_from_dict_preserves_nested(self):
        config = Config.from_dict(
            {
                "agent": {"name": "nested-agent"},
                "llm": {"model": "gpt-4"},
                "rag": {"enabled": True, "top_k": 5},
                "guardrails": {"input": {"max_length": 1000}},
            }
        )
        assert config.rag_enabled is True
        assert config.get("rag", "top_k") == 5
        assert config.get("guardrails", "input", "max_length") == 1000


class TestConfigFromString:
    def test_from_string_basic(self):
        yaml_str = """
agent:
  name: string-agent
  max_iterations: 5
llm:
  model: claude-sonnet-4-20250514
  api_key: sk-ant-test
"""
        config = Config.from_string(yaml_str)
        assert config.agent_name == "string-agent"
        assert config.max_iterations == 5
        assert config.model == "claude-sonnet-4-20250514"

    def test_from_string_minimal(self):
        config = Config.from_string("agent:\n  name: s\nllm:\n  model: m")
        assert config.agent_name == "s"

    def test_from_string_empty(self):
        config = Config.from_string("", validate=False)
        assert config.agent_name == "koboi-agent"

    def test_from_string_with_env_vars(self, monkeypatch):
        monkeypatch.setenv("STR_KEY", "from-env")
        config = Config.from_string("agent:\n  name: env-agent\nllm:\n  api_key: ${STR_KEY}\n  model: x")
        assert config.api_key == "from-env"

    def test_from_string_multiline_prompt(self):
        yaml_str = """
agent:
  name: prompt-agent
  system_prompt: |
    You are a helpful assistant.
    Be concise and accurate.
llm:
  model: gpt-4o
"""
        config = Config.from_string(yaml_str)
        assert "helpful assistant" in config.system_prompt
        assert "concise" in config.system_prompt


class TestConfigInheritance:
    def _write_yaml(self, path: Path, data: dict):
        path.write_text(yaml.dump(data, default_flow_style=False))

    def test_basic_extends(self, tmp_path):
        base = {"agent": {"name": "base", "max_iterations": 5}, "llm": {"model": "gpt-4"}}
        child = {"extends": "base.yaml", "agent": {"name": "child"}}
        self._write_yaml(tmp_path / "base.yaml", base)
        self._write_yaml(tmp_path / "child.yaml", child)
        config = Config.from_yaml(tmp_path / "child.yaml")
        assert config.agent_name == "child"
        assert config.max_iterations == 5

    def test_deep_merge(self, tmp_path):
        base = {
            "agent": {"name": "base"},
            "guardrails": {"input": {"max_length": 100}, "output": {"detect": True}},
            "llm": {"model": "gpt-4"},
        }
        child = {"extends": "base.yaml", "guardrails": {"input": {"max_length": 500}}}
        self._write_yaml(tmp_path / "base.yaml", base)
        self._write_yaml(tmp_path / "child.yaml", child)
        config = Config.from_yaml(tmp_path / "child.yaml")
        assert config.get("guardrails", "input", "max_length") == 500
        assert config.get("guardrails", "output", "detect") is True

    def test_list_override_replaces(self, tmp_path):
        base = {"agent": {"name": "base"}, "tools": {"builtin": ["calculator", "shell"]}, "llm": {"model": "gpt-4"}}
        child = {"extends": "base.yaml", "tools": {"builtin": ["web_search"]}}
        self._write_yaml(tmp_path / "base.yaml", base)
        self._write_yaml(tmp_path / "child.yaml", child)
        config = Config.from_yaml(tmp_path / "child.yaml")
        assert config.get("tools", "builtin") == ["web_search"]

    def test_multiple_extends(self, tmp_path):
        a = {"agent": {"name": "a", "max_iterations": 3}, "llm": {"model": "gpt-4"}}
        b = {"agent": {"name": "b"}}
        child = {"extends": ["a.yaml", "b.yaml"], "llm": {"model": "gpt-4o"}}
        self._write_yaml(tmp_path / "a.yaml", a)
        self._write_yaml(tmp_path / "b.yaml", b)
        self._write_yaml(tmp_path / "child.yaml", child)
        config = Config.from_yaml(tmp_path / "child.yaml")
        assert config.agent_name == "b"
        assert config.max_iterations == 3

    def test_circular_extends_raises(self, tmp_path):
        a = {"extends": "b.yaml", "agent": {"name": "a"}, "llm": {"model": "x"}}
        b = {"extends": "a.yaml", "agent": {"name": "b"}, "llm": {"model": "x"}}
        self._write_yaml(tmp_path / "a.yaml", a)
        self._write_yaml(tmp_path / "b.yaml", b)
        with pytest.raises(ValueError, match="Circular"):
            Config.from_yaml(tmp_path / "a.yaml")

    def test_missing_parent_raises(self, tmp_path):
        child = {"extends": "nonexistent.yaml", "agent": {"name": "x"}, "llm": {"model": "x"}}
        self._write_yaml(tmp_path / "child.yaml", child)
        with pytest.raises(FileNotFoundError):
            Config.from_yaml(tmp_path / "child.yaml")

    def test_extends_key_not_in_result(self, tmp_path):
        base = {"agent": {"name": "base"}, "llm": {"model": "gpt-4"}}
        child = {"extends": "base.yaml", "agent": {"name": "child"}}
        self._write_yaml(tmp_path / "base.yaml", base)
        self._write_yaml(tmp_path / "child.yaml", child)
        config = Config.from_yaml(tmp_path / "child.yaml")
        assert "extends" not in config.raw

    def test_from_string_no_extends(self):
        config = Config.from_string("agent:\n  name: test\nllm:\n  model: gpt-4")
        assert config.agent_name == "test"


class TestConfigValidation:
    def test_valid_config_passes(self):
        config = Config({"agent": {"name": "x"}, "llm": {"model": "y"}}, validate=True)
        assert config.agent_name == "x"

    def test_invalid_max_iterations_raises(self):
        with pytest.raises(ValueError, match="greater than or equal"):
            Config({"agent": {"name": "x", "max_iterations": 0}, "llm": {"model": "y"}}, validate=True)

    def test_validation_opt_in(self):
        config = Config({"agent": {"name": "x"}, "llm": {"model": "y"}})
        assert config._schema is None

    def test_builder_validates(self):
        config = Config.builder().agent(name="x").llm(model="y").build()
        assert config._schema is not None

    def test_builder_rejects_empty_name(self):
        with pytest.raises(ValueError, match="agent.name is required"):
            Config.builder().llm(model="y").build()

    def test_builder_rejects_empty_model(self):
        with pytest.raises(ValueError, match="llm.model is required"):
            Config.builder().agent(name="x").build()
