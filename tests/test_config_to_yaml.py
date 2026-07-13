"""Tests for Config.to_yaml() full-fidelity serialization (S1)."""

import yaml

from koboi.config import Config


class TestConfigToYaml:
    def test_to_yaml_preserves_forwardable_llm_keys(self):
        cfg = Config.from_dict(
            {
                "agent": {"name": "x"},
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "seed": 42,
                    "top_p": 0.9,
                    "response_format": {"type": "json_object"},
                },
            }
        )
        back = yaml.safe_load(cfg.to_yaml())
        assert back["llm"]["seed"] == 42
        assert back["llm"]["top_p"] == 0.9
        assert back["llm"]["response_format"] == {"type": "json_object"}

    def test_to_dict_still_lossy_unlike_to_yaml(self):
        # Documents the behavior we deliberately did NOT change: to_dict() is the
        # lossy validated-schema view; to_yaml() is the full-fidelity raw view.
        cfg = Config.from_dict({"agent": {"name": "x"}, "llm": {"provider": "openai", "model": "m", "seed": 42}})
        assert "seed" not in cfg.to_dict().get("llm", {})
        assert cfg.raw["llm"]["seed"] == 42
        assert yaml.safe_load(cfg.to_yaml())["llm"]["seed"] == 42

    def test_to_yaml_round_trips_through_from_string(self):
        cfg = Config.from_dict(
            {"agent": {"name": "x", "system_prompt": "hi"}, "llm": {"provider": "openai", "model": "m"}}
        )
        cfg2 = Config.from_string(cfg.to_yaml())
        assert cfg2.raw["agent"]["name"] == "x"
        assert cfg2.raw["llm"]["model"] == "m"

    def test_to_yaml_is_human_readable_unsorted(self):
        cfg = Config.from_dict({"agent": {"name": "x"}, "llm": {"provider": "openai", "model": "m"}})
        text = cfg.to_yaml()
        # sort_keys=False keeps insertion order; agent precedes llm as inserted.
        assert text.index("agent:") < text.index("llm:")
