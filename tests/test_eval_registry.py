"""Tests for koboi/eval/registry.py -- Scorer and loader registries."""
from __future__ import annotations

import pytest

from koboi.eval.registry import ScorerRegistry, register_default_scorers


class TestScorerRegistry:
    def test_register_and_create(self):
        ScorerRegistry.register("test_scorer", lambda: "scorer_instance")
        result = ScorerRegistry.create("test_scorer")
        assert result == "scorer_instance"
        # cleanup
        ScorerRegistry._factories.pop("test_scorer", None)

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown scorer"):
            ScorerRegistry.create("definitely_nonexistent_scorer")

    def test_list_available(self):
        register_default_scorers()
        available = ScorerRegistry.list_available()
        assert "tool_usage" in available
        assert "keyword_presence" in available
        assert "output_length" in available

    def test_from_config(self):
        register_default_scorers()
        configs = [{"name": "tool_usage"}, {"name": "keyword_presence"}]
        scorers = ScorerRegistry.from_config(configs)
        assert len(scorers) == 2

    def test_from_config_missing_name(self):
        scorers = ScorerRegistry.from_config([{"no_name": "value"}])
        assert len(scorers) == 0

    def test_from_config_unknown_scorer(self):
        scorers = ScorerRegistry.from_config([{"name": "nonexistent"}])
        assert len(scorers) == 0

    def test_from_config_with_kwargs(self):
        register_default_scorers()
        configs = [{"name": "output_length", "min_length": 50}]
        scorers = ScorerRegistry.from_config(configs)
        assert len(scorers) == 1
