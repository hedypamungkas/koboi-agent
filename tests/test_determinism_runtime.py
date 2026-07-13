"""Tests for facade._apply_determinism (S2): workflow+node determinism merge
into per-node llm_config, reusing _has_client_overrides + extract_extra_params."""

from koboi.config import extract_extra_params
from koboi.facade import _apply_determinism
from koboi.orchestration.factory import _has_client_overrides
from koboi.types import AgentDef


class TestApplyDeterminism:
    def test_workflow_determinism_applied_to_node(self):
        ad = AgentDef(name="x")
        _apply_determinism(ad, {"temperature": 0.0, "seed": 42})
        assert ad.llm_config["temperature"] == 0.0
        assert ad.llm_config["seed"] == 42
        assert _has_client_overrides(ad.llm_config) is True

    def test_node_overrides_workflow(self):
        ad = AgentDef(name="x", determinism={"temperature": 0.7})
        _apply_determinism(ad, {"temperature": 0.0, "seed": 42})
        assert ad.llm_config["temperature"] == 0.7  # node wins
        assert ad.llm_config["seed"] == 42  # workflow fills the gap

    def test_explicit_node_llm_preserved_setdefault(self):
        ad = AgentDef(name="x", llm_config={"model": "gpt-4o", "temperature": 0.9})
        _apply_determinism(ad, {"temperature": 0.0, "model_pin": "gpt-4o-mini-2024"})
        assert ad.llm_config["model"] == "gpt-4o"  # explicit node model NOT clobbered
        assert ad.llm_config["temperature"] == 0.9  # explicit node temp preserved

    def test_seed_and_top_p_picked_by_extract_extra_params(self):
        ad = AgentDef(name="x")
        _apply_determinism(ad, {"seed": 42, "top_p": 0.9})
        extras = extract_extra_params(ad.llm_config)
        assert extras is not None
        assert extras["seed"] == 42
        assert extras["top_p"] == 0.9

    def test_no_determinism_leaves_llm_unchanged(self):
        ad = AgentDef(name="x", llm_config={"max_context_tokens": 8000})
        _apply_determinism(ad, {})
        assert ad.llm_config == {"max_context_tokens": 8000}
        assert _has_client_overrides(ad.llm_config) is False

    def test_model_pin_maps_to_model_key(self):
        ad = AgentDef(name="x")
        _apply_determinism(ad, {"model_pin": "gpt-4o-mini-2024-07-18"})
        assert ad.llm_config["model"] == "gpt-4o-mini-2024-07-18"

    def test_node_only_determinism_without_workflow(self):
        ad = AgentDef(name="x", determinism={"temperature": 0.0, "seed": 1})
        _apply_determinism(ad, {})
        assert ad.llm_config["temperature"] == 0.0
        assert ad.llm_config["seed"] == 1
