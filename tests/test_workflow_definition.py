"""Tests for koboi.workflows.definition (S1): data model, bundle round-trip,
placeholder-aware redaction, determinism validator, graph snapshot."""

import koboi
from koboi.config import Config
from koboi.types import AgentDef
from koboi.workflows import (
    DeterminismProfile,
    WorkflowDefinition,
    build_from_config_path,
    build_graph_snapshot,
    parse_determinism,
    validate_workflow,
)


class TestDeterminismProfile:
    def test_merge_node_overrides_workflow(self):
        wf = DeterminismProfile(temperature=0.0, seed=42)
        node = DeterminismProfile(temperature=0.7)
        merged = wf.merge(node)
        assert merged.temperature == 0.7  # node wins
        assert merged.seed == 42  # workflow fills the gap

    def test_merge_none_returns_self(self):
        wf = DeterminismProfile(temperature=0.0)
        assert wf.merge(None) is wf

    def test_to_llm_overrides_maps_model_pin_to_model(self):
        p = DeterminismProfile(temperature=0.0, seed=1, top_p=0.9, model_pin="gpt-4o-mini-2024")
        assert p.to_llm_overrides() == {
            "temperature": 0.0,
            "seed": 1,
            "top_p": 0.9,
            "model": "gpt-4o-mini-2024",
        }

    def test_parse_determinism_none_when_unset(self):
        assert parse_determinism({}) is None
        assert parse_determinism({"orchestration": {}}) is None


class TestBundleRoundTrip:
    def test_to_bundle_yaml_then_from_bundle_yaml(self):
        wd = WorkflowDefinition(name="w", description="d", config={"agent": {"name": "x"}, "llm": {"model": "m"}})
        text = wd.to_bundle_yaml()
        assert text.lstrip().startswith("workflow:")  # envelope first
        wd2 = WorkflowDefinition.from_bundle_yaml(text)
        assert wd2.name == "w"
        assert wd2.description == "d"
        assert wd2.config["agent"]["name"] == "x"
        assert wd2.config["llm"]["model"] == "m"
        assert "workflow" not in wd2.config  # envelope popped on deserialize

    def test_determinism_property_reads_config(self):
        wd = WorkflowDefinition(config={"orchestration": {"determinism": {"temperature": 0.0, "seed": 5}}})
        det = wd.determinism
        assert det is not None
        assert det.temperature == 0.0
        assert det.seed == 5

    def test_determinism_property_none_when_unset(self):
        assert WorkflowDefinition(config={}).determinism is None


class TestRedactForExport:
    def test_placeholder_preserved_on_sensitive_key(self, tmp_path):
        cfg = tmp_path / "c.yaml"
        cfg.write_text(
            "agent:\n  name: x\n"
            "llm:\n  provider: openai\n  model: m\n"
            "  api_key: ${OPENAI_API_KEY:}\n"
            "  base_url: ${OPENAI_BASE_URL:http://x/v1}\n",
            encoding="utf-8",
        )
        wd = build_from_config_path(cfg, name="w")
        # api_key is a sensitive key with a ${VAR:} template -> KEPT (re-runnable)
        assert wd.config["llm"]["api_key"] == "${OPENAI_API_KEY:}"
        assert wd.config["llm"]["base_url"] == "${OPENAI_BASE_URL:http://x/v1}"

    def test_concrete_secret_masked(self, tmp_path):
        cfg = tmp_path / "c.yaml"
        cfg.write_text(
            "agent:\n  name: x\n"
            "llm:\n  provider: openai\n  model: m\n"
            "  api_key: sk-live-supersecretkey1234567890abcdef\n",
            encoding="utf-8",
        )
        wd = build_from_config_path(cfg, name="w")
        assert wd.config["llm"]["api_key"] != "sk-live-supersecretkey1234567890abcdef"
        assert "sk-live" not in str(wd.config["llm"]["api_key"])

    def test_provenance_stamps_koboi_version(self, tmp_path):
        cfg = tmp_path / "c.yaml"
        cfg.write_text("agent:\n  name: x\nllm:\n  provider: openai\n  model: m\n", encoding="utf-8")
        wd = build_from_config_path(cfg, name="w")
        assert wd.provenance.koboi_version == koboi.__version__
        assert wd.provenance.captured_at is not None

    def test_bundle_re_runs_as_config(self, tmp_path):
        # The exported body (envelope popped) must pass Config validation.
        cfg = tmp_path / "c.yaml"
        cfg.write_text(
            "agent:\n  name: x\nllm:\n  provider: openai\n  model: ${OPENAI_MODEL:m}\n  api_key: ${OPENAI_API_KEY:}\n",
            encoding="utf-8",
        )
        wd = build_from_config_path(cfg, name="w")
        reloaded = Config.from_dict(dict(wd.config))  # validates
        assert reloaded.raw["agent"]["name"] == "x"


class TestValidateWorkflow:
    def test_warns_anthropic_seed(self):
        wd = WorkflowDefinition(
            config={
                "agent": {"name": "x"},
                "llm": {"provider": "anthropic", "model": "claude"},
                "orchestration": {"determinism": {"seed": 1}},
            }
        )
        warns = validate_workflow(wd)
        assert any("seed" in w.lower() and "anthropic" in w.lower() for w in warns)

    def test_warns_unpinned_model(self):
        wd = WorkflowDefinition(
            config={
                "agent": {"name": "x"},
                "llm": {"provider": "openai", "model": "gpt-4o-mini"},
                "orchestration": {"determinism": {"temperature": 0.0}},
            }
        )
        assert any("model_pin" in w for w in validate_workflow(wd))

    def test_warns_sliding_window(self):
        wd = WorkflowDefinition(
            config={
                "agent": {"name": "x"},
                "llm": {"provider": "openai", "model": "m"},
                "context": {"strategy": "sliding_window"},
            }
        )
        assert any("sliding_window" in w for w in validate_workflow(wd))

    def test_no_warnings_for_clean_workflow(self):
        wd = WorkflowDefinition(config={"agent": {"name": "x"}, "llm": {"provider": "openai", "model": "m"}})
        assert validate_workflow(wd) == []


class TestGraphSnapshot:
    def test_non_lossy_and_backward_compatible(self):
        agent_defs = [
            AgentDef(
                name="classify",
                conditionals=[{"to": "praise", "when": {"contains": "POS"}}],
            ),
            AgentDef(name="praise", depends_on=["classify"]),
        ]
        cfg = Config.from_dict(
            {
                "agent": {"name": "x"},
                "llm": {"provider": "openai", "model": "m"},
                "orchestration": {"execution": {"mode": "dag"}, "router": {"type": "keyword"}},
            }
        )
        snap = build_graph_snapshot(agent_defs, cfg)
        # legacy keys preserved (backward compat)
        assert snap["nodes"] == ["classify", "praise"]
        assert {"from": "classify", "to": "praise"} in snap["edges"]
        # new non-lossy fields
        assert snap["conditionals"] == [{"from": "classify", "to": "praise", "when": {"contains": "POS"}}]
        assert snap["execution_mode"] == "dag"
        assert snap["router"] == {"type": "keyword"}
        assert snap["agents"][0]["name"] == "classify"
