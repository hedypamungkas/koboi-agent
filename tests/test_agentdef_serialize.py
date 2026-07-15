"""Tests for AgentDef.to_dict / from_dict round-trip (S1)."""

from koboi.types import AgentDef


class TestAgentDefSerialize:
    def test_to_dict_remaps_config_keys_to_yaml(self):
        ad = AgentDef(
            name="classify",
            system_prompt="s",
            tools_config={"allowed": ["calc"]},
            rag_config={"top_k": 3},
            llm_config={"temperature": 0},
            depends_on=["a"],
            conditionals=[{"to": "b", "when": {"contains": "x"}}],
            output_schema={"type": "object"},
            force_response_format_with_tools=True,
            determinism={"seed": 1},
        )
        d = ad.to_dict()
        assert d["name"] == "classify"
        assert d["tools"] == {"allowed": ["calc"]}
        assert d["rag"] == {"top_k": 3}
        assert d["llm"] == {"temperature": 0}
        assert "tools_config" not in d
        assert "rag_config" not in d
        assert "llm_config" not in d
        assert d["output_schema"] == {"type": "object"}
        assert d["force_response_format_with_tools"] is True
        assert d["determinism"] == {"seed": 1}

    def test_to_dict_omits_empty_sections(self):
        d = AgentDef(name="n").to_dict()
        assert d == {"name": "n"}

    def test_from_dict_reads_yaml_keys(self):
        ac = {
            "name": "x",
            "tools": {"allowed": ["t"]},
            "rag": {"top_k": 2},
            "llm": {"temperature": 0.5},
            "depends_on": ["p"],
            "output_schema": {"type": "object"},
            "force_response_format_with_tools": True,
            "determinism": {"seed": 7},
        }
        ad = AgentDef.from_dict(ac)
        assert ad.tools_config == {"allowed": ["t"]}
        assert ad.rag_config == {"top_k": 2}
        assert ad.llm_config == {"temperature": 0.5}
        assert ad.depends_on == ["p"]
        assert ad.output_schema == {"type": "object"}
        assert ad.force_response_format_with_tools is True
        assert ad.determinism == {"seed": 7}

    def test_round_trip_to_dict_then_from_dict(self):
        ad = AgentDef(
            name="x",
            system_prompt="s",
            tools_config={"a": 1},
            llm_config={"temperature": 0},
            conditionals=[{"to": "y", "when": {"contains": "z"}}],
            interrupt_after=True,
            output_schema={"type": "object"},
        )
        assert AgentDef.from_dict(ad.to_dict()) == ad

    def test_from_dict_accepts_config_keys_too(self):
        ad = AgentDef.from_dict({"name": "x", "tools_config": {"a": 1}, "llm_config": {"seed": 1}})
        assert ad.tools_config == {"a": 1}
        assert ad.llm_config == {"seed": 1}
