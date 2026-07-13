"""koboi/config.py -- branch coverage for accessors, builder, env/extends/provider-ref."""

from __future__ import annotations

import pytest

from koboi.config import (
    Config,
    _deep_merge,
    _expand_provider_refs,
    _resolve_env,
    _walk_resolve,
    extract_extra_params,
)


class TestEnvResolve:
    def test_resolve_env_with_default(self, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        assert _resolve_env("v-${MISSING_VAR:fallback}") == "v-fallback"

    def test_resolve_env_with_value(self, monkeypatch):
        monkeypatch.setenv("MY_VAR", "real")
        assert _resolve_env("${MY_VAR}") == "real"

    def test_resolve_env_no_default_no_var(self, monkeypatch):
        monkeypatch.delenv("NOPE_VAR", raising=False)
        # No default -> the raw ${...} token is preserved
        assert _resolve_env("${NOPE_VAR}") == "${NOPE_VAR}"

    def test_walk_resolve_nested(self, monkeypatch):
        monkeypatch.setenv("TOP", "prod")
        out = _walk_resolve({"a": "${TOP}", "b": ["${TOP}", 1], "c": {"d": "${TOP:dev}"}})
        assert out == {"a": "prod", "b": ["prod", 1], "c": {"d": "prod"}}
        # non-str/dict/list passthrough
        assert _walk_resolve(42) == 42


class TestExpandProviderRefs:
    def test_expand_llm_and_embedding_and_agents(self):
        data = {
            "providers": {"p1": {"provider": "openai", "model": "gpt", "api_key": "k"}},
            "llm": "p1",
            "embedding": "p1",
            "orchestration": {"agents": [{"name": "a", "llm": "p1"}]},
        }
        _expand_provider_refs(data)
        assert data["llm"] == {"provider": "openai", "model": "gpt", "api_key": "k"}
        assert data["embedding"]["model"] == "gpt"
        assert data["orchestration"]["agents"][0]["llm"]["model"] == "gpt"

    def test_expand_unknown_ref_left_untouched(self):
        data = {"providers": {"p1": {"model": "x"}}, "llm": "ghost"}
        _expand_provider_refs(data)
        assert data["llm"] == "ghost"  # unknown ref not expanded

    def test_expand_no_providers(self):
        data = {"llm": "x"}
        _expand_provider_refs(data)  # no-op
        assert data == {"llm": "x"}

    def test_expand_providers_not_dict(self):
        data = {"providers": "oops", "llm": "x"}
        _expand_provider_refs(data)
        assert data == {"providers": "oops", "llm": "x"}


class TestAccessors:
    def test_all_property_defaults_on_empty(self):
        c = Config({}, validate=False)
        assert c.tools == {}
        assert c.context == {}
        assert c.rag == {}
        assert c.guardrails == {}
        assert c.policy == {}
        assert c.skills == {}
        assert c.mcp == {}
        assert c.tracing == {}
        assert c.harness == {}
        assert c.eval == {}
        assert c.subagent == {}
        assert c.sandbox == {}
        assert c.journal == {}
        assert c.server == {}
        assert c.jobs == {}
        assert c.providers == {}
        assert c.pools == {}
        assert c.orchestration == {}
        assert c.keybindings == {}

    def test_agent_llm_accessor_defaults(self):
        c = Config({}, validate=False)
        assert c.agent == {}
        assert c.agent_name == "koboi-agent"
        assert c.theme == "koboi-dark"
        assert c.system_prompt == ""
        assert c.max_iterations == 10
        assert c.provider == "openai"
        assert c.model == "gpt-4o-mini"
        assert c.api_key == ""
        assert c.base_url == ""
        assert c.llm_timeout == 120.0
        assert c.llm_max_tokens is None
        assert c.llm_auth_token == ""
        assert c.auth_type == "api_key"
        assert c.temperature is None
        assert c.max_retries == 3
        assert c.retry_backoff_base == 2.0
        assert c.transport_retries == 2
        assert c.embedding_model == "text-embedding-3-small"
        assert c.api_version == "2023-06-01"
        assert c.rag_enabled is False
        assert c.mode == "chat"
        assert c.graduated_permissions is False
        assert c.trust_db_path == "koboi_trust.db"

    def test_get_nested_default_paths(self):
        c = Config({"a": {"b": None}}, validate=False)
        # node becomes None mid-path -> default
        assert c.get("a", "b", "c", default="D") == "D"
        # node not a dict mid-path -> default
        assert c.get("a", "b", default="D") == "D"
        # missing top key
        assert c.get("missing", default="D") == "D"
        # present
        c2 = Config({"x": 5}, validate=False)
        assert c2.get("x") == 5


class TestWarnUnknownLlmKeys:
    def test_warns_for_unknown_keys(self, caplog):
        c = Config(
            {
                "llm": {"bogus_key": 1, "model": "m"},
                "orchestration": {
                    "agents": [
                        {"name": "named", "llm": {"weird": 2}},
                        "string-agent",  # non-dict agent -> continue
                        {"name": "no-llm-dict", "llm": "ref"},  # non-dict llm -> continue
                    ]
                },
            },
            validate=False,
        )
        with caplog.at_level("WARNING"):
            c._warn_unknown_llm_keys()
        msgs = " ".join(r.message for r in caplog.records)
        assert "bogus_key" in msgs
        assert "named" in msgs and "weird" in msgs

    def test_valid_known_keys_no_warn(self, caplog):
        c = Config({"llm": {"model": "m", "top_p": 0.5}}, validate=False)
        with caplog.at_level("WARNING"):
            c._warn_unknown_llm_keys()
        assert not caplog.records


class TestToDictAndFroms:
    def test_to_dict_non_validated(self):
        c = Config({"agent": {"name": "x"}}, validate=False)
        assert c.to_dict()["agent"]["name"] == "x"
        assert c.raw["agent"]["name"] == "x"
        assert c.schema is None

    def test_from_yaml_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Config.from_yaml(tmp_path / "nope.yaml")

    def test_from_string_and_from_dict(self):
        c = Config.from_string("agent: {name: x}\nllm: {model: m}", validate=False)
        assert c.agent_name == "x"
        c2 = Config.from_dict({"agent": {"name": "y"}}, validate=False)
        assert c2.agent_name == "y"


class TestExtends:
    def test_single_extends(self, tmp_path):
        base = tmp_path / "base.yaml"
        base.write_text("agent:\n  name: base\n  max_iterations: 3\n")
        child = tmp_path / "child.yaml"
        child.write_text("extends: base.yaml\nllm: {model: m}\nagent:\n  name: child\n")
        c = Config.from_yaml(child)
        assert c.agent_name == "child"
        assert c.max_iterations == 3  # inherited

    def test_list_extends(self, tmp_path):
        a = tmp_path / "a.yaml"
        a.write_text("agent:\n  name: a\n")
        b = tmp_path / "b.yaml"
        b.write_text("agent:\n  max_iterations: 7\n")
        child = tmp_path / "c.yaml"
        child.write_text("extends: [a.yaml, b.yaml]\nllm: {model: m}\n")
        c = Config.from_yaml(child)
        assert c.agent_name == "a"
        assert c.max_iterations == 7

    def test_circular_extends_detected(self, tmp_path):
        f1 = tmp_path / "f1.yaml"
        f2 = tmp_path / "f2.yaml"
        f1.write_text("extends: f2.yaml\nagent: {name: a}\n")
        f2.write_text("extends: f1.yaml\nagent: {name: b}\n")
        with pytest.raises(ValueError, match="Circular"):
            Config.from_yaml(f1)


class TestDeepMerge:
    def test_merge_nested_and_override(self):
        out = _deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 9, "z": 3}, "b": 5})
        assert out == {"a": {"x": 1, "y": 9, "z": 3}, "b": 5}


class TestExtractExtraParams:
    def test_picks_known_drops_none(self):
        out = extract_extra_params({"top_p": 0.5, "model": "m", "seed": None, "stop": ["x"]})
        assert out == {"top_p": 0.5, "stop": ["x"]}

    def test_none_when_empty(self):
        assert extract_extra_params({"model": "m"}) is None


class TestConfigBuilder:
    def test_full_build_covers_all_kwargs(self):
        c = (
            Config.builder()
            .agent(name="a", system_prompt="s", max_iterations=4, description="d")
            .llm(
                provider="openai",
                model="m",
                api_key="k",
                base_url="u",
                temperature=0.1,
                max_tokens=100,
                timeout=10,
                max_retries=5,
                retry_backoff_base=1.5,
                auth_token="t",
                auth_type="bearer",
                embedding_model="emb",
            )
            .tools(builtin=["calc"], custom=[{"name": "x"}], defaults={"k": 1}, overrides={"y": 2})
            .context(strategy="smart", max_context_tokens=1000, keep_last=2)
            .rag(
                enabled=True,
                chunker="sentence",
                chunk_size=128,
                retriever="hybrid",
                top_k=5,
                augmentation="prepend",
                documents=["p.txt", {"path": "q"}],
            )
            .guardrails(input={"a": 1}, output={"b": 2}, rate_limit={"c": 3}, approval={"d": 4})
            .memory(backend="sqlite", db_path="/tmp/x.db", session_id="s1")
            .harness(telemetry=True, carryover=True, doom_loop={"enabled": True})
            .tracing(provider="langfuse", public_key="pk", secret_key="sk", base_url="bu")
            .policy(rules=[{"tool": "x"}])
            .skills(search_paths=["."], budget_chars=1000)
            .mcp(servers=[{"group": "g"}])
            .orchestration(enabled=True, router_type="llm", execution_mode="dag", agents=[{"name": "a"}])
            .sandbox(
                backend="restricted",
                workdir="/w",
                network="soft",
                network_binaries=["curl"],
                safe_path=["/tmp"],
                env_passthrough=True,
                rlimits={"nproc": 10},
                timeout=30,
                max_output=1000,
            )
            .journal(enabled=True, record_tool_calls=True)
            .server(
                enabled=True,
                host="0.0.0.0",
                port=8000,
                api_keys_file="k.json",
                api_keys=["a"],
                auth_required=True,
                cors={"origins": ["*"]},
                pool={"max": 4},
                timeouts={"drain_seconds": 5},
                limits={"max_iterations_cap": 20},
                idempotency={"chat_ttl_seconds": 60},
                workdir_ttl_seconds=120,
            )
            .build()
        )
        d = c.raw
        assert d["agent"]["description"] == "d"
        assert d["llm"]["embedding_model"] == "emb"
        assert d["tools"]["overrides"] == {"y": 2}
        assert d["context"]["keep_last"] == 2
        assert d["rag"]["documents"] == [{"path": "p.txt"}, {"path": "q"}]
        assert d["guardrails"]["approval"] == {"d": 4}
        assert d["memory"]["session_id"] == "s1"
        assert d["harness"]["doom_loop"] == {"enabled": True}
        assert d["tracing"]["secret_key"] == "sk"
        assert d["policy"]["rules"] == [{"tool": "x"}]
        assert d["skills"]["budget_chars"] == 1000
        assert d["mcp"]["servers"] == [{"group": "g"}]
        assert d["orchestration"]["execution"]["mode"] == "dag"
        assert d["sandbox"]["rlimits"] == {"nproc": 10}
        assert d["journal"]["record_tool_calls"] is True
        assert d["server"]["workdir_ttl_seconds"] == 120
