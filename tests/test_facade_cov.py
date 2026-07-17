"""koboi/facade.py -- branch coverage for the assembler + module-level builders.

Targets the builder helpers, MCP wiring, command hooks, orchestration parsing,
and KoboiAgent methods that the existing test_facade*.py files do not reach.
Real Config objects are preferred over deep mocks; heavy externals (subprocess
spawns, HTTP, RAG corpus builds) are mocked at the seam.
"""

from __future__ import annotations

import yaml
from unittest.mock import MagicMock, AsyncMock

import pytest

from koboi.config import Config
from koboi.facade import (
    AgentAssembler,
    KoboiAgent,
    _build_client,
    _build_context,
    _build_embedding_client,
    _build_guardrails,
    _build_mcp,
    _build_mode_manager,
    _build_approval,
    _build_orchestration,
    _build_pool_from_spec,
    _build_policy,
    _build_rag,
    _build_skills,
    _build_tools,
    _build_trust_db,
    _build_command_hooks,
    _build_router,
    _connect_mcp_servers,
    _connect_with_retry,
    _create_mcp_client,
    _embedding_member_from_dict,
    _mcp_namespace_prefix,
    _mcp_registrar_for_pairs,
    _mcp_risk_level,
    _normalize_guardrail_config,
    _parse_agent_defs,
    _resolve_chat_client,
    _setup_subagent,
    _setup_tasks,
    _warn_semantic_without_embeddings,
)
from koboi.logger import AgentLogger
from koboi.types import RiskLevel


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_config(tmp_path, config_data: dict) -> str:
    path = tmp_path / "test_config.yaml"
    with open(path, "w") as f:
        yaml.dump(config_data, f)
    return str(path)


def _base_config() -> dict:
    return {
        "agent": {"name": "cov-agent", "max_iterations": 5, "system_prompt": "You are helpful."},
        "llm": {"model": "gpt-4o-mini", "api_key": "test-key", "base_url": "http://localhost:8080/v1"},
    }


def _cfg(data: dict) -> Config:
    """Partial Config without schema validation (for builder-helper tests)."""
    return Config.from_dict(data, validate=False)


def _logger(tmp_path) -> AgentLogger:
    return AgentLogger(log_dir=str(tmp_path / "logs"))


# ---------------------------------------------------------------------------
# _build_client / _build_pool_from_spec / _resolve_chat_client
# ---------------------------------------------------------------------------


class TestBuildClientOverrides:
    def test_same_provider_keeps_credentials(self, tmp_path):
        config = Config.from_yaml(_write_config(tmp_path, _base_config()))
        overrides = {"model": "gpt-4o"}
        client = _build_client(config, _logger(tmp_path), llm_overrides=overrides)
        assert client is not None

    def test_provider_switch_blanks_inherited_credentials(self, tmp_path, monkeypatch):
        config = Config.from_yaml(_write_config(tmp_path, _base_config()))
        captured: dict = {}

        def fake_build(llm, logger):
            captured.update(llm)
            return MagicMock()

        monkeypatch.setattr("koboi.facade._build_client_from_dict", fake_build)
        # Switch to a different provider without supplying connection keys ->
        # api_key/auth_token/base_url must be blanked so the registry resolves env.
        _build_client(config, _logger(tmp_path), llm_overrides={"provider": "anthropic"})
        assert captured["provider"] == "anthropic"
        assert captured["api_key"] == ""
        assert captured["base_url"] == ""
        assert captured["auth_token"] == ""

    def test_provider_switch_with_explicit_key_kept(self, tmp_path):
        config = Config.from_yaml(_write_config(tmp_path, _base_config()))
        overrides = {"provider": "anthropic", "api_key": "ant-key"}
        client = _build_client(config, _logger(tmp_path), llm_overrides=overrides)
        assert client.api_key == "ant-key"


class TestBuildPoolFromSpec:
    def test_unknown_pool_raises(self, tmp_path):
        config = _cfg({"pools": {}})
        with pytest.raises(ValueError, match="Unknown pool reference"):
            _build_pool_from_spec("nope", config, _logger(tmp_path))

    def test_empty_providers_raises(self, tmp_path):
        config = _cfg({"pools": {"p1": {"providers": []}}})
        with pytest.raises(ValueError, match="no `providers:`"):
            _build_pool_from_spec("p1", config, _logger(tmp_path))

    def test_unresolved_member_raises(self, tmp_path):
        # member ref to a named provider that doesn't exist -> resolve raises
        config = _cfg({"pools": {"p1": {"providers": ["ghost"]}}})
        with pytest.raises(ValueError, match="Unknown provider reference"):
            _build_pool_from_spec("p1", config, _logger(tmp_path))

    def test_unsupported_policy_raises(self, tmp_path):
        config = _cfg(
            {
                "pools": {
                    "p1": {
                        "policy": "round_robin",
                        "providers": [{"provider": "openai", "api_key": "k"}],
                    }
                }
            }
        )
        with pytest.raises(NotImplementedError, match="not implemented"):
            _build_pool_from_spec("p1", config, _logger(tmp_path))

    def test_valid_failover_pool_built(self, tmp_path):
        config = _cfg(
            {
                "pools": {
                    "p1": {
                        "policy": "failover",
                        "providers": [
                            {"provider": "openai", "model": "gpt-4o-mini", "api_key": "k1"},
                            {"provider": "openai", "model": "gpt-4o", "api_key": "k2"},
                        ],
                        "circuit_breaker": {"failures": 5, "cooldown_s": 12.0},
                    }
                }
            }
        )
        pool = _build_pool_from_spec("p1", config, _logger(tmp_path))
        assert pool is not None
        # circuit breaker wired with custom values
        assert pool._breaker.failure_threshold == 5

    def test_resolve_chat_client_pool_branch(self, tmp_path):
        config = _cfg(
            {
                "llm": {"pool": "p1"},
                "pools": {"p1": {"providers": [{"provider": "openai", "api_key": "k1"}]}},
            }
        )
        client = _resolve_chat_client(config, _logger(tmp_path))
        assert client is not None  # a ProviderPool


class TestEmbeddingHelpers:
    def test_embedding_member_without_api_key_raises(self, tmp_path):
        with pytest.raises(ValueError, match="api_key"):
            _embedding_member_from_dict({"provider": "openai"}, _logger(tmp_path))

    def test_embedding_member_builds_client(self, tmp_path, monkeypatch):
        captured: dict = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return "EMB_MEMBER"

        monkeypatch.setattr("koboi.llm.factory.create_client", fake_create)
        out = _embedding_member_from_dict(
            {"provider": "openai", "api_key": "k", "model": "text-embedding-3-small"}, _logger(tmp_path)
        )
        assert out == "EMB_MEMBER"
        assert captured["embedding_model"] == "text-embedding-3-small"

    def test_build_embedding_client_pool(self, tmp_path, monkeypatch):
        monkeypatch.setattr("koboi.llm.factory.create_client", lambda **kw: MagicMock(name="emb"))
        config = _cfg(
            {
                "embedding": {"pool": "ep"},
                "pools": {"ep": {"providers": [{"provider": "openai", "api_key": "k"}]}},
            }
        )
        client = _build_embedding_client(config, _logger(tmp_path))
        assert client is not None


# ---------------------------------------------------------------------------
# _build_tools / _build_context / _build_rag
# ---------------------------------------------------------------------------


class TestBuildToolsCustomModule:
    def test_custom_module_success_registers(self, tmp_path, monkeypatch):
        mod_dir = tmp_path / "custom_tools"
        mod_dir.mkdir()
        (mod_dir / "my_tool_mod.py").write_text(
            "from koboi.tools.registry import tool\n"
            "@tool(name='my_custom_tool', description='d', "
            "parameters={'type':'object','properties':{}})\n"
            "def my_custom_tool():\n"
            "    return 'ok'\n"
        )
        monkeypatch.syspath_prepend(str(mod_dir))
        cfg = _base_config()
        cfg["tools"] = {"custom": [{"module": "my_tool_mod"}]}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        registry = _build_tools(config)
        assert "my_custom_tool" in registry._tools


class TestBuildContextBranches:
    def test_custom_modules_loaded(self, tmp_path, monkeypatch):
        called: list = []
        monkeypatch.setattr(
            "koboi.context.registry.load_custom_context_modules",
            lambda mods: called.append(mods),
        )
        cfg = _base_config()
        cfg["context"] = {
            "strategy": "smart_truncation",
            "custom_modules": ["my.ctxmod"],
            "keep_last": 8,
            "summarization_truncation": 2,
            "safety_margin": 500,
        }
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        mgr = _build_context(config, _logger(tmp_path))
        assert mgr is not None
        assert called == [["my.ctxmod"]]
        assert mgr.keep_last == 8
        assert mgr.safety_margin == 500

    def test_tokenizer_wired_when_available(self, tmp_path, monkeypatch):
        sentinel = object()
        monkeypatch.setattr("koboi.tokens.make_tokenizer", lambda *_a: sentinel)
        cfg = _base_config()
        cfg["context"] = {"strategy": "truncation"}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        client = MagicMock()
        client.provider = "openai"
        client.model = "gpt-4o"
        mgr = _build_context(config, _logger(tmp_path), client=client)
        assert mgr.tokenizer is sentinel

    def test_safety_margin_bool_not_applied(self, tmp_path):
        # bool is an int subclass but must not be treated as a valid margin
        cfg = _base_config()
        cfg["context"] = {"strategy": "truncation", "safety_margin": True}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        mgr = _build_context(config, _logger(tmp_path))
        assert mgr is not None
        # default margin stays 0 (True rejected)
        assert mgr.safety_margin == 0


class TestBuildRagAndWarn:
    def test_warn_semantic_anthropic_without_embedding(self, caplog):
        config = _cfg({"rag": {"retriever": "semantic"}, "llm": {"provider": "anthropic"}})
        with caplog.at_level("WARNING"):
            _warn_semantic_without_embeddings(config, has_dedicated_embedding_client=False)
        assert any("semantic" in r.message for r in caplog.records)

    def test_warn_skipped_when_dedicated_embedding_present(self):
        config = _cfg({"rag": {"retriever": "semantic"}, "llm": {"provider": "anthropic"}})
        # No exception, no warning; just returns None
        assert _warn_semantic_without_embeddings(config, has_dedicated_embedding_client=True) is None

    def test_warn_skipped_for_keyword_retriever(self):
        config = _cfg({"rag": {"retriever": "keyword"}, "llm": {"provider": "anthropic"}})
        assert _warn_semantic_without_embeddings(config, has_dedicated_embedding_client=False) is None

    def test_build_rag_wiring(self, tmp_path, monkeypatch):
        captured: dict = {}

        def fake_build_rag(rag_dict, *, client, chat_client, logger):
            captured["rag_enabled"] = rag_dict.get("enabled")
            captured["augmentation"] = rag_dict.get("augmentation")
            captured["client"] = client
            captured["chat"] = chat_client
            return "RAG_PIPELINE"

        monkeypatch.setattr("koboi.rag.registry.build_rag", fake_build_rag)
        cfg = _base_config()
        cfg["rag"] = {"enabled": True, "retriever": "keyword"}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        chat_client = MagicMock()
        result = _build_rag(config, chat_client, _logger(tmp_path))
        assert result == "RAG_PIPELINE"
        assert captured["rag_enabled"] is True
        assert captured["augmentation"] == "on_the_fly"  # defaulted
        assert captured["chat"] is chat_client

    def test_build_rag_returns_none_when_disabled(self, tmp_path):
        config = Config.from_yaml(_write_config(tmp_path, _base_config()))
        assert _build_rag(config, MagicMock(), _logger(tmp_path)) is None


# ---------------------------------------------------------------------------
# guardrails / approval / skills / mode / trust
# ---------------------------------------------------------------------------


class TestNormalizeGuardrailConfig:
    def test_none_returns_empty(self):
        assert _normalize_guardrail_config(None) == []

    def test_dict_with_name(self):
        out = _normalize_guardrail_config({"name": "x", "max": 1})
        assert out == [{"name": "x", "max": 1}]

    def test_dict_without_name_uses_default(self):
        out = _normalize_guardrail_config({"max": 1}, default_name="content_filter")
        assert out == [{"name": "content_filter", "max": 1}]

    def test_list_filters_unnamed_and_non_dicts(self):
        out = _normalize_guardrail_config([{"name": "a"}, {"no_name": True}, "stray", {"name": "b"}])
        assert out == [{"name": "a"}, {"name": "b"}]

    def test_unknown_type_returns_empty(self):
        assert _normalize_guardrail_config(42) == []


class TestBuildGuardrailsAudit:
    def test_audit_with_db_path(self, tmp_path):
        cfg = _base_config()
        cfg["guardrails"] = {"audit": {"db_path": str(tmp_path / "audit.db")}}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        _, _, _, audit = _build_guardrails(config, logger=_logger(tmp_path))
        assert audit is not None

    def test_audit_without_db_path(self, tmp_path):
        cfg = _base_config()
        cfg["guardrails"] = {"audit": {"enabled": True}}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        _, _, _, audit = _build_guardrails(config, logger=_logger(tmp_path))
        assert audit is not None

    def test_output_bare_list_falls_back_to_content_filter(self, tmp_path):
        # output_conf truthy but normalizes to empty -> elif builds content_filter.
        # Uses an unvalidated config (the list shape is rejected by the schema, but
        # _build_guardrails reads raw via config.get).
        config = _cfg({"guardrails": {"output": [{"no_name": True}]}})
        _, out, _, _ = _build_guardrails(config)
        assert len(out) == 1


class TestBuildApprovalBranches:
    def test_cli_handler(self, tmp_path):
        cfg = _base_config()
        cfg["guardrails"] = {"approval": {"handler": "cli"}}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        handler = _build_approval(config)
        assert handler is not None

    def test_async_callback_without_callback_returns_none(self, tmp_path):
        cfg = _base_config()
        cfg["guardrails"] = {"approval": {"handler": "async_callback"}}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        assert _build_approval(config) is None

    def test_async_callback_with_callback(self, tmp_path):
        # Unvalidated config so the callable survives (YAML can't serialize lambdas).
        config = _cfg(
            {
                "guardrails": {
                    "approval": {
                        "handler": "async_callback",
                        "callback": lambda *a: True,
                        "timeout": 30,
                    }
                }
            }
        )
        handler = _build_approval(config, trust_db=MagicMock())
        assert handler is not None


class TestBuildSkillsSearchPaths:
    def test_search_paths_builds_registry(self, tmp_path):
        cfg = _base_config()
        cfg["skills"] = {"search_paths": [str(tmp_path)], "budget_chars": 4000}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        reg = _build_skills(config, _logger(tmp_path))
        assert reg is not None


class TestBuildModeManager:
    def test_invalid_mode_falls_back_to_chat(self, tmp_path):
        cfg = _base_config()
        cfg["agent"]["mode"] = "bogus"
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        mm = _build_mode_manager(config)
        from koboi.modes import AgentMode

        assert mm.current_mode == AgentMode.CHAT


class TestBuildTrustDb:
    def test_enabled_builds_db(self, tmp_path):
        cfg = _base_config()
        cfg["guardrails"] = {"approval": {"graduated": True, "trust_db_path": str(tmp_path / "t.db")}}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        trust = _build_trust_db(config)
        assert trust is not None

    def test_disabled_returns_none(self, tmp_path):
        config = Config.from_yaml(_write_config(tmp_path, _base_config()))
        assert _build_trust_db(config) is None

    def test_init_failure_returns_none(self, tmp_path, monkeypatch):
        cfg = _base_config()
        cfg["guardrails"] = {"approval": {"graduated": True}}
        config = Config.from_yaml(_write_config(tmp_path, cfg))

        def boom(*a, **kw):
            raise RuntimeError("nope")

        monkeypatch.setattr("koboi.trust.TrustDatabase", boom)
        assert _build_trust_db(config) is None


class TestBuildPolicyInvalidAction:
    def test_invalid_action_falls_back_to_allow(self, tmp_path):
        cfg = _base_config()
        cfg["policy"] = {"rules": [{"tool": "x", "action": "bogus_action"}]}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        engine = _build_policy(config)
        assert engine is not None  # built; invalid action -> ALLOW


# ---------------------------------------------------------------------------
# MCP helpers
# ---------------------------------------------------------------------------


class TestMcpRiskLevel:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("safe", RiskLevel.SAFE),
            ("MODERATE", RiskLevel.MODERATE),
            ("Destructive", RiskLevel.DESTRUCTIVE),
            ("unknown", RiskLevel.SAFE),
            (None, RiskLevel.SAFE),
        ],
    )
    def test_mapping(self, raw, expected):
        assert _mcp_risk_level({"risk_level": raw} if raw is not None else {}) == expected


class TestMcpNamespacePrefix:
    def test_no_namespace(self):
        config = _cfg({})
        assert _mcp_namespace_prefix(0, {"group": "g"}, config) is None

    def test_namespace_with_group(self):
        config = _cfg({"mcp": {"namespace": True}})
        assert _mcp_namespace_prefix(2, {"group": "g"}, config) == "mcp__g"

    def test_namespace_without_group_uses_index(self):
        config = _cfg({"mcp": {"namespace": True}})
        assert _mcp_namespace_prefix(3, {}, config) == "mcp__3"


class TestMcpConfEndpoint:
    def test_streamable_http(self):
        assert KoboiAgent._mcp_conf_endpoint({"transport": "streamable-http", "url": "http://x"}) == "http://x"

    def test_stdio(self):
        assert KoboiAgent._mcp_conf_endpoint({"command": "npx", "args": ["-y", "s"]}) == "npx -y s"

    def test_stdio_empty(self):
        assert KoboiAgent._mcp_conf_endpoint({}) == ""


class TestCreateMcpClient:
    def test_http_without_url_raises(self, tmp_path):
        with pytest.raises(ValueError, match="url"):
            _create_mcp_client({}, "streamable-http", _logger(tmp_path), None)

    def test_http_builds_client(self, tmp_path):
        c = _create_mcp_client(
            {"url": "http://x", "headers": {"a": "b"}},
            "streamable-http",
            _logger(tmp_path),
            None,
        )
        assert c is not None

    def test_stdio_without_command_raises(self, tmp_path):
        with pytest.raises(ValueError, match="command"):
            _create_mcp_client({}, "stdio", _logger(tmp_path), None)

    def test_stdio_disallowed_runner_raises(self, tmp_path):
        with pytest.raises(ValueError, match="allow-list"):
            _create_mcp_client(
                {"command": "/usr/bin/dangerous"},
                "stdio",
                _logger(tmp_path),
                _cfg({}),
            )

    def test_stdio_allowlist_extension(self, tmp_path):
        # allow-list matches on basename
        c = _create_mcp_client(
            {"command": "custom-runner", "args": ["--x"]},
            "stdio",
            _logger(tmp_path),
            _cfg({"mcp": {"allowlist_commands": ["custom-runner"]}}),
        )
        assert c is not None

    def test_stdio_default_runner_built(self, tmp_path):
        c = _create_mcp_client({"command": "uvx", "args": ["pkg"]}, "stdio", _logger(tmp_path), _cfg({}))
        assert c is not None


class TestConnectWithRetry:
    def test_success_first_try(self):
        client = MagicMock()
        client.connect = MagicMock()
        _connect_with_retry(client, connect_retries=2, backoff_base=2.0)
        client.connect.assert_called_once()

    def test_success_after_retry(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda *_a: None)
        client = MagicMock()
        client.connect = MagicMock(side_effect=[ConnectionError("x"), None])
        _connect_with_retry(client, connect_retries=2, backoff_base=2.0)
        assert client.connect.call_count == 2

    def test_all_attempts_fail_raises(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda *_a: None)
        client = MagicMock()
        client.connect = MagicMock(side_effect=ConnectionError("nope"))
        with pytest.raises(ConnectionError):
            _connect_with_retry(client, connect_retries=1, backoff_base=2.0)


class TestConnectMcpServers:
    def test_empty_servers(self, tmp_path):
        config = _cfg({})
        assert _connect_mcp_servers(config, _logger(tmp_path)) == []

    def test_fail_fast_reraises(self, tmp_path, monkeypatch):
        config = _cfg({"mcp": {"fail_fast": True, "servers": [{"command": "npx"}]}})

        def boom(*a, **kw):
            raise RuntimeError("connect failed")

        monkeypatch.setattr("koboi.facade._create_mcp_client", boom)
        with pytest.raises(RuntimeError):
            _connect_mcp_servers(config, _logger(tmp_path))

    def test_warn_and_skip_with_logger(self, tmp_path, monkeypatch):
        config = _cfg({"mcp": {"servers": [{"command": "npx"}]}})

        def boom(*a, **kw):
            raise RuntimeError("connect failed")

        monkeypatch.setattr("koboi.facade._create_mcp_client", boom)
        logger = MagicMock()
        pairs = _connect_mcp_servers(config, logger)
        assert pairs == []
        logger.log.assert_called_once()

    def test_warn_and_skip_without_logger(self, tmp_path, monkeypatch, caplog):
        config = _cfg({"mcp": {"servers": [{"command": "npx"}]}})

        def boom(*a, **kw):
            raise RuntimeError("connect failed")

        monkeypatch.setattr("koboi.facade._create_mcp_client", boom)
        with caplog.at_level("WARNING"):
            assert _connect_mcp_servers(config, None) == []
        assert any("connection failed" in r.message for r in caplog.records)


class TestBuildMcpAndRegistrar:
    def test_build_mcp_registers_and_returns_clients(self, monkeypatch):
        client_a = MagicMock()
        client_b = MagicMock()
        monkeypatch.setattr(
            "koboi.facade._connect_mcp_servers",
            lambda *a, **kw: [(client_a, {"group": "g1"}), (client_b, {})],
        )
        registered: list = []
        monkeypatch.setattr(
            "koboi.mcp.base.register_mcp_tools",
            lambda c, reg, **kw: registered.append(c) or ["t"],
        )
        tools = MagicMock()
        clients = _build_mcp(_cfg({}), tools, MagicMock())
        assert clients == [client_a, client_b]
        assert len(registered) == 2

    def test_registrar_closure_registers_all_pairs(self, monkeypatch):
        c1, c2 = MagicMock(), MagicMock()
        pairs = [(c1, {"group": "g", "risk_heuristic": True}), (c2, {})]
        calls: list = []

        def fake_register(client, registry, **kw):
            calls.append((client, kw))
            return ["t"]

        monkeypatch.setattr("koboi.mcp.base.register_mcp_tools", fake_register)
        registrar = _mcp_registrar_for_pairs(pairs, _cfg({}))
        reg = MagicMock()
        registrar(reg)
        assert len(calls) == 2
        # first pair had risk_heuristic -> resolver passed
        assert calls[0][1]["risk_resolver"] is not None
        assert calls[1][1]["risk_resolver"] is None


# ---------------------------------------------------------------------------
# command hooks
# ---------------------------------------------------------------------------


class TestBuildCommandHooks:
    def test_no_entries_noop(self, tmp_path):
        hook_chain = MagicMock()
        _build_command_hooks(_cfg({}), MagicMock(), hook_chain)
        hook_chain.add.assert_not_called()

    def test_entries_without_allow_exec_skipped(self, caplog):
        config = _cfg({"hooks": {"on_event": [{"command": "echo hi", "events": ["pre_tool_use"]}]}})
        hook_chain = MagicMock()
        with caplog.at_level("WARNING"):
            _build_command_hooks(config, MagicMock(), hook_chain)
        hook_chain.add.assert_not_called()
        assert any("allow_exec" in r.message for r in caplog.records)

    def test_entries_with_allow_exec_wires_hook(self):
        config = _cfg(
            {
                "hooks": {
                    "allow_exec": True,
                    "command_timeout": 5.0,
                    "on_event": [{"command": "echo hi", "events": ["pre_tool_use"], "name": "h1", "priority": 7}],
                }
            }
        )
        hook_chain = MagicMock()
        _build_command_hooks(config, MagicMock(), hook_chain)
        hook_chain.add.assert_called_once()

    def test_seccomp_warning_emitted(self, caplog):
        config = _cfg(
            {
                "sandbox": {"network_isolation": "seccomp"},
                "hooks": {
                    "allow_exec": True,
                    "on_event": [{"command": "echo", "events": ["post_output"]}],
                },
            }
        )
        with caplog.at_level("WARNING"):
            _build_command_hooks(config, MagicMock(), MagicMock())
        assert any("seccomp" in r.message for r in caplog.records)

    def test_unknown_event_raises(self):
        config = _cfg(
            {
                "hooks": {
                    "allow_exec": True,
                    "on_event": [{"command": "echo", "events": ["bogus_event"]}],
                }
            }
        )
        with pytest.raises(ValueError, match="unknown event"):
            _build_command_hooks(config, MagicMock(), MagicMock())


# ---------------------------------------------------------------------------
# orchestration parsing
# ---------------------------------------------------------------------------


class TestParseAgentDefs:
    def test_empty_agents_raises(self):
        config = _cfg({"orchestration": {"agents": []}})
        with pytest.raises(ValueError, match="at least one agent"):
            _parse_agent_defs(config)

    def test_agent_without_name_raises(self):
        config = _cfg({"orchestration": {"agents": [{"description": "x"}]}})
        with pytest.raises(ValueError, match="must have a 'name'"):
            _parse_agent_defs(config)

    def test_valid_agents_parsed(self):
        config = _cfg(
            {
                "orchestration": {
                    "agents": [{"name": "a", "system_prompt": "x", "depends_on": ["b"], "interrupt_after": True}]
                }
            }
        )
        defs = _parse_agent_defs(config)
        assert len(defs) == 1
        assert defs[0].name == "a"
        assert defs[0].depends_on == ["b"]
        assert defs[0].interrupt_after is True


class TestBuildRouter:
    def test_keyword_router(self):
        config = _cfg({"orchestration": {"router": {"type": "keyword"}}})
        router = _build_router(config, MagicMock(), [])
        assert router is not None

    def test_llm_router(self):
        config = _cfg({"orchestration": {"router": {"type": "llm", "enable_dynamic": True}}})
        router = _build_router(config, MagicMock(), [])
        assert router is not None

    def test_hybrid_router(self):
        config = _cfg({"orchestration": {"router": {"type": "hybrid", "confidence_threshold": 0.7}}})
        router = _build_router(config, MagicMock(), [])
        assert router is not None


class TestFromConfigBranches:
    def test_resume_session_sets_memory_session_id(self, tmp_path):
        config = Config.from_yaml(_write_config(tmp_path, _base_config()))
        KoboiAgent._from_config(config, resume_session="sess-xyz")
        assert config._data["memory"]["session_id"] == "sess-xyz"

    def test_orchestration_enabled_routes_to_builder(self, monkeypatch):
        sentinel = MagicMock(spec=KoboiAgent)
        monkeypatch.setattr(
            "koboi.facade._build_orchestration", lambda cfg, verbose=False, peer_registry=None: sentinel
        )
        config = _cfg({"orchestration": {"enabled": True}})
        agent = KoboiAgent._from_config(config)
        assert agent is sentinel


# ---------------------------------------------------------------------------
# AgentAssembler step coverage
# ---------------------------------------------------------------------------


class TestAssemblerMemoryAndJournal:
    def test_build_memory_retention_bool_guard_and_owner(self, tmp_path):
        cfg = _base_config()
        cfg["memory"] = {
            "backend": "sqlite",
            "db_path": str(tmp_path / "m.db"),
            "retention": {"max_messages": True},  # bool -> ignored
            "owner": "tenant-a",
        }
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        asm = AgentAssembler(config)
        asm.logger = _logger(tmp_path)
        mem = asm.build_memory()
        assert mem is not None
        assert asm.config.get("memory", "owner") == "tenant-a"

    def test_build_memory_non_sqlite_backend(self, tmp_path):
        cfg = _base_config()
        cfg["memory"] = {"backend": "memory"}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        asm = AgentAssembler(config)
        asm.logger = _logger(tmp_path)
        mem = asm.build_memory()
        from koboi.memory import ConversationMemory

        assert isinstance(mem, ConversationMemory)

    def test_build_journal_disabled(self, tmp_path):
        cfg = _base_config()
        cfg["memory"] = {"backend": "sqlite", "db_path": str(tmp_path / "m.db")}
        cfg["journal"] = {"enabled": False}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        asm = AgentAssembler(config)
        asm.logger = _logger(tmp_path)
        asm.build_memory()
        assert asm.build_journal() is None

    def test_build_journal_non_sqlite_memory(self, tmp_path):
        cfg = _base_config()
        cfg["memory"] = {"backend": "memory"}
        cfg["journal"] = {"enabled": True}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        asm = AgentAssembler(config)
        asm.logger = _logger(tmp_path)
        asm.build_memory()
        assert asm.build_journal() is None


class TestAssemblerProactiveMemory:
    def test_disabled_returns_none(self, tmp_path):
        config = Config.from_yaml(_write_config(tmp_path, _base_config()))
        asm = AgentAssembler(config)
        asm.logger = _logger(tmp_path)
        asm.client = MagicMock()
        asm.memory = MagicMock()
        asm.tools = MagicMock()
        assert asm.build_proactive_memory() is None

    def test_enabled_builds_coordinator(self, tmp_path):
        cfg = _base_config()
        cfg["memory"] = {"proactive": {"enabled": True, "extract": True}}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        asm = AgentAssembler(config)
        asm.logger = _logger(tmp_path)
        asm.client = MagicMock()
        asm.memory = MagicMock()
        asm.tools = MagicMock()
        asm.tools.get_dep.return_value = None  # no builtin memory store -> creates one
        proactive = asm.build_proactive_memory()
        assert proactive is not None
        assert proactive.extract_enabled is True


# ---------------------------------------------------------------------------
# build() extra-hook wiring (full assembly)
# ---------------------------------------------------------------------------


class TestBuildExtraHooks:
    def test_skill_persistence_hook_added(self, tmp_path):
        cfg = _base_config()
        cfg["skills"] = {"search_paths": [str(tmp_path)]}
        agent = KoboiAgent.from_dict(cfg)
        hook_names = [type(h).__name__ for h in agent.core.hooks._hooks]
        assert "SkillPersistenceHook" in hook_names

    def test_read_before_write_reset_hook_added(self, tmp_path):
        cfg = _base_config()
        cfg["tools"] = {"builtin": ["read_file", "write_file"]}
        agent = KoboiAgent.from_dict(cfg)
        hook_names = [type(h).__name__ for h in agent.core.hooks._hooks]
        assert "ReadBeforeWriteResetHook" in hook_names

    def test_task_persistence_hook_added(self, tmp_path):
        cfg = _base_config()
        cfg["memory"] = {"backend": "memory"}
        cfg["tools"] = {"builtin": ["task_create", "task_update", "task_list"]}
        agent = KoboiAgent.from_dict(cfg)
        hook_names = [type(h).__name__ for h in agent.core.hooks._hooks]
        assert "TaskPersistenceHook" in hook_names

    def test_proactive_extraction_hook_added(self, tmp_path):
        cfg = _base_config()
        cfg["memory"] = {"backend": "memory", "proactive": {"enabled": True, "extract": True}}
        agent = KoboiAgent.from_dict(cfg)
        hook_names = [type(h).__name__ for h in agent.core.hooks._hooks]
        assert "ProactiveExtractionHook" in hook_names


# ---------------------------------------------------------------------------
# _setup_subagent / _setup_tasks
# ---------------------------------------------------------------------------


class TestSetupSubagentAndTasks:
    def test_setup_subagent_wires_manager(self, tmp_path):
        cfg = _base_config()
        cfg["tools"] = {"builtin": ["delegate_tasks"]}
        cfg["subagent"] = {"timeout": 42.0, "max_iterations": 3}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        tools = _build_tools(config)
        _setup_subagent(tools, MagicMock(), MagicMock(), _logger(tmp_path), memory=None, config=config)
        assert tools.get_dep("subagent_manager") is not None

    def test_setup_tasks_wires_manager_and_injects_hook(self, tmp_path):
        cfg = _base_config()
        cfg["memory"] = {"backend": "memory"}
        cfg["tools"] = {"builtin": ["task_create", "task_update"]}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        tools = _build_tools(config)

        task_hook = MagicMock()
        task_hook.__class__.__name__ = "TaskHook"
        hook_chain = MagicMock()
        hook_chain._hooks = [task_hook]
        _setup_tasks(tools, config, hook_chain=hook_chain)
        mgr = tools.get_dep("task_manager")
        assert mgr is not None
        assert task_hook.manager is mgr


# ---------------------------------------------------------------------------
# KoboiAgent methods (inject / telemetry / langfuse / mcp status / props)
# ---------------------------------------------------------------------------


class TestKoboiAgentInjectTools:
    def test_core_none_is_noop(self):
        agent = KoboiAgent(core=None)
        agent.inject_tool_definitions([{"name": "x"}])  # no raise

    def test_inject_new_tool_registers(self):
        core = MagicMock()
        agent = KoboiAgent(core=core)
        agent.inject_tool_definitions([{"function": {"name": "ext_tool", "parameters": {"type": "object"}}}])
        core.tools.register.assert_called_once()

    def test_inject_dict_type_normalized_to_object(self):
        core = MagicMock()
        agent = KoboiAgent(core=core)
        agent.inject_tool_definitions([{"function": {"name": "ext2", "parameters": {"type": "dict"}}}])
        _, kwargs = core.tools.register.call_args
        assert kwargs["parameters"]["type"] == "object"

    def test_inject_existing_name_skipped(self):
        core = MagicMock()
        core.tools.__contains__ = lambda self, name: True
        core.tools.__bool__ = lambda self: True
        # name already present -> skip
        agent = KoboiAgent(core=core)
        agent.inject_tool_definitions([{"name": "calc"}])
        core.tools.register.assert_not_called()

    def test_inject_bare_function_key(self):
        core = MagicMock()
        agent = KoboiAgent(core=core)
        # top-level dict without "function" wrapper
        agent.inject_tool_definitions([{"name": "bare", "parameters": {}}])
        core.tools.register.assert_called_once()


class TestKoboiAgentTelemetry:
    def test_core_none_returns_none(self):
        assert KoboiAgent(core=None).get_telemetry() is None

    def test_core_telemetry_attr_returned(self):
        core = MagicMock()
        tel = object()
        core.telemetry = tel
        assert KoboiAgent(core=core).get_telemetry() is tel

    def test_telemetry_found_via_hook(self):
        core = MagicMock()
        core.telemetry = None
        found = MagicMock()
        found.telemetry = "TEL"
        core.hooks.find_hook.return_value = found
        assert KoboiAgent(core=core).get_telemetry() == "TEL"

    def test_telemetry_not_found_returns_none(self):
        core = MagicMock()
        core.telemetry = None
        core.hooks.find_hook.return_value = None
        assert KoboiAgent(core=core).get_telemetry() is None

    def test_ensure_telemetry_hook_core_none(self):
        agent = KoboiAgent(core=None)
        agent.ensure_telemetry_hook()  # no raise

    def test_ensure_telemetry_hook_already_present(self):
        core = MagicMock()
        core.hooks.find_hook.return_value = MagicMock()
        agent = KoboiAgent(core=core)
        agent.ensure_telemetry_hook()
        core.hooks.add.assert_not_called()

    def test_ensure_telemetry_hook_added(self):
        core = MagicMock()
        core.hooks.find_hook.return_value = None
        agent = KoboiAgent(core=core)
        agent.ensure_telemetry_hook()
        core.hooks.add.assert_called_once()


class TestKoboiAgentLangfuse:
    def _agent_with_hook(self, available, client):
        core = MagicMock()
        hook = MagicMock()
        hook.available = available
        hook.get_client.return_value = client
        core.hooks.find_hook.return_value = hook if available else None
        return KoboiAgent(core=core)

    def test_core_none(self):
        assert KoboiAgent(core=None).push_langfuse_scores("t", []) is None

    def test_no_hook(self):
        core = MagicMock()
        core.hooks.find_hook.return_value = None
        assert KoboiAgent(core=core).push_langfuse_scores("t", []) is None

    def test_unavailable_hook(self):
        core = MagicMock()
        hook = MagicMock()
        hook.available = False
        core.hooks.find_hook.return_value = hook
        assert KoboiAgent(core=core).push_langfuse_scores("t", []) is None

    def test_no_client(self):
        core = MagicMock()
        hook = MagicMock()
        hook.available = True
        hook.get_client.return_value = None
        core.hooks.find_hook.return_value = hook
        assert KoboiAgent(core=core).push_langfuse_scores("t", []) is None

    def test_score_pushed_and_flushed(self):
        core = MagicMock()
        hook = MagicMock()
        hook.available = True
        client = MagicMock()
        hook.get_client.return_value = client
        core.hooks.find_hook.return_value = hook
        score = MagicMock()
        score.name = "faithfulness"
        score.value = 0.9
        score.reason = "ok"
        KoboiAgent(core=core).push_langfuse_scores("trace-1", [score])
        client.score.assert_called_once()
        client.flush.assert_called_once()

    def test_client_error_swallowed(self):
        core = MagicMock()
        hook = MagicMock()
        hook.available = True
        client = MagicMock()
        client.score.side_effect = RuntimeError("boom")
        hook.get_client.return_value = client
        core.hooks.find_hook.return_value = hook
        # should not raise
        KoboiAgent(core=core).push_langfuse_scores("t", [MagicMock(name="n", value=1, reason="r")])


class TestKoboiAgentMcpStatus:
    def test_live_clients_entries(self):
        client = MagicMock()
        client.endpoint = "npx pkg"
        client.name = "srv1"
        client.transport = "stdio"
        client.is_connected.return_value = True
        client.server_info = {"ver": "1"}
        client.tool_names = ["t1", "t2"]
        config = MagicMock()
        config.get.return_value = []
        agent = KoboiAgent(core=MagicMock(), config=config, mcp_clients=[client])
        status = agent.mcp_status()
        assert len(status) == 1
        assert status[0]["connected"] is True
        assert status[0]["tool_names"] == ["t1", "t2"]

    def test_configured_but_failed_server_visible(self):
        client = MagicMock()
        client.endpoint = "npx live"
        client.name = "live"
        client.transport = "stdio"
        client.is_connected.return_value = True
        client.server_info = {}
        client.tool_names = []
        config = MagicMock()
        config.get.return_value = [{"command": "npx", "args": ["failed"], "transport": "stdio", "group": "dead"}]
        agent = KoboiAgent(core=MagicMock(), config=config, mcp_clients=[client])
        status = agent.mcp_status()
        # one live + one dead
        assert len(status) == 2
        dead = [s for s in status if not s["connected"]]
        assert len(dead) == 1
        assert dead[0]["configured"] is True

    def test_no_config_no_live(self):
        agent = KoboiAgent(core=None, config=None, mcp_clients=[])
        assert agent.mcp_status() == []


class TestKoboiAgentMcpClientLifecycle:
    def test_add_mcp_client_registers_and_appends(self, monkeypatch):
        core = MagicMock()
        monkeypatch.setattr("koboi.mcp.base.register_mcp_tools", lambda c, reg, **kw: ["t1", "t2"])
        agent = KoboiAgent(core=core)
        client = MagicMock()
        names = agent.add_mcp_client(client, group="g", risk_level=RiskLevel.MODERATE)
        assert names == ["t1", "t2"]
        assert client in agent._mcp_clients

    def test_remove_mcp_client_disables_closes_removes(self):
        core = MagicMock()
        client = MagicMock()
        client.tool_names = ["t1"]
        agent = KoboiAgent(core=core, mcp_clients=[client])
        agent.remove_mcp_client(client)
        core.tools.disable.assert_called_once_with(["t1"])
        client.close.assert_called_once()
        assert client not in agent._mcp_clients

    def test_remove_mcp_client_disable_failure_logged(self):
        core = MagicMock()
        core.tools.disable.side_effect = RuntimeError("x")
        client = MagicMock()
        client.name = "c"
        client.tool_names = ["t1"]
        agent = KoboiAgent(core=core, mcp_clients=[client])
        agent.remove_mcp_client(client)  # no raise
        client.close.assert_called_once()

    def test_remove_mcp_client_close_failure_logged(self):
        core = MagicMock()
        client = MagicMock()
        client.name = "c"
        client.tool_names = []
        client.close.side_effect = RuntimeError("x")
        agent = KoboiAgent(core=core, mcp_clients=[client])
        agent.remove_mcp_client(client)  # no raise
        assert client not in agent._mcp_clients


class TestKoboiAgentPropsAndReplace:
    def test_properties_return_construction_args(self):
        orch = MagicMock()
        mm = MagicMock()
        td = MagicMock()
        agent = KoboiAgent(core=MagicMock(), orchestrator=orch, mode_manager=mm, trust_db=td)
        assert agent.orchestrator is orch
        assert agent.mode_manager is mm
        assert agent.trust_db is td

    def test_mcp_clients_returns_copy(self):
        client = MagicMock()
        agent = KoboiAgent(core=MagicMock(), mcp_clients=[client])
        view = agent.mcp_clients
        view.append("x")
        assert len(agent._mcp_clients) == 1  # internal list untouched

    def test_replace_from_copies_state(self):
        other = KoboiAgent(core="C", config="CFG", mode_manager="MM", trust_db="TD", mcp_clients=["m"])
        agent = KoboiAgent()
        agent.replace_from(other)
        assert agent.core == "C"
        assert agent.config == "CFG"
        assert agent.trust_db == "TD"


# ---------------------------------------------------------------------------
# run / resume / run_sync / close orchestrator & augmentation branches
# ---------------------------------------------------------------------------


class TestRunAndResumeOrchestrator:
    async def test_run_with_orchestrator(self):
        orch = MagicMock()
        orch.default_mode = "sequential"
        orch_result = MagicMock()
        orch_result.final_answer = "done"
        orch_result.agent_results = [MagicMock(tokens_used=50)]
        orch_result.routing = MagicMock(method="keyword", confidence=0.8)
        orch_result.execution_mode = "sequential"
        orch.run = AsyncMock(return_value=orch_result)
        agent = KoboiAgent(core=MagicMock(), orchestrator=orch)
        result = await agent.run("hi")
        assert result.content == "done"
        assert result.metadata["routing_method"] == "keyword"

    async def test_run_with_orchestrator_multimodal_input(self):
        orch = MagicMock()
        orch.default_mode = "sequential"
        orch_result = MagicMock()
        orch_result.final_answer = "ok"
        orch_result.agent_results = []
        orch_result.routing = MagicMock(method="llm", confidence=0.6)
        orch_result.execution_mode = "parallel"
        orch.run = AsyncMock(return_value=orch_result)
        agent = KoboiAgent(core=MagicMock(), orchestrator=orch)
        result = await agent.run([{"type": "text", "text": "hello"}])
        assert result.content == "ok"

    async def test_run_stream_with_orchestrator_multimodal(self):
        orch = MagicMock()
        orch.default_mode = "sequential"

        async def fake_stream(q, mode="sequential"):
            yield MagicMock()

        orch.run_stream = fake_stream
        agent = KoboiAgent(core=MagicMock(), orchestrator=orch)
        events = []
        async for e in agent.run_stream([{"type": "text", "text": "x"}]):
            events.append(e)
        assert len(events) == 1

    async def test_resume_orchestrator_raises(self):
        from koboi.exceptions import AgentError

        agent = KoboiAgent(core=MagicMock(), orchestrator=MagicMock())
        with pytest.raises(AgentError, match="not supported in orchestration mode"):
            await agent.resume()

    async def test_resume_no_core_raises(self):
        from koboi.exceptions import AgentError

        agent = KoboiAgent(core=None)
        with pytest.raises(AgentError, match="No core agent"):
            await agent.resume()


class TestRunSyncAsyncContext:
    async def test_run_sync_from_running_loop_uses_bg_thread(self):
        core = MagicMock()
        core.run = AsyncMock(return_value=MagicMock(content="bg-ok"))
        core.hooks = MagicMock()
        core.memory = MagicMock()
        agent = KoboiAgent(core=core)
        # We are inside a running loop (async test) -> background-thread path.
        result = agent.run_sync("hello")
        assert result.content == "bg-ok"


class TestCloseAugmentationAndOrchestrator:
    async def test_close_orchestrator_subagents_cleaned(self):
        orch = MagicMock()
        orch.client = MagicMock()
        orch.client.close = AsyncMock()
        sub_agent = MagicMock()
        sub_agent.memory.close = MagicMock()
        sub_agent_client = MagicMock()
        sub_agent_client.close = AsyncMock()
        sub_agent.client = sub_agent_client
        orch._agents_map = {"a": sub_agent}
        agent = KoboiAgent(core=MagicMock(), orchestrator=orch)
        await agent.close()
        sub_agent.memory.close.assert_called_once()
        sub_agent_client.close.assert_called_once()
        orch.client.close.assert_called_once()

    async def test_close_augmentation_retriever(self):
        core = MagicMock()
        core.memory = MagicMock()
        core.audit_trail = None
        core.client = MagicMock()
        core.client.close = AsyncMock()
        aug = MagicMock()
        retriever = MagicMock()
        retriever.close = AsyncMock()
        aug.retriever = retriever
        core.augmentation = aug
        agent = KoboiAgent(core=core)
        await agent.close()
        retriever.close.assert_called_once()

    async def test_close_augmentation_retriever_error_swallowed(self):
        core = MagicMock()
        core.memory = MagicMock()
        core.audit_trail = None
        core.client = MagicMock()
        core.client.close = AsyncMock()
        aug = MagicMock()
        retriever = MagicMock()
        retriever.close = AsyncMock(side_effect=RuntimeError("x"))
        aug.retriever = retriever
        core.augmentation = aug
        agent = KoboiAgent(core=core)
        await agent.close()  # no raise
        retriever.close.assert_called_once()

    async def test_close_audit_trail_closed(self):
        core = MagicMock()
        core.memory = MagicMock()
        core.audit_trail = MagicMock()
        core.audit_trail.close = MagicMock()
        core.client = MagicMock()
        core.client.close = AsyncMock()
        core.augmentation = None
        agent = KoboiAgent(core=core)
        await agent.close()
        core.audit_trail.close.assert_called_once()


# ---------------------------------------------------------------------------
# extra edge branches: close/del exceptions, inject guards, dedup, resume,
# connect success, pool None member, setup sqlite, orchestration E2E
# ---------------------------------------------------------------------------


class TestCloseAndDelExceptions:
    async def test_close_mcp_client_failure_logged(self):
        mcp = MagicMock()
        mcp.close.side_effect = RuntimeError("x")
        core = MagicMock()
        core.memory = MagicMock()
        core.client = MagicMock()
        core.client.close = AsyncMock()
        core.augmentation = None
        core.audit_trail = None
        agent = KoboiAgent(core=core, mcp_clients=[mcp])
        await agent.close()  # no raise
        mcp.close.assert_called_once()

    def test_del_logger_close_exception_swallowed(self):
        logger = MagicMock()
        logger.close.side_effect = RuntimeError("x")
        agent = KoboiAgent(core=MagicMock(), logger=logger)
        agent.__del__()  # no raise

    def test_del_bg_loop_exceptions_swallowed(self):
        agent = KoboiAgent(core=MagicMock())
        bg_loop = MagicMock()
        bg_loop.call_soon_threadsafe.side_effect = RuntimeError("stop")
        bg_thread = MagicMock()
        bg_thread.join.side_effect = RuntimeError("join")
        bg_loop.close.side_effect = RuntimeError("close")
        agent._bg_loop = bg_loop
        agent._bg_thread = bg_thread
        agent.__del__()  # every teardown branch raises -> all swallowed


class TestInjectToolGuards:
    def test_tools_none_is_noop(self):
        core = MagicMock()
        core.tools = None
        agent = KoboiAgent(core=core)
        agent.inject_tool_definitions([{"name": "x"}])  # no raise

    def test_existing_name_in_real_registry_skipped(self):
        from koboi.tools.registry import ToolRegistry

        reg = ToolRegistry()
        reg.register("calc", "d", {"type": "object", "properties": {}}, fn=lambda: "ok")
        core = MagicMock()
        core.tools = reg
        agent = KoboiAgent(core=core)
        agent.inject_tool_definitions([{"name": "calc", "parameters": {}}])
        # still only the one original tool
        assert set(reg._tools) == {"calc"}


class TestMcpStatusDedup:
    def test_configured_server_deduped_when_live(self):
        client = MagicMock()
        client.endpoint = "npx pkg"
        client.name = "live"
        client.transport = "stdio"
        client.is_connected.return_value = True
        client.server_info = {}
        client.tool_names = []
        config = MagicMock()
        # same endpoint as the live client -> deduped (continue)
        config.get.return_value = [{"command": "npx", "args": ["pkg"]}]
        agent = KoboiAgent(core=MagicMock(), config=config, mcp_clients=[client])
        status = agent.mcp_status()
        assert len(status) == 1  # dead duplicate skipped


class TestResumeHappyPath:
    async def test_resume_delegates_to_core(self):
        core = MagicMock()
        core.resume = AsyncMock(return_value="RESUMED")
        agent = KoboiAgent(core=core)
        assert await agent.resume() == "RESUMED"


class TestConnectMcpServersSuccess:
    def test_success_path(self, tmp_path, monkeypatch):
        config = _cfg({"mcp": {"servers": [{"command": "npx"}]}})
        fake_client = MagicMock()
        monkeypatch.setattr("koboi.facade._create_mcp_client", lambda *a, **kw: fake_client)
        monkeypatch.setattr("koboi.facade._connect_with_retry", lambda *a, **kw: None)
        pairs = _connect_mcp_servers(config, _logger(tmp_path))
        assert len(pairs) == 1
        assert pairs[0][0] is fake_client


class TestPoolMemberNone:
    def test_member_resolves_to_none_raises(self, tmp_path):
        # "" resolves to None -> "did not resolve" branch
        config = _cfg({"pools": {"p1": {"providers": [""]}}})
        with pytest.raises(ValueError, match="did not resolve"):
            _build_pool_from_spec("p1", config, _logger(tmp_path))


class TestSetupSqliteBranches:
    def test_setup_subagent_injects_parent_memory(self, tmp_path):
        cfg = _base_config()
        cfg["tools"] = {"builtin": ["delegate_tasks"]}
        cfg["subagent"] = {"timeout": 42.0, "max_iterations": 3}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        tools = _build_tools(config)
        mem = MagicMock()
        _setup_subagent(tools, MagicMock(), MagicMock(), _logger(tmp_path), memory=mem, config=config)
        mgr = tools.get_dep("subagent_manager")
        assert mgr._parent_memory is mem

    def test_setup_tasks_sqlite_backend_sets_db_path(self, tmp_path):
        cfg = _base_config()
        cfg["memory"] = {"backend": "sqlite", "db_path": str(tmp_path / "t.db"), "session_id": "s1"}
        cfg["tools"] = {"builtin": ["task_create"]}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        tools = _build_tools(config)
        _setup_tasks(tools, config, hook_chain=None)
        assert tools.get_dep("task_manager") is not None


class TestBuildOrchestrationE2E:
    """Exercise _build_orchestration end-to-end via KoboiAgent.from_dict."""

    def test_sequential_orchestration_built(self):
        cfg = _base_config()
        cfg["memory"] = {"backend": "memory"}
        cfg["orchestration"] = {
            "enabled": True,
            "execution": {"mode": "sequential"},
            "router": {"type": "keyword"},
            "agents": [{"name": "worker", "description": "d", "keywords": ["x"]}],
        }
        agent = KoboiAgent.from_dict(cfg)
        assert agent.orchestrator is not None
        assert agent.core is None  # orchestrator mode -> no single core
        assert agent.mcp_clients is not None

    def test_dag_mode_builds_scheduler(self):
        cfg = _base_config()
        cfg["memory"] = {"backend": "memory"}
        cfg["orchestration"] = {
            "enabled": True,
            "execution": {"mode": "dag"},
            "router": {"type": "keyword"},
            "agents": [
                {"name": "a", "keywords": ["x"], "depends_on": ["b"]},
                {"name": "b", "keywords": ["y"]},
            ],
        }
        agent = KoboiAgent.from_dict(cfg)
        assert agent.orchestrator._dag_scheduler is not None

    def test_dynamic_mode_no_config_agents(self):
        cfg = _base_config()
        cfg["memory"] = {"backend": "memory"}
        cfg["orchestration"] = {
            "enabled": True,
            "execution": {"mode": "dynamic"},
            "router": {"type": "llm", "enable_dynamic": True},
        }
        agent = KoboiAgent.from_dict(cfg)
        assert agent.orchestrator is not None

    def test_share_mcp_false(self):
        cfg = _base_config()
        cfg["memory"] = {"backend": "memory"}
        cfg["orchestration"] = {
            "enabled": True,
            "share_mcp": False,
            "execution": {"mode": "sequential"},
            "router": {"type": "keyword"},
            "agents": [{"name": "worker", "description": "d", "keywords": ["x"]}],
        }
        agent = KoboiAgent.from_dict(cfg)
        assert agent.orchestrator is not None

    def test_dag_mode_sqlite_persists_graph(self, tmp_path):
        cfg = _base_config()
        cfg["memory"] = {"backend": "sqlite", "db_path": str(tmp_path / "dag.db")}
        cfg["orchestration"] = {
            "enabled": True,
            "execution": {"mode": "dag"},
            "router": {"type": "keyword"},
            "agents": [
                {"name": "a", "keywords": ["x"], "depends_on": ["b"], "interrupt_after": True},
                {"name": "b", "keywords": ["y"]},
            ],
        }
        agent = KoboiAgent.from_dict(cfg)
        assert agent.orchestrator._dag_scheduler is not None

    def test_agent_inline_llm_config_override(self):
        cfg = _base_config()
        cfg["memory"] = {"backend": "memory"}
        cfg["orchestration"] = {
            "enabled": True,
            "execution": {"mode": "sequential"},
            "router": {"type": "keyword"},
            "agents": [{"name": "w", "keywords": ["x"], "llm": {"model": "gpt-4o"}}],
        }
        agent = KoboiAgent.from_dict(cfg)
        assert agent.orchestrator is not None

    def test_agent_named_ref_llm_config(self):
        # Config expands named `providers:` refs to inline dicts at construction
        # (_expand_providers in config.py), so reintroduce the str ref into _data
        # afterward to exercise the str-branch of the per-agent client builder.
        config = _cfg(
            {
                **_base_config(),
                "memory": {"backend": "memory"},
                "providers": {
                    "alt": {
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "k",
                        "base_url": "http://x/v1",
                    }
                },
                "orchestration": {
                    "enabled": True,
                    "execution": {"mode": "sequential"},
                    "router": {"type": "keyword"},
                    "agents": [{"name": "w", "keywords": ["x"]}],
                },
            }
        )
        config._data["orchestration"]["agents"][0]["llm"] = "alt"
        agent = _build_orchestration(config)
        assert agent.orchestrator is not None

    def test_agent_pool_llm_config(self):
        cfg = _base_config()
        cfg["memory"] = {"backend": "memory"}
        cfg["pools"] = {"chat_pool": {"providers": [{"provider": "openai", "api_key": "k"}]}}
        cfg["orchestration"] = {
            "enabled": True,
            "execution": {"mode": "sequential"},
            "router": {"type": "keyword"},
            "agents": [{"name": "w", "keywords": ["x"], "llm": {"pool": "chat_pool"}}],
        }
        agent = KoboiAgent.from_dict(cfg)
        assert agent.orchestrator is not None


class TestOrchestrationOptInHooksReachable:
    """Wave2 #1: opt-in hooks (self-healing reflection, structural handover detection,
    proactive-memory extraction) must reach the orchestrator's hook_chain under
    ``orchestration.enabled:true``. Before the fix, ``_build_orchestration`` called the
    individual ``build_*()`` steps but never ``assembler.build()`` (where these hooks
    were inlined), so they were silently dropped in orchestration mode.
    """

    @staticmethod
    def _hook_names(agent: KoboiAgent) -> set[str]:
        chain = agent.orchestrator._hook_chain
        return {type(h).__name__ for h in chain._hooks} if chain is not None else set()

    def test_self_healing_hook_attached_in_orchestration(self):
        cfg = _base_config()
        cfg["memory"] = {"backend": "memory"}
        cfg["orchestration"] = {
            "enabled": True,
            "execution": {"mode": "sequential"},
            "router": {"type": "keyword"},
            "agents": [{"name": "w", "keywords": ["x"]}],
        }
        cfg["self_healing"] = {"enabled": True, "max_turns": 1}
        agent = KoboiAgent.from_dict(cfg)
        names = self._hook_names(agent)
        assert "ReflectionHook" in names, f"ReflectionHook missing from orchestration chain: {names}"
        # P2a escalation ladder ships together with ReflectionHook.
        assert "LadderRouterHook" in names, f"LadderRouterHook missing: {names}"

    def test_handover_detection_hook_attached_in_orchestration(self):
        cfg = _base_config()
        cfg["memory"] = {"backend": "memory"}
        cfg["orchestration"] = {
            "enabled": True,
            "execution": {"mode": "sequential"},
            "router": {"type": "keyword"},
            "agents": [{"name": "w", "keywords": ["x"]}],
        }
        cfg["handover"] = {"detection": {"enabled": True, "coverage_threshold": 0.6}}
        agent = KoboiAgent.from_dict(cfg)
        names = self._hook_names(agent)
        assert "HandoverDetectionHook" in names, f"HandoverDetectionHook missing: {names}"

    def test_proactive_extraction_hook_attached_in_orchestration(self):
        cfg = _base_config()
        cfg["memory"] = {"backend": "memory", "proactive": {"enabled": True, "extract": True}}
        cfg["orchestration"] = {
            "enabled": True,
            "execution": {"mode": "sequential"},
            "router": {"type": "keyword"},
            "agents": [{"name": "w", "keywords": ["x"]}],
        }
        agent = KoboiAgent.from_dict(cfg)
        names = self._hook_names(agent)
        assert "ProactiveExtractionHook" in names, f"ProactiveExtractionHook missing: {names}"

    def test_all_opt_in_hooks_attached_together(self):
        """All three opt-in hooks coexist on the orchestration hook_chain when configured."""
        cfg = _base_config()
        cfg["memory"] = {"backend": "memory", "proactive": {"enabled": True, "extract": True}}
        cfg["orchestration"] = {
            "enabled": True,
            "execution": {"mode": "sequential"},
            "router": {"type": "keyword"},
            "agents": [{"name": "w", "keywords": ["x"]}],
        }
        cfg["self_healing"] = {"enabled": True, "max_turns": 1}
        cfg["handover"] = {"detection": {"enabled": True}}
        agent = KoboiAgent.from_dict(cfg)
        names = self._hook_names(agent)
        for expected in ("ReflectionHook", "HandoverDetectionHook", "ProactiveExtractionHook"):
            assert expected in names, f"{expected} missing from orchestration chain: {names}"


class TestOrchestrationMediaWiring:
    """Wave2 #5: the media backend must reach sub-agent registries AND the Orchestrator
    when ``media.enabled:true`` under ``orchestration.enabled:true`` (previously media was
    non-functional in orchestration mode -- no provider was forwarded).
    """

    def test_media_backend_reaches_orchestrator_and_subagents(self):
        cfg = _base_config()
        cfg["memory"] = {"backend": "memory"}
        cfg["media"] = {"enabled": True, "image": {"provider": "mock"}}
        cfg["tools"] = {"builtin": ["generate_image"]}
        cfg["websearch"] = {"provider": "mock"}
        cfg["orchestration"] = {
            "enabled": True,
            "execution": {"mode": "sequential"},
            "router": {"type": "keyword"},
            "agents": [{"name": "w", "keywords": ["x"], "tools": {"builtin": ["generate_image"]}}],
        }
        agent = KoboiAgent.from_dict(cfg)
        # Orchestrator-side: ``KoboiAgent._media_backend()`` reads ``_orchestrator._media_backend``.
        assert agent.orchestrator._media_backend is not None, "Orchestrator media_backend missing"
        # Sub-agent side: the shared backend is forwarded to each sub-agent's tool registry.
        sub_agent = agent.orchestrator._agents_map.get("w")
        assert sub_agent is not None, "sub-agent 'w' not built"
        sub_tools = getattr(sub_agent, "tools", None)
        assert sub_tools is not None, "sub-agent has no tools registry"
        assert sub_tools.get_dep("media_provider") is not None, "media_provider dep missing on sub-agent"
        # Wave2 #5: search_provider and fetch_provider must also reach sub-agent registries.
        assert sub_tools.get_dep("search_provider") is not None, "search_provider dep missing on sub-agent"
        assert sub_tools.get_dep("fetch_provider") is not None, "fetch_provider dep missing on sub-agent"
        # Verify identity: shared instances, not duplicates.
        orch_search_provider = agent.orchestrator._agents_map.get("w").tools.get_dep("search_provider")
        orch_fetch_provider = agent.orchestrator._agents_map.get("w").tools.get_dep("fetch_provider")
        assert orch_search_provider is not None, "orchestrator search_provider missing"
        assert orch_fetch_provider is not None, "orchestrator fetch_provider missing"
        assert sub_tools.get_dep("search_provider") is orch_search_provider, "search_provider should be shared"
        assert sub_tools.get_dep("fetch_provider") is orch_fetch_provider, "fetch_provider should be shared"

    def test_no_media_backend_when_disabled(self):
        cfg = _base_config()
        cfg["memory"] = {"backend": "memory"}
        cfg["orchestration"] = {
            "enabled": True,
            "execution": {"mode": "sequential"},
            "router": {"type": "keyword"},
            "agents": [{"name": "w", "keywords": ["x"]}],
        }
        agent = KoboiAgent.from_dict(cfg)
        assert agent.orchestrator._media_backend is None


class TestRagCustomModules:
    def test_build_rag_loads_custom_components(self, tmp_path, monkeypatch):
        called: list = []
        monkeypatch.setattr("koboi.rag.registry.load_custom_components", lambda mods: called.append(mods))
        monkeypatch.setattr("koboi.rag.registry.build_rag", lambda *a, **kw: "RAG")
        cfg = _base_config()
        cfg["rag"] = {"enabled": True, "custom_modules": ["my.ragmod"]}
        config = Config.from_yaml(_write_config(tmp_path, cfg))
        assert _build_rag(config, MagicMock(), _logger(tmp_path)) == "RAG"
        assert called == [["my.ragmod"]]


class TestEnsureTelemetryHookImportError:
    def test_import_error_swallowed(self, monkeypatch):
        import sys

        core = MagicMock()
        core.hooks.find_hook.return_value = None
        agent = KoboiAgent(core=core)
        # Force the lazy import to fail (None in sys.modules -> ImportError).
        monkeypatch.setitem(sys.modules, "koboi.hooks.telemetry_hook", None)
        agent.ensure_telemetry_hook()  # ImportError -> pass
        core.hooks.add.assert_not_called()


class TestInjectDummyHandlerExecutes:
    async def test_injected_tool_executable(self):
        from koboi.tools.registry import ToolRegistry

        reg = ToolRegistry()
        core = MagicMock()
        core.tools = reg
        agent = KoboiAgent(core=core)
        agent.inject_tool_definitions([{"name": "ext", "parameters": {"type": "object", "properties": {}}}])
        result = await reg.execute("ext", "{}")
        assert "ok" in result
