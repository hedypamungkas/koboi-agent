"""tests/test_provider_resolver.py -- Tier 0/1 multi-provider config resolution.

Covers:
- ``resolve_llm_spec``: inline dict (Tier 0) unchanged; named ``providers:`` ref
  (Tier 1, str); ``{pool: name}`` (Tier 2) raises until W2; clear errors.
- ``Config.providers`` / ``Config.pools`` properties.
- ``_build_client`` with a top-level named ref (Tier 1) and inline (Tier 0 back-compat).
- Per-agent provider: a named ref fully replaces; an inline dict merges.
"""

from __future__ import annotations

import yaml
import pytest

from koboi.config import Config
from koboi.facade import _build_client, _build_client_from_dict
from koboi.llm.resolve import resolve_llm_spec


def _config(tmp_path, yaml_text) -> Config:
    """Config with validation OFF (unit tests only need resolution + build, which
    still run _walk_resolve + _expand_provider_refs in __init__)."""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_text)
    return Config(yaml.safe_load(p.read_text()), validate=False)


# ---------------------------------------------------------------------------
# resolve_llm_spec
# ---------------------------------------------------------------------------


class TestResolveLlmSpec:
    def test_none_and_empty_return_none(self, tmp_path):
        cfg = _config(tmp_path, "llm: { provider: openai, api_key: k }\n")
        assert resolve_llm_spec(None, cfg) is None
        assert resolve_llm_spec("", cfg) is None

    def test_named_ref_resolves_to_provider_dict(self, tmp_path):
        cfg = _config(
            tmp_path,
            "providers:\n  primary: { provider: openai, model: gpt-5.4, api_key: k }\n",
        )
        resolved = resolve_llm_spec("primary", cfg)
        assert resolved == {"provider": "openai", "model": "gpt-5.4", "api_key": "k"}

    def test_unknown_named_ref_raises_with_available_list(self, tmp_path):
        cfg = _config(
            tmp_path,
            "providers:\n  primary: { provider: openai, api_key: k }\n",
        )
        with pytest.raises(ValueError, match="Unknown provider reference 'missing'"):
            resolve_llm_spec("missing", cfg)

    def test_inline_dict_returned_as_copy(self, tmp_path):
        cfg = _config(tmp_path, "llm: { provider: openai }\n")
        spec = {"provider": "anthropic", "model": "claude-x"}
        resolved = resolve_llm_spec(spec, cfg)
        assert resolved == spec
        assert resolved is not spec  # defensive copy

    def test_pool_form_raises_not_implemented(self, tmp_path):
        cfg = _config(tmp_path, "providers: {}\n")
        with pytest.raises(NotImplementedError, match="W2"):
            resolve_llm_spec({"pool": "resilient"}, cfg)

    def test_wrong_type_raises(self, tmp_path):
        cfg = _config(tmp_path, "llm: { provider: openai }\n")
        with pytest.raises(TypeError):
            resolve_llm_spec(123, cfg)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Config properties
# ---------------------------------------------------------------------------


class TestConfigProperties:
    def test_providers_and_pools_properties(self, tmp_path):
        cfg = _config(
            tmp_path,
            "providers:\n  a: { provider: openai }\n  b: { provider: anthropic }\n"
            "pools:\n  resilient: { providers: [a, b], policy: failover }\n",
        )
        assert set(cfg.providers.keys()) == {"a", "b"}
        assert cfg.pools["resilient"]["policy"] == "failover"

    def test_missing_sections_default_to_empty(self, tmp_path):
        cfg = _config(tmp_path, "llm: { provider: openai, api_key: k }\n")
        assert cfg.providers == {}
        assert cfg.pools == {}

    def test_named_top_level_ref_passes_strict_validation(self, tmp_path):
        """from_yaml validates strictly (llm.model required); a named ref must be
        pre-expanded to an inline dict so validation passes (Tier 1 unblocked)."""
        p = tmp_path / "cfg.yaml"
        p.write_text(
            "agent: { name: x }\n"
            "providers:\n  primary: { provider: openai, model: gpt-5.4, api_key: test-key }\n"
            "llm: primary\n"
        )
        cfg = Config.from_yaml(str(p))  # validate=True; fails without pre-expansion
        assert cfg.model == "gpt-5.4"


# ---------------------------------------------------------------------------
# _build_client: Tier 0 (inline) vs Tier 1 (named ref)
# ---------------------------------------------------------------------------


class TestBuildClientResolution:
    def test_inline_top_level_is_backward_compatible(self, tmp_path):
        cfg = _config(
            tmp_path,
            "llm: { provider: openai, model: gpt-5.4, api_key: test-key, base_url: http://x }\n",
        )
        client = _build_client(cfg, logger=None)
        assert client.model == "gpt-5.4"
        assert client.provider == "openai"

    def test_named_top_level_ref_uses_provider_spec(self, tmp_path):
        cfg = _config(
            tmp_path,
            "providers:\n  primary: { provider: openai, model: gpt-5.4, api_key: test-key-a, base_url: http://a }\n"
            "llm: primary\n",
        )
        client = _build_client(cfg, logger=None)
        assert client.model == "gpt-5.4"
        assert client.api_key == "test-key-a"

    def test_per_agent_inline_overrides_merge_over_top_level(self, tmp_path):
        cfg = _config(
            tmp_path,
            "providers:\n  base: { provider: openai, model: gpt-5.4, api_key: test-key, base_url: http://x }\n"
            "llm: base\n",
        )
        # An agent inline-override of just temperature merges over the base spec.
        client = _build_client(cfg, logger=None, llm_overrides={"temperature": 0.1})
        assert client.model == "gpt-5.4"  # inherited from base
        assert client.temperature == 0.1  # overridden


# ---------------------------------------------------------------------------
# Per-agent provider: named ref (full replace) vs inline dict (merge)
# Mirrors the facade._agent_client_builder closure decision.
# ---------------------------------------------------------------------------


class TestPerAgentProviderDecision:
    def test_named_ref_full_replaces_top_level(self, tmp_path):
        cfg = _config(
            tmp_path,
            "providers:\n"
            "  base: { provider: openai, model: gpt-5.4, api_key: test-key, base_url: http://x }\n"
            "  reasoner: { provider: anthropic, model: claude-sonnet-5, api_key: test-key-r, base_url: http://r }\n"
            "llm: base\n",
        )
        # str path: full replace (does NOT inherit base's model)
        resolved = resolve_llm_spec("reasoner", cfg)
        client = _build_client_from_dict(resolved, logger=None)
        assert client.model == "claude-sonnet-5"
        assert client.provider == "anthropic"
        assert client.api_key == "test-key-r"

    def test_inline_agent_dict_merges_over_top_level(self, tmp_path):
        cfg = _config(
            tmp_path,
            "providers:\n  base: { provider: openai, model: gpt-5.4, api_key: test-key, base_url: http://x }\n"
            "llm: base\n",
        )
        # dict path: merge over top-level (today's behavior)
        overrides = {"temperature": 0.7}
        client = _build_client(cfg, logger=None, llm_overrides=overrides)
        assert client.model == "gpt-5.4"  # from base
        assert client.temperature == 0.7  # agent override
