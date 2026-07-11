"""Tests for koboi/web/registry.py -- provider registry + build_search_provider."""

from __future__ import annotations

import pytest

from koboi.web import build_search_provider, load_custom_components, search_provider_registry
from koboi.web.providers.brave import BraveSearchProvider
from koboi.web.providers.firecrawl import FirecrawlSearchProvider
from koboi.web.providers.mock import MockSearchProvider
from koboi.web.registry import ProviderRegistry, _redact


class TestBuiltinsRegistered:
    def test_all_builtins_registered(self):
        available = search_provider_registry.list_available()
        for name in ("mock", "ddg", "brave", "firecrawl"):
            assert name in available, f"{name} not registered"


class TestBuildSearchProvider:
    def test_default_is_mock(self):
        assert isinstance(build_search_provider(None), MockSearchProvider)

    def test_build_mock_explicit(self):
        provider = build_search_provider({"search": {"provider": "mock"}})
        assert isinstance(provider, MockSearchProvider)

    def test_build_brave_from_config(self):
        provider = build_search_provider({"search": {"provider": "brave", "brave": {"api_key": "k", "country": "US"}}})
        assert isinstance(provider, BraveSearchProvider)
        assert provider._api_key == "k"
        assert provider._country == "US"

    def test_build_firecrawl_from_config(self):
        provider = build_search_provider(
            {"search": {"provider": "firecrawl", "firecrawl": {"api_key": "fc", "scrape_results": True}}}
        )
        assert isinstance(provider, FirecrawlSearchProvider)
        assert provider._api_key == "fc"
        assert provider._scrape_results is True

    def test_unknown_falls_back_to_mock(self, caplog):
        provider = build_search_provider({"search": {"provider": "nonexistent_xyz"}})
        assert isinstance(provider, MockSearchProvider)
        assert any("Unknown search provider" in r.message for r in caplog.records)

    def test_shared_max_results_knob_flows_to_provider(self):
        provider = build_search_provider({"search": {"provider": "mock", "max_results": 5}})
        assert isinstance(provider, MockSearchProvider)
        assert provider._max_results == 5

    def test_per_provider_knob_overrides_shared(self):
        # web.search.max_results=9 (shared) but web.search.brave.max_results overrides.
        provider = build_search_provider(
            {"search": {"provider": "brave", "max_results": 9, "brave": {"api_key": "k", "max_results": 3}}}
        )
        assert isinstance(provider, BraveSearchProvider)
        assert provider._max_results == 3


class TestSecretRedaction:
    def test_redact_masks_credentials(self):
        out = _redact({"api_key": "secret", "token": "tok", "country": "US"})
        assert out["api_key"] == "***"
        assert out["token"] == "***"
        assert out["country"] == "US"

    def test_redact_leaves_empty_secret_as_is(self):
        # An empty value is not a secret leak; left untouched (falsy -> not masked).
        out = _redact({"api_key": ""})
        assert out["api_key"] == ""


class TestCustomModules:
    def test_bad_module_path_warns(self, caplog):
        load_custom_components(["definitely.not.a.real.module.path.xyz"])
        assert any("Failed to import custom web module" in r.message for r in caplog.records)


class TestConfigAliasesValidation:
    def test_bad_alias_raises_at_register(self):
        class _FakeProvider:
            def __init__(self, real_param: str = "x") -> None:
                self.real_param = real_param

        reg = ProviderRegistry("test")
        with pytest.raises(ValueError, match="config_aliases"):
            reg.register("_fake", _FakeProvider, config_aliases={"yaml_key": "nonexistent_param"})

    def test_good_alias_registers(self):
        class _FakeProvider:
            def __init__(self, real_param: str = "x") -> None:
                self.real_param = real_param

        reg = ProviderRegistry("test")
        reg.register("_fake2", _FakeProvider, config_aliases={"yaml_key": "real_param"})
        assert reg.get("_fake2") is not None
