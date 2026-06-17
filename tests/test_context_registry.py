"""Tests for koboi/context/registry.py -- Context strategy registry."""

from __future__ import annotations

import copy

import pytest

from koboi.context.registry import (
    context_registry,
    register_context_strategy,
    build_context,
    load_custom_context_modules,
)
from koboi.context.manager import (
    ContextManager,
    TruncationManager,
    SmartTruncationManager,
    KeyFactsManager,
    SlidingWindowManager,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Save and restore registry state for test isolation."""
    saved = copy.deepcopy(context_registry._entries)
    yield
    context_registry._entries = saved


# ---------------------------------------------------------------------------
# Built-in registrations
# ---------------------------------------------------------------------------


class TestBuiltinRegistrations:
    def test_truncation_registered(self):
        assert "truncation" in context_registry.list_available()

    def test_smart_truncation_registered(self):
        assert "smart_truncation" in context_registry.list_available()

    def test_key_facts_registered(self):
        assert "key_facts" in context_registry.list_available()

    def test_sliding_window_registered(self):
        assert "sliding_window" in context_registry.list_available()

    def test_all_four_strategies_registered(self):
        assert len(context_registry.list_available()) >= 4


# ---------------------------------------------------------------------------
# register_context_strategy decorator
# ---------------------------------------------------------------------------


class TestRegisterContextStrategy:
    def test_register_and_get(self):
        @register_context_strategy("custom_strat", description="A custom strategy")
        class CustomManager(ContextManager):
            @property
            def _strategy_name(self):
                return "CUSTOM"

            async def _build_result(self, system_msgs, non_system):
                return system_msgs + non_system, "custom"

        entry = context_registry.get("custom_strat")
        assert entry is not None
        assert entry.cls is CustomManager
        assert entry.description == "A custom strategy"

    def test_unknown_strategy_returns_none(self):
        assert context_registry.get("nonexistent_strategy") is None


# ---------------------------------------------------------------------------
# build_context factory
# ---------------------------------------------------------------------------


class TestBuildContext:
    def test_noop_returns_none(self):
        result = build_context("noop")
        assert result is None

    def test_build_truncation(self):
        result = build_context("truncation", keep_last=10)
        assert isinstance(result, TruncationManager)
        assert result.keep_last == 10

    def test_build_smart_truncation(self):
        result = build_context("smart_truncation", keep_last=8)
        assert isinstance(result, SmartTruncationManager)
        assert result.keep_last == 8

    def test_build_key_facts(self):
        result = build_context("key_facts")
        assert isinstance(result, KeyFactsManager)

    def test_build_sliding_window_with_client(self):
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        result = build_context("sliding_window", client=mock_client, keep_last=5)
        assert isinstance(result, SlidingWindowManager)
        assert result.client is mock_client
        assert result.keep_last == 5

    def test_unknown_strategy_returns_none(self, caplog):
        result = build_context("nonexistent")
        assert result is None
        assert "Unknown context strategy" in caplog.text


# ---------------------------------------------------------------------------
# load_custom_context_modules
# ---------------------------------------------------------------------------


class TestLoadCustomContextModules:
    def test_import_failure_warns(self, caplog):
        load_custom_context_modules(["nonexistent_module_xyz"])
        assert "Failed to import custom context module" in caplog.text

    def test_empty_list_noop(self):
        load_custom_context_modules([])
