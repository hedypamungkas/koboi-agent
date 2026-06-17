"""Tests for koboi/plugins.py -- plugin discovery via entry_points."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from koboi.plugins import discover_plugins


class TestDiscoverPlugins:
    def test_returns_empty_when_no_plugins(self):
        """discover_plugins returns empty lists when no entry points exist."""
        with patch("koboi.plugins.entry_points", return_value=[]) as mock_ep:
            result = discover_plugins()
        assert all(v == [] for v in result.values())

    def test_loads_provider_plugin(self):
        """discovers and calls provider entry point factory."""
        factory = MagicMock()
        ep = MagicMock()
        ep.name = "test_provider"
        ep.load.return_value = factory

        # entry_points is called once per group; return the plugin only for providers
        def fake_entry_points(group=None):
            if group == "koboi.providers":
                return [ep]
            return []

        with patch("koboi.plugins.entry_points", side_effect=fake_entry_points):
            result = discover_plugins()

        factory.assert_called_once()
        assert "test_provider" in result["koboi.providers"]

    def test_handles_load_error_gracefully(self):
        """Failed plugin load does not raise."""
        ep = MagicMock()
        ep.name = "broken_plugin"
        ep.load.side_effect = ImportError("no module")

        def fake_entry_points(group=None):
            if group == "koboi.providers":
                return [ep]
            return []

        with patch("koboi.plugins.entry_points", side_effect=fake_entry_points):
            result = discover_plugins()

        assert "broken_plugin" not in result.get("koboi.providers", [])

    def test_groups_returned(self):
        """Result contains all expected group keys."""
        with patch("koboi.plugins.entry_points", return_value=[]):
            result = discover_plugins()
        assert "koboi.providers" in result
        assert "koboi.guardrails" in result
        assert "koboi.scorers" in result
        assert "koboi.tools" in result
