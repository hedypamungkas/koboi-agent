"""Tests for TUI screens -- subagent monitor helper functions."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from koboi.tui.screens.subagent_monitor import _STATUS_SYMBOLS


class TestStatusSymbols:
    def test_all_symbols_defined(self):
        assert "pending" in _STATUS_SYMBOLS
        assert "running" in _STATUS_SYMBOLS
        assert "done" in _STATUS_SYMBOLS
        assert "failed" in _STATUS_SYMBOLS

    def test_symbols_are_markup_strings(self):
        for key, val in _STATUS_SYMBOLS.items():
            assert isinstance(val, str)
            assert len(val) > 0


class TestSubagentMonitorLogic:
    """Test the pure logic of SubagentMonitorScreen without Textual widgets."""

    def test_build_summary_empty(self):
        from koboi.tui.screens.subagent_monitor import SubagentMonitorScreen
        screen = SubagentMonitorScreen.__new__(SubagentMonitorScreen)
        screen._agent_states = {}
        summary = screen._build_summary()
        assert "No agents" in summary

    def test_build_summary_with_states(self):
        from koboi.tui.screens.subagent_monitor import SubagentMonitorScreen
        screen = SubagentMonitorScreen.__new__(SubagentMonitorScreen)
        screen._agent_states = {
            "a1": {"status": "running"},
            "a2": {"status": "done"},
            "a3": {"status": "failed"},
        }
        summary = screen._build_summary()
        assert "3 agent(s)" in summary
