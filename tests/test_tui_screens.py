"""Tests for TUI screens -- permission_dialog (no Textual widget instantiation)."""

from __future__ import annotations

import pytest

from koboi.tui.screens.permission_dialog import PermissionDialog, PermissionResult


class TestPermissionResult:
    def test_approved(self):
        r = PermissionResult(approved=True, always_allow=True)
        assert r.approved is True
        assert r.always_allow is True

    def test_denied(self):
        r = PermissionResult(approved=False, always_allow=False)
        assert r.approved is False
        assert r.always_allow is False


class TestPermissionDialog:
    def test_format_args_short(self):
        result = PermissionDialog._format_args_preview("short args")
        assert result == "short args"

    def test_format_args_long(self):
        long_args = "x" * 500
        result = PermissionDialog._format_args_preview(long_args, max_len=300)
        assert len(result) == 303  # 300 + "..."
        assert result.endswith("...")

    def test_format_args_exact_max(self):
        args = "x" * 300
        result = PermissionDialog._format_args_preview(args, max_len=300)
        assert result == args

    def test_is_modal_screen(self):
        from textual.screen import ModalScreen

        assert issubclass(PermissionDialog, ModalScreen)
