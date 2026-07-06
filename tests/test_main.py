"""Tests for koboi/__main__.py entry point."""

from __future__ import annotations

import runpy
from unittest.mock import patch


class TestMainModule:
    def test_main_module_importable(self):
        """__main__.py should be importable without error."""
        import koboi.__main__

        assert koboi.__main__ is not None

    @patch("koboi.cli.main")
    def test_main_entry_point(self, mock_main):
        """Running as `python -m koboi` should call koboi.cli.main()."""
        runpy.run_module("koboi", run_name="__main__", alter_sys=True)
        mock_main.assert_called_once()
