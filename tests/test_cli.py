"""Tests for koboi/cli -- console-script entry point."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestCliMain:
    """Tests for the cli.main() entry point."""

    def test_main_importable(self):
        """cli.py should be importable without TUI dependencies."""
        import koboi.cli

        assert koboi.cli is not None

    def test_main_delegates_to_tui_main(self):
        """When TUI deps are available, main() delegates to tui.app.main."""
        mock_tui_main = MagicMock()
        mock_module = MagicMock()
        mock_module.main = mock_tui_main
        with patch.dict("sys.modules", {"koboi.tui.app": mock_module}):
            from importlib import reload

            import koboi.cli

            reload(koboi.cli)
            koboi.cli.main()
            mock_tui_main.assert_called_once()

    def test_main_exits_gracefully_on_import_error(self):
        """When TUI deps are missing, main() exits with code 1."""
        import koboi.cli

        # Setting a module to None in sys.modules blocks the import
        with patch.dict("sys.modules", {"koboi.tui.app": None}):
            with pytest.raises(SystemExit) as exc_info:
                koboi.cli.main()
            assert exc_info.value.code == 1

    def test_main_error_message_mentions_tui_extra(self, capsys):
        """Error message should mention pip install koboi-agent[tui]."""
        import koboi.cli

        with patch.dict("sys.modules", {"koboi.tui.app": None}):
            with pytest.raises(SystemExit):
                koboi.cli.main()

        captured = capsys.readouterr()
        assert "pip install koboi-agent[tui]" in captured.err

    def test_main_error_message_mentions_all_extra(self, capsys):
        """Error message should also mention the [all] extra."""
        import koboi.cli

        with patch.dict("sys.modules", {"koboi.tui.app": None}):
            with pytest.raises(SystemExit):
                koboi.cli.main()

        captured = capsys.readouterr()
        assert "pip install koboi-agent[all]" in captured.err
