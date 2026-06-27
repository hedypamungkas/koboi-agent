"""Tests for koboi/notifications.py -- Desktop notifications."""

from __future__ import annotations

from unittest.mock import patch

from koboi.notifications import notify


class TestNotify:
    @patch("koboi.notifications.sys")
    @patch("koboi.notifications.subprocess")
    def test_notify_macos(self, mock_subprocess, mock_sys):
        mock_sys.platform = "darwin"
        notify("Title", "Message")
        mock_subprocess.run.assert_called_once()
        args = mock_subprocess.run.call_args[0][0]
        assert "osascript" in args

    @patch("koboi.notifications.sys")
    @patch("koboi.notifications.subprocess")
    def test_notify_linux(self, mock_subprocess, mock_sys):
        mock_sys.platform = "linux"
        notify("Title", "Message")
        mock_subprocess.run.assert_called_once()
        args = mock_subprocess.run.call_args[0][0]
        assert "notify-send" in args

    @patch("koboi.notifications.sys")
    def test_notify_windows_no_plyer(self, mock_sys):
        mock_sys.platform = "win32"
        # Should not raise even if plyer is not installed
        notify("Title", "Message")

    @patch("koboi.notifications.sys")
    @patch("koboi.notifications.subprocess", side_effect=Exception("fail"))
    def test_notify_exception_swallowed(self, mock_subprocess, mock_sys):
        mock_sys.platform = "darwin"
        # Should not raise
        notify("Title", "Message")
