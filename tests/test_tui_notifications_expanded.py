"""Tests for koboi/notifications.py -- platform dispatch and sound."""

from __future__ import annotations

from unittest.mock import patch


from koboi.notifications import (
    notify,
    play_sound,
    _notify_macos,
    _notify_linux,
    _play_sound_macos,
)


class TestNotify:
    @patch("koboi.notifications.sys")
    @patch("koboi.notifications._notify_macos")
    def test_notify_macos(self, mock_macos, mock_sys):
        mock_sys.platform = "darwin"
        notify("Title", "Message", sound=True)
        mock_macos.assert_called_once_with("Title", "Message", sound=True)

    @patch("koboi.notifications.sys")
    @patch("koboi.notifications._notify_linux")
    def test_notify_linux(self, mock_linux, mock_sys):
        mock_sys.platform = "linux"
        notify("Title", "Message")
        mock_linux.assert_called_once_with("Title", "Message", sound=False)

    @patch("koboi.notifications.sys")
    @patch("koboi.notifications._notify_windows")
    def test_notify_windows(self, mock_windows, mock_sys):
        mock_sys.platform = "win32"
        notify("Title", "Message")
        mock_windows.assert_called_once_with("Title", "Message")

    @patch("koboi.notifications.sys")
    @patch("koboi.notifications._notify_macos", side_effect=Exception("fail"))
    def test_notify_exception_swallowed(self, mock_macos, mock_sys):
        mock_sys.platform = "darwin"
        notify("Title", "Message")  # Should not raise


class TestPlaySound:
    @patch("koboi.notifications.sys")
    @patch("koboi.notifications._play_sound_macos")
    def test_play_sound_macos(self, mock_macos, mock_sys):
        mock_sys.platform = "darwin"
        play_sound("Glass")
        mock_macos.assert_called_once_with("Glass")

    @patch("koboi.notifications.sys")
    @patch("koboi.notifications._play_sound_linux")
    def test_play_sound_linux(self, mock_linux, mock_sys):
        mock_sys.platform = "linux"
        play_sound()
        mock_linux.assert_called_once()

    @patch("koboi.notifications.sys")
    @patch("koboi.notifications._play_sound_windows")
    def test_play_sound_windows(self, mock_windows, mock_sys):
        mock_sys.platform = "win32"
        play_sound()
        mock_windows.assert_called_once()

    @patch("koboi.notifications.sys")
    @patch("koboi.notifications._play_sound_macos", side_effect=Exception("fail"))
    def test_play_sound_exception_swallowed(self, mock_macos, mock_sys):
        mock_sys.platform = "darwin"
        play_sound()  # Should not raise


class TestPlatformFunctions:
    @patch("koboi.notifications.subprocess")
    def test_notify_macos_calls_osascript(self, mock_subprocess):
        _notify_macos("Title", "Msg", sound=True)
        mock_subprocess.run.assert_called_once()
        args = mock_subprocess.run.call_args[0][0]
        assert "osascript" in args

    @patch("koboi.notifications.subprocess")
    def test_notify_linux_calls_notify_send(self, mock_subprocess):
        _notify_linux("Title", "Msg")
        mock_subprocess.run.assert_called()

    @patch("koboi.notifications.subprocess")
    def test_play_sound_macos_calls_osascript(self, mock_subprocess):
        _play_sound_macos("Ping")
        mock_subprocess.run.assert_called_once()

    @patch("koboi.notifications.subprocess")
    def test_play_sound_macos_default(self, mock_subprocess):
        _play_sound_macos("default")
        args = mock_subprocess.run.call_args[0][0]
        assert "Ping" in str(args)
