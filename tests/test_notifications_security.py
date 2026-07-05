"""tests/test_notifications_security -- AppleScript-injection regression.

``NotificationHook`` builds the macOS notification ``message`` from the first
~100 chars of the LLM response (driven by untrusted prompts / RAG docs) and
hands it to ``_notify_macos``, which used to f-string-interpolate it straight
into an ``osascript -e`` AppleScript source string. A ``"`` broke out and
``do shell script "..."`` ran arbitrary shell -- outside the tool pipeline.

These tests pin the fix: ``message``/``title`` must travel as argv (data), and
``sound_name`` (which ``play sound`` won't accept as an argv expression) must be
backslash-escaped so it cannot break out of the AppleScript string.
"""

from __future__ import annotations

from unittest.mock import patch

from koboi.notifications import _notify_macos, _play_sound_macos

# The exact AppleScript-concatenation breakout that achieved RCE pre-fix.
PAYLOAD = 'Sure!" & (do shell script "touch /tmp/x") & "'


class TestNotifyAppleScriptInjection:
    @patch("koboi.notifications.subprocess")
    def test_message_is_argv_not_script_source(self, mock_subproc):
        _notify_macos("Koboi Agent", PAYLOAD, sound=False)
        args = mock_subproc.run.call_args[0][0]
        assert args[0] == "osascript"
        assert args[1] == "-e"
        script = args[2]
        # message + title ride as separate argv elements (pure string values)
        assert args[3] == PAYLOAD
        assert args[4] == "Koboi Agent"
        # the handler references argv items, NOT the message text
        assert "on run(argv)" in script
        assert "(item 1 of argv)" in script
        assert "(item 2 of argv)" in script
        # CRITICAL: the payload must not be parsed as AppleScript source
        assert PAYLOAD not in script
        assert "do shell script" not in script

    @patch("koboi.notifications.subprocess")
    def test_injection_payload_inert(self, mock_subproc):
        _notify_macos("Koboi Agent", PAYLOAD)
        script = mock_subproc.run.call_args[0][0][2]
        assert "do shell script" not in script
        assert "& (" not in script  # no AppleScript concat injected from payload

    @patch("koboi.notifications.subprocess")
    def test_benign_message_still_routed(self, mock_subproc):
        _notify_macos("Koboi Agent", "build completed")
        assert mock_subproc.run.call_count == 1
        args = mock_subproc.run.call_args[0][0]
        assert args[3] == "build completed"
        assert args[4] == "Koboi Agent"

    @patch("koboi.notifications.subprocess")
    def test_sound_flag_keeps_argv_handler_form(self, mock_subproc):
        _notify_macos("Koboi Agent", "x", sound=True)
        script = mock_subproc.run.call_args[0][0][2]
        assert "on run(argv)" in script
        assert 'sound name "Ping"' in script  # fixed operator constant only
        assert "do shell script" not in script


class TestPlaySoundMacosEscaping:
    @patch("koboi.notifications.subprocess")
    def test_default_sound_unchanged(self, mock_subproc):
        _play_sound_macos("default")
        script = mock_subproc.run.call_args[0][0][2]
        assert script == 'play sound "Ping"'

    @patch("koboi.notifications.subprocess")
    def test_quote_in_sound_name_is_escaped(self, mock_subproc):
        payload = 'Ping" do shell script "touch /tmp/x"'
        _play_sound_macos(payload)
        script = mock_subproc.run.call_args[0][0][2]
        # the implementation's own escaping, reproduced, must match exactly
        expected_inner = payload.replace("\\", "\\\\").replace('"', '\\"')
        assert script == f'play sound "{expected_inner}"'
        # every quote inside the body is backslash-escaped -> cannot break out
        body = script[len('play sound "') : -1]
        for i, ch in enumerate(body):
            assert ch != '"' or (i > 0 and body[i - 1] == "\\"), f"unescaped quote at {i}"
