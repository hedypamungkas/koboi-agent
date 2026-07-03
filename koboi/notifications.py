"""koboi/notifications.py -- Desktop notifications and sound alerts."""

from __future__ import annotations

import subprocess
import sys


def notify(title: str, message: str, sound: bool = False) -> None:
    """Send a desktop notification using platform-native commands.

    Args:
        title: Notification title.
        message: Notification body text.
        sound: If True, play a system alert sound with the notification.
    """
    try:
        if sys.platform == "darwin":
            _notify_macos(title, message, sound=sound)
        elif sys.platform.startswith("linux"):
            _notify_linux(title, message, sound=sound)
        elif sys.platform == "win32":
            _notify_windows(title, message)
    except Exception:  # nosec B110 - best-effort; intentionally swallows transient errors (cleanup/export/teardown)
        pass


def play_sound(sound_name: str = "default") -> None:
    """Play a system alert sound.

    Args:
        sound_name: Sound identifier. "default" uses the system default.
                    On macOS, any NSSound name (e.g., "Ping", "Glass", "Hero").
    """
    try:
        if sys.platform == "darwin":
            _play_sound_macos(sound_name)
        elif sys.platform.startswith("linux"):
            _play_sound_linux()
        elif sys.platform == "win32":
            _play_sound_windows()
    except Exception:  # nosec B110 - best-effort; intentionally swallows transient errors (cleanup/export/teardown)
        pass


def _notify_macos(title: str, message: str, *, sound: bool = False) -> None:
    sound_clause = ' sound name "Ping"' if sound else ""
    script = f'display notification "{message}" with title "{title}"{sound_clause}'
    subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)  # nosec B607 - intentional PATH-based launch of a user tool/editor


def _notify_linux(title: str, message: str, *, sound: bool = False) -> None:
    subprocess.run(["notify-send", title, message], capture_output=True, timeout=5)  # nosec B607 - intentional PATH-based launch of a user tool/editor
    if sound:
        _play_sound_linux()


def _notify_windows(title: str, message: str) -> None:
    try:
        from plyer import notification

        notification.notify(title=title, message=message, timeout=5)
    except ImportError:
        pass


def _play_sound_macos(sound_name: str) -> None:
    if sound_name == "default":
        sound_name = "Ping"
    subprocess.run(  # nosec B607 - intentional PATH-based launch of a user tool/editor
        ["osascript", "-e", f'play sound "{sound_name}"'],
        capture_output=True,
        timeout=5,
    )


def _play_sound_linux() -> None:
    for cmd in (
        ["paplay", "/usr/share/sounds/freedesktop/stereo/complete.oga"],
        ["aplay", "/usr/share/sounds/alsa/Front_Center.wav"],
        ["beep"],
    ):
        try:
            subprocess.run(cmd, capture_output=True, timeout=5)
            return
        except FileNotFoundError:
            continue


def _play_sound_windows() -> None:
    import winsound

    winsound.MessageBeep()  # type: ignore[attr-defined]  # winsound is Windows-only; stub unavailable on other platforms
