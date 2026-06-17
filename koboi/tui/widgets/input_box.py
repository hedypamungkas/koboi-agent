"""InputBox -- text input with submit, suggester, history navigation, and vim mode."""
from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path

from textual.events import Key
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Input

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".svg"}
# Matches @path/to/file, @file.ext, or bare paths ending in image extensions
_IMAGE_PATH_RE = re.compile(
    r"@([\w./~-]+\.(?:png|jpg|jpeg|gif|webp|bmp|tiff|svg))\b",
    re.IGNORECASE,
)


class ChatSubmit(Message):
    """Posted when the user presses Enter with non-empty text."""

    def __init__(self, value: str, images: list[dict] | None = None) -> None:
        super().__init__()
        self.value = value
        self.images = images or []


class VimModeChanged(Message):
    """Posted when vim mode changes (normal/insert)."""

    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode


# Word boundary pattern for vim word motions
_WORD_RE = re.compile(r"\w+|[^\w\s]")


class InputBox(Input):
    """Input widget that posts ChatSubmit on Enter, with history and vim mode.

    History is managed externally via set_history() and add_to_history().
    Up/Down arrows navigate history when in insert mode.
    In normal mode, j/k navigate history instead.

    Vim mode (toggled via /vim command):
      Normal mode: h/j/k/l, w/b/e, i/a/o/A/I, x/X, dd, yy, p, 0/$, ^, u
      Insert mode: standard text input (same as default)
    """

    vim_enabled: reactive[bool] = reactive(False)
    vim_mode: reactive[str] = reactive("insert")

    def __init__(self, placeholder: str = "Type a message...", **kwargs):
        super().__init__(placeholder=placeholder, **kwargs)
        self._history: list[str] = []
        self._history_idx: int = -1
        self._editing_from_history: bool = False
        self._vim_buffer: str = ""
        self._vim_pending_d: bool = False
        self._vim_pending_y: bool = False

    def set_history(self, history: list[str]) -> None:
        """Set the history list (reference to app's history)."""
        self._history = history
        self._history_idx = -1

    def watch_vim_mode(self, mode: str) -> None:
        """Post a message when vim mode changes."""
        self.post_message(VimModeChanged(mode))

    def add_to_history(self, entry: str) -> None:
        """Add an entry to history. Called by the app after submit."""
        if entry and (not self._history or self._history[-1] != entry):
            self._history.append(entry)

    def on_key(self, event: Key) -> None:
        """Dispatch key handling based on vim mode."""
        if self.vim_enabled:
            if self.vim_mode == "normal":
                self._handle_vim_normal(event)
                return
            elif self.vim_mode == "insert":
                if event.key == "escape":
                    self.vim_mode = "normal"
                    self._clamp_cursor()
                    event.prevent_default()
                    return
                # In insert mode, handle history nav with up/down
                self._handle_history_nav(event)
            return

        # Non-vim mode: original behavior
        self._handle_history_nav(event)

    def _handle_history_nav(self, event: Key) -> None:
        """Handle Up/Down for history navigation (insert mode or non-vim)."""
        if event.key == "up":
            if not self._history:
                return
            if self._history_idx == -1:
                self._history_idx = len(self._history) - 1
            elif self._history_idx > 0:
                self._history_idx -= 1
            else:
                return
            self._editing_from_history = True
            self.value = self._history[self._history_idx]
            self.cursor_position = len(self.value)
            event.prevent_default()

        elif event.key == "down":
            if self._history_idx == -1:
                return
            if self._history_idx < len(self._history) - 1:
                self._history_idx += 1
                self.value = self._history[self._history_idx]
                self.cursor_position = len(self.value)
            else:
                self._history_idx = -1
                self.value = ""
            self._editing_from_history = True
            event.prevent_default()

    def _handle_vim_normal(self, event: Key) -> None:
        """Handle a key press in vim normal mode."""
        key = event.key
        val = self.value
        pos = self.cursor_position

        # Prevent default for all normal mode keys (no text insertion)
        event.prevent_default()

        # Pending operator combos (dd, yy)
        if self._vim_pending_d:
            self._vim_pending_d = False
            if key == "d":
                # dd -- delete entire line
                self._vim_buffer = val
                self.value = ""
                self.cursor_position = 0
                return
            # d{motion} -- delete to motion target
            target = self._motion_target(key, pos, val)
            if target is not None:
                start, end = min(pos, target), max(pos, target)
                self._vim_buffer = val[start:end]
                self.value = val[:start] + val[end:]
                self.cursor_position = start
            return

        if self._vim_pending_y:
            self._vim_pending_y = False
            if key == "y":
                # yy -- yank entire line
                self._vim_buffer = val
                return
            target = self._motion_target(key, pos, val)
            if target is not None:
                start, end = min(pos, target), max(pos, target)
                self._vim_buffer = val[start:end]
            return

        # -- Mode switching --
        if key == "i":
            self.vim_mode = "insert"
            return
        if key == "a":
            self.vim_mode = "insert"
            if pos < len(val):
                self.cursor_position = pos + 1
            return
        if key == "A":
            self.vim_mode = "insert"
            self.cursor_position = len(val)
            return
        if key == "I":
            self.vim_mode = "insert"
            self.cursor_position = 0
            return
        if key == "o":
            # Open below: in single-line input, just go to end + insert
            self.vim_mode = "insert"
            self.cursor_position = len(val)
            return

        # -- Character motion --
        if key == "h":
            if pos > 0:
                self.cursor_position = pos - 1
            return
        if key == "l":
            if pos < len(val):
                self.cursor_position = pos + 1
            return

        # -- Line motion --
        if key == "0":
            self.cursor_position = 0
            return
        if key == "dollar":  # $
            self.cursor_position = len(val) if val else 0
            return
        if key == "caret":  # ^
            # Move to first non-whitespace
            stripped = val.lstrip()
            if stripped:
                self.cursor_position = len(val) - len(stripped)
            else:
                self.cursor_position = 0
            return

        # -- Word motion --
        if key == "w":
            target = self._find_word_forward(val, pos)
            if target is not None:
                self.cursor_position = target
            return
        if key == "b":
            target = self._find_word_backward(val, pos)
            if target is not None:
                self.cursor_position = target
            return
        if key == "e":
            target = self._find_word_end(val, pos)
            if target is not None:
                self.cursor_position = target
            return

        # -- Editing --
        if key == "x":
            if val and pos < len(val):
                self._vim_buffer = val[pos]
                self.value = val[:pos] + val[pos + 1:]
                if self.cursor_position > len(self.value):
                    self.cursor_position = len(self.value)
            return
        if key == "X":
            if val and pos > 0:
                self._vim_buffer = val[pos - 1]
                self.value = val[:pos - 1] + val[pos:]
                self.cursor_position = pos - 1
            return
        if key == "d":
            self._vim_pending_d = True
            return
        if key == "y":
            self._vim_pending_y = True
            return
        if key == "p":
            if self._vim_buffer:
                insert_pos = min(pos + 1, len(val))
                self.value = val[:insert_pos] + self._vim_buffer + val[insert_pos:]
                self.cursor_position = insert_pos
            return
        if key == "u":
            # Undo: just clear (no real undo stack in Input widget)
            # A more sophisticated approach would maintain a history stack
            return

        # -- History navigation (j/k in normal mode) --
        if key == "j":
            if self._history_idx == -1:
                return
            if self._history_idx < len(self._history) - 1:
                self._history_idx += 1
                self.value = self._history[self._history_idx]
                self.cursor_position = min(self.cursor_position, len(self.value))
            else:
                self._history_idx = -1
                self.value = ""
                self.cursor_position = 0
            self._editing_from_history = True
            return
        if key == "k":
            if not self._history:
                return
            if self._history_idx == -1:
                self._history_idx = len(self._history) - 1
            elif self._history_idx > 0:
                self._history_idx -= 1
            else:
                return
            self._editing_from_history = True
            self.value = self._history[self._history_idx]
            self.cursor_position = min(self.cursor_position, len(self.value))
            return

    def _clamp_cursor(self) -> None:
        """Ensure cursor is within valid range."""
        if self.cursor_position > len(self.value):
            self.cursor_position = len(self.value)

    def _motion_target(self, key: str, pos: int, val: str) -> int | None:
        """Compute the target position for a vim motion key."""
        if key == "h":
            return max(0, pos - 1)
        if key == "l":
            return min(len(val), pos + 1)
        if key == "w":
            return self._find_word_forward(val, pos)
        if key == "b":
            return self._find_word_backward(val, pos)
        if key == "e":
            return self._find_word_end(val, pos)
        if key == "0":
            return 0
        if key == "dollar":
            return len(val)
        return None

    @staticmethod
    def _find_word_forward(val: str, pos: int) -> int | None:
        """Find the start of the next word from pos."""
        if pos >= len(val):
            return None
        # Skip current word chars
        i = pos
        while i < len(val) and val[i].isalnum():
            i += 1
        # Skip whitespace
        while i < len(val) and val[i].isspace():
            i += 1
        return i if i < len(val) else None

    @staticmethod
    def _find_word_backward(val: str, pos: int) -> int | None:
        """Find the start of the previous word from pos."""
        if pos <= 0:
            return None
        i = pos - 1
        # Skip whitespace backwards
        while i >= 0 and val[i].isspace():
            i -= 1
        # Skip word chars backwards
        while i > 0 and val[i - 1].isalnum():
            i -= 1
        return i if i >= 0 else None

    @staticmethod
    def _find_word_end(val: str, pos: int) -> int | None:
        """Find the end of the current/next word from pos."""
        if pos >= len(val) - 1:
            return None
        i = pos + 1
        # Skip whitespace
        while i < len(val) and val[i].isspace():
            i += 1
        # Skip word chars
        while i < len(val) - 1 and val[i + 1].isalnum():
            i += 1
        return i if i < len(val) else None

    def on_input_changed(self, event: Input.Changed) -> None:
        """Reset history index when user types manually."""
        if not self._editing_from_history:
            self._history_idx = -1
        self._editing_from_history = False

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key: in insert mode submit, in normal mode ignore."""
        event.stop()
        if self.vim_enabled and self.vim_mode == "normal":
            return
        value = event.value.strip()
        if value:
            images = _extract_images(value)
            self.post_message(ChatSubmit(value, images=images))
        self.clear()
        self._history_idx = -1
        # Return to normal mode after submit if vim is enabled
        if self.vim_enabled:
            self.vim_mode = "normal"


def _extract_images(text: str) -> list[dict]:
    """Detect image file references in text and return content blocks.

    Scans for @path or bare paths with image extensions. Each found image
    is read, base64-encoded, and returned as a dict suitable for inclusion
    in multimodal API content blocks.

    Returns a list of dicts with keys: type, media_type, data, path.
    """
    images: list[dict] = []
    for match in _IMAGE_PATH_RE.finditer(text):
        raw_path = match.group(1)
        if not raw_path:
            continue
        expanded = Path(raw_path).expanduser()
        if not expanded.is_file():
            continue
        ext = expanded.suffix.lower()
        if ext not in _IMAGE_EXTENSIONS:
            continue
        media_type, _ = mimetypes.guess_type(str(expanded))
        if not media_type:
            media_type = f"image/{ext.lstrip('.')}"
        try:
            data = expanded.read_bytes()
            images.append({
                "type": "image",
                "media_type": media_type,
                "data": base64.b64encode(data).decode("ascii"),
                "path": str(expanded),
            })
        except (OSError, PermissionError):
            continue
    return images
