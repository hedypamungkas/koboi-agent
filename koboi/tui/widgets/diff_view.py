"""DiffViewWidget -- unified and side-by-side diff renderers."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static


def is_diff_content(text: str) -> bool:
    """Heuristic: detect whether a string contains unified diff content."""
    if text.startswith("diff --git"):
        return True
    if "@@" in text and ("+++" in text or "---" in text):
        return True
    lines = text.strip().split("\n")
    if len(lines) < 3:
        return False
    diff_lines = sum(1 for line in lines if line.startswith("+") or line.startswith("-"))
    return diff_lines / len(lines) > 0.3


def count_changes(text: str) -> tuple[int, int]:
    """Return (additions, deletions) count from diff text."""
    additions = 0
    deletions = 0
    for line in text.split("\n"):
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            additions += 1
        elif line.startswith("-"):
            deletions += 1
    return additions, deletions


def _parse_diff_lines(text: str) -> list[tuple[str, str]]:
    """Parse diff text into (line, style) pairs."""
    result = []
    for line in text.split("\n"):
        if line.startswith("diff --git") or line.startswith("index "):
            result.append((line, "bold yellow"))
        elif line.startswith("@@"):
            result.append((line, "bold cyan"))
        elif line.startswith("+++") or line.startswith("---"):
            result.append((line, "bold"))
        elif line.startswith("+"):
            result.append((line, "green"))
        elif line.startswith("-"):
            result.append((line, "red"))
        else:
            result.append((line, ""))
    return result


class DiffViewWidget(Static):
    """Renders unified diff content with per-line syntax highlighting."""

    DEFAULT_CSS = """
    DiffViewWidget {
        width: 100%;
        max-height: 20;
        overflow-y: auto;
        background: $surface;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    """

    def __init__(self, diff_text: str, **kwargs) -> None:
        rich_text = self._build_rich_text(diff_text)
        super().__init__(rich_text, **kwargs)

    @classmethod
    def _build_rich_text(cls, diff_text: str) -> Text:
        """Build a Rich Text object from parsed diff lines."""
        parsed = _parse_diff_lines(diff_text)
        text = Text()
        for i, (line, style) in enumerate(parsed):
            if i > 0:
                text.append("\n")
            text.append(line, style=style)
        return text

    @staticmethod
    def _parse_diff(text: str) -> list[tuple[str, str]]:
        """Parse diff text into (line, style) pairs. Backward-compatible alias."""
        return _parse_diff_lines(text)


def _pair_diff_lines(text: str) -> list[tuple[str, str, str, str]]:
    """Parse unified diff into (left_text, left_style, right_text, right_style) rows.

    Context lines appear on both sides. Deletions on left, additions on right.
    Adjacent delete/add blocks are paired as changes.
    """
    lines = text.split("\n")
    pairs: list[tuple[str, str, str, str]] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Skip diff headers -- show on both sides
        if line.startswith("diff --git") or line.startswith("index "):
            pairs.append((line, "bold yellow", line, "bold yellow"))
            i += 1
            continue
        if line.startswith("@@"):
            pairs.append((line, "bold cyan", line, "bold cyan"))
            i += 1
            continue
        if line.startswith("+++") or line.startswith("---"):
            pairs.append((line, "bold", line, "bold"))
            i += 1
            continue

        # Context line -- same on both sides
        if not line.startswith("+") and not line.startswith("-"):
            pairs.append((line, "", line, ""))
            i += 1
            continue

        # Collect consecutive deletion and addition blocks
        dels: list[str] = []
        adds: list[str] = []
        while i < len(lines) and (lines[i].startswith("-") or lines[i].startswith("+")):
            if lines[i].startswith("-"):
                dels.append(lines[i][1:])  # strip the -/+ prefix
            else:
                adds.append(lines[i][1:])
            i += 1

        # Pair up deletions and additions as side-by-side changes
        max_len = max(len(dels), len(adds))
        for j in range(max_len):
            left = dels[j] if j < len(dels) else ""
            right = adds[j] if j < len(adds) else ""
            left_style = "red" if left else ""
            right_style = "green" if right else ""
            pairs.append((left, left_style, right, right_style))

    return pairs


class SideBySideDiffWidget(Static):
    """Renders diff content as two-column side-by-side view for wide terminals."""

    DEFAULT_CSS = """
    SideBySideDiffWidget {
        width: 100%;
        max-height: 20;
        overflow-y: auto;
        background: $surface;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    """

    def __init__(self, diff_text: str, **kwargs) -> None:
        rich_text = self._build_rich_text(diff_text)
        super().__init__(rich_text, **kwargs)

    @classmethod
    def _build_rich_text(cls, diff_text: str) -> Text:
        """Build a Rich Text object with two-column layout."""
        pairs = _pair_diff_lines(diff_text)
        # Calculate column width (half of typical wide terminal minus divider)
        col_width = 58  # ~120 cols / 2 - 2 for divider

        text = Text()
        for idx, (left, ls, right, rs) in enumerate(pairs):
            if idx > 0:
                text.append("\n")

            # Left column (old)
            left_display = left[:col_width].ljust(col_width)
            text.append(left_display, style=ls)

            # Divider
            text.append(" | ", style="dim")

            # Right column (new)
            right_display = right[:col_width]
            text.append(right_display, style=rs)

        return text
