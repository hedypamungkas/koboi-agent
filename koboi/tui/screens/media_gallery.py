"""media_gallery.py -- Modal screen listing generated media artifacts (W5c #2).

Surfaces the artifacts produced by the ``generate_image``/``generate_video``/``generate_music``/
``generate_speech`` tools during a session (path + modality + metadata). Inline pixel rendering is
deferred (needs a ``textual-image`` ext + terminal-graphics support); this screen shows paths +
metadata only, mirroring the f2/MCP-status pattern.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

_MEDIA_TOOLS = frozenset({"generate_image", "generate_video", "generate_music", "generate_speech"})
_LABEL_TO_MODALITY = {"Image": "image", "Video": "video", "Music": "music", "Speech": "speech"}


def parse_media_artifact(tool_name: str, result: str) -> dict | None:
    """Parse a ``generate_*`` tool result string into an artifact dict, or ``None`` if not media.

    The tool output shape is ``"<Label> saved: <path> (<content_type>, <dims>, $<cost>/<unit>, model=<m>)"``
    (see ``koboi.tools.builtin.media._format_result``). Non-saved results (errors/rejections) -> None.
    """
    if tool_name not in _MEDIA_TOOLS or not result or " saved: " not in result:
        return None
    label, rest = result.split(" saved: ", 1)
    modality = _LABEL_TO_MODALITY.get(label.strip())
    if modality is None:
        return None
    path = rest.split(" (", 1)[0].strip()
    meta = rest.split(" (", 1)[1].rstrip(")") if " (" in rest else ""
    return {"modality": modality, "path": path, "meta": meta, "tool": tool_name, "raw": result}


class _MediaArtifactRow(Static):
    """A single media artifact row."""

    DEFAULT_CSS = """
    _MediaArtifactRow {
        width: 100%;
        height: auto;
        padding: 0 2;
    }
    """

    def __init__(self, data: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = data

    def render(self) -> str:
        modality = self._data.get("modality", "?")
        path = self._data.get("path", "(no artifact)")
        meta = self._data.get("meta", "")
        line = f"  [cyan]{modality}[/cyan]  {path}"
        if meta:
            line += f" [dim]({meta})[/dim]"
        return line


class MediaGalleryScreen(ModalScreen[None]):
    """Modal overlay listing generated media artifacts (paths + metadata)."""

    CSS = """
    MediaGalleryScreen {
        background: rgba(0, 0, 0, 0.7);
    }
    #media-panel {
        width: 80;
        max-width: 90%;
        height: auto;
        max-height: 70%;
        background: $surface;
        border: tall $accent;
        padding: 1 2;
    }
    #media-panel .heading {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    #media-panel .hint {
        color: $text-muted;
        margin-top: 1;
        text-align: center;
    }
    #media-list {
        height: auto;
        max-height: 50;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
    ]

    def __init__(self, artifacts: list[dict], **kwargs) -> None:
        super().__init__(**kwargs)
        self._artifacts = artifacts

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="media-panel"):
            yield Static("Media Gallery", classes="heading")
            with VerticalScroll(id="media-list"):
                if not self._artifacts:
                    yield Static("  No media generated yet this session.", classes="hint")
                else:
                    for idx, art in enumerate(self._artifacts):
                        yield _MediaArtifactRow(art, id=f"media-row-{idx}")
            yield Static("Press Esc to close", classes="hint")
