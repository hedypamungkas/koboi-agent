"""koboi/tools/state -- per-session mutable tool state.

Decoupled from the module-global ``_read_paths`` (filesystem) so concurrent
sessions in a long-running server don't share read-before-write tracking.
Injected per-agent via ``ToolRegistry.set_dep("tool_state", ToolState())``.
"""

from __future__ import annotations


class ToolState:
    """Per-agent (per-session) mutable state injected via ``ToolRegistry._deps``."""

    def __init__(self) -> None:
        # M6: paths read this session (advisory read-before-write tracking).
        self.read_paths: set[str] = set()
