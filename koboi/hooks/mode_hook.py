"""koboi/hooks/mode_hook.py -- Mode-aware hook for agent lifecycle.

Enforces mode constraints:
- PRE_INPUT: Injects mode-specific system prompt suffix.
- PRE_TOOL_USE: Blocks state-modifying tools in CHAT/PLAN modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from koboi.hooks.chain import Hook, HookContext, HookEvent

if TYPE_CHECKING:
    from koboi.modes import ModeManager

# Tools that are always read-only (allowed in CHAT/PLAN)
_READ_ONLY_TOOLS = {
    "read",
    "search",
    "grep",
    "find",
    "list",
    "glob",
    "web_search",
    "web_fetch",
    "calculator",
    "delegate_tasks",
}


class ModeHook(Hook):
    """Hook that enforces interaction mode constraints.

    Subscribes to PRE_INPUT (to inject mode context) and PRE_TOOL_USE
    (to block disallowed tools based on the current mode).
    """

    def __init__(self, mode_manager: ModeManager):
        self._mode_manager = mode_manager

    def handles(self) -> list[HookEvent]:
        return [HookEvent.PRE_INPUT, HookEvent.PRE_TOOL_USE]

    async def execute(self, ctx: HookContext) -> HookContext:
        if ctx.event == HookEvent.PRE_INPUT:
            return self._on_pre_input(ctx)
        elif ctx.event == HookEvent.PRE_TOOL_USE:
            return self._on_pre_tool_use(ctx)
        return ctx

    def _on_pre_input(self, ctx: HookContext) -> HookContext:
        """Inject mode-specific system prompt suffix."""
        config = self._mode_manager.config
        if config.system_prompt_suffix:
            ctx.inject_message = config.system_prompt_suffix
        return ctx

    def _on_pre_tool_use(self, ctx: HookContext) -> HookContext:
        """Flag tools that are not allowed in the current mode.

        Sets metadata flags instead of aborting, so other hooks (callbacks)
        still fire. The agent loop enforces the actual blocking.
        """
        if not ctx.tool_name:
            return ctx

        mode = self._mode_manager.current_mode

        # In CHAT mode: only read-only tools allowed
        if mode.value == "chat":
            if not self._is_read_only(ctx.tool_name):
                ctx.metadata["mode_blocked"] = True
                ctx.metadata["mode_block_reason"] = (
                    f"CHAT mode: tool '{ctx.tool_name}' is not allowed. "
                    "Switch to ACT or AUTO mode to execute state-changing tools."
                )
                return ctx

        # In PLAN mode: only read-only tools allowed
        if mode.value == "plan":
            if not self._is_read_only(ctx.tool_name):
                ctx.metadata["mode_blocked"] = True
                ctx.metadata["mode_block_reason"] = (
                    f"PLAN mode: tool '{ctx.tool_name}' is not allowed. "
                    "Only read-only tools are permitted in PLAN mode."
                )
                return ctx

        # In ACT mode: allow all tools (permission dialog handles approval)
        # In AUTO mode: allow all tools (trust DB handles auto-approval)
        # In YOLO mode: allow all tools (pipeline skips rate limit and approval)
        return ctx

    @staticmethod
    def _is_read_only(tool_name: str) -> bool:
        """Check if a tool is read-only (safe for CHAT/PLAN modes)."""
        name_lower = tool_name.lower()
        # Exact match
        if name_lower in _READ_ONLY_TOOLS:
            return True
        # Prefix match for namespaced tools (e.g., "filesystem.read")
        for prefix in _READ_ONLY_TOOLS:
            if name_lower.startswith(prefix + "."):
                return True
        return False
