"""koboi/hooks/mode_hook.py -- Mode-aware hook for agent lifecycle.

Enforces mode constraints:
- PRE_INPUT: Injects mode-specific system prompt suffix.
- PRE_TOOL_USE: Blocks state-modifying tools in CHAT/PLAN modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from koboi.hooks.chain import Hook, HookContext, HookEvent
from koboi.modes import is_read_only_tool

if TYPE_CHECKING:
    from koboi.modes import ModeManager


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

        Delegates to ``ModeManager.is_tool_allowed`` (single source of truth, shared
        with the pipeline's pre-approval mode gate). Sets metadata flags instead of
        aborting so other hooks still fire; the pipeline enforces the actual block
        before approval, so a blocked tool normally never reaches PRE_TOOL_USE. This
        hook remains as defense-in-depth + audit signal when it does.
        """
        if not ctx.tool_name:
            return ctx

        allowed, reason = self._mode_manager.is_tool_allowed(ctx.tool_name)
        if not allowed:
            ctx.metadata["mode_blocked"] = True
            ctx.metadata["mode_block_reason"] = reason
        return ctx

    @staticmethod
    def _is_read_only(tool_name: str) -> bool:
        """Check if a tool is read-only (safe for CHAT/PLAN modes).

        Thin delegator to ``koboi.modes.is_read_only_tool``; kept for backward
        compatibility (exercised directly by tests).
        """
        return is_read_only_tool(tool_name)
