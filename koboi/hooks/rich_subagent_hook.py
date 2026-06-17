"""koboi/hooks/rich_subagent_hook.py -- Rich console output for subagent events.

Prints subagent dispatch/completion to a Rich Console, useful for
non-TUI contexts (examples, scripts, CLI).
"""
from __future__ import annotations

from typing import Any

from koboi.hooks.chain import Hook, HookContext, HookEvent


class RichSubagentHook(Hook):
    """Prints subagent lifecycle events to a Rich Console."""

    def __init__(self, console: Any = None) -> None:
        self._console = console

    def handles(self) -> list[HookEvent]:
        return [HookEvent.AGENT_DISPATCHED, HookEvent.AGENT_COMPLETED]

    async def execute(self, ctx: HookContext) -> HookContext:
        meta = ctx.metadata
        if "subagent_label" not in meta:
            return ctx

        label = meta.get("subagent_label", "unknown")
        index = meta.get("subagent_index", 0)
        total = meta.get("subagent_total", 1)
        task = meta.get("subagent_task", "")

        if self._console is None:
            return ctx

        if ctx.event == HookEvent.AGENT_DISPATCHED:
            self._console.print(
                f"  [bold cyan]>[/bold cyan] Subagent [bold]{label}[/bold] "
                f"dispatched ({index + 1}/{total}): "
                f"[dim]{task[:80]}{'...' if len(task) > 80 else ''}[/dim]"
            )
        elif ctx.event == HookEvent.AGENT_COMPLETED:
            elapsed = meta.get("subagent_elapsed", 0.0)
            success = meta.get("subagent_success", True)
            error = meta.get("subagent_error")

            if success:
                self._console.print(
                    f"  [bold green]<[/bold green] Subagent [bold]{label}[/bold] "
                    f"completed in [cyan]{elapsed:.1f}s[/cyan]"
                )
            else:
                err_text = error or "unknown error"
                self._console.print(
                    f"  [bold red]<[/bold red] Subagent [bold]{label}[/bold] "
                    f"failed after [cyan]{elapsed:.1f}s[/cyan]: [red]{err_text}[/red]"
                )

        return ctx
