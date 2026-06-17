"""koboi/hooks/rich_task_hook.py -- Rich console output for task events.

Prints task creation/updates to a Rich Console, useful for
non-TUI contexts (examples, scripts, CLI).
"""
from __future__ import annotations

from typing import Any

from koboi.hooks.chain import Hook, HookContext, HookEvent

_TASK_TOOLS = {"task_create", "task_list", "task_get", "task_update"}


class RichTaskHook(Hook):
    """Prints task lifecycle events to a Rich Console."""

    def __init__(self, console: Any = None) -> None:
        self._console = console

    def handles(self) -> list[HookEvent]:
        return [HookEvent.POST_TOOL_USE]

    async def execute(self, ctx: HookContext) -> HookContext:
        if ctx.tool_name not in _TASK_TOOLS:
            return ctx
        if self._console is None:
            return ctx

        result = ctx.tool_result or ""

        if ctx.tool_name == "task_create":
            if "blocked by" in result:
                self._console.print(
                    f"  [bold cyan]+[/bold cyan] Task created (blocked): [bold]{result}[/bold]"
                )
            else:
                self._console.print(
                    f"  [bold cyan]+[/bold cyan] Task created: [bold]{result}[/bold]"
                )
        elif ctx.tool_name == "task_update":
            if "completed" in result:
                self._console.print(
                    f"  [bold green]✓[/bold green] Task completed: {result}"
                )
            elif "in_progress" in result:
                self._console.print(
                    f"  [bold yellow]►[/bold yellow] Task started: {result}"
                )
            elif "Cannot start" in result:
                self._console.print(
                    f"  [bold red]⊘[/bold red] Task blocked: {result}"
                )
            else:
                self._console.print(
                    f"  [bold blue]↻[/bold blue] Task updated: {result}"
                )
        elif ctx.tool_name == "task_add_dependency":
            self._console.print(
                f"  [bold magenta]⤳[/bold magenta] Dependency added: {result}"
            )

        return ctx
