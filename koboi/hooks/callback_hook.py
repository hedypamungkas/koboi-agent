"""koboi/hooks/callback_hook.py -- Convenience hook that wraps a plain callback."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from koboi.hooks.chain import Hook, HookContext, HookEvent


class CallbackHook(Hook):
    """Wraps a plain callable (sync or async) into a Hook.

    Usage:
        agent.add_hook(
            lambda ctx: print(ctx.event.value),
            events=[HookEvent.POST_OUTPUT],
        )
    """

    def __init__(
        self,
        callback: Callable[[HookContext], HookContext | Awaitable[HookContext]],
        events: list[HookEvent] | None = None,
    ):
        self._callback = callback
        self._events = events or list(HookEvent)

    def handles(self) -> list[HookEvent]:
        return list(self._events)

    async def execute(self, ctx: HookContext) -> HookContext:
        result = self._callback(ctx)
        if asyncio.iscoroutine(result):
            return await result
        return result  # type: ignore[return-value]  # callback typed sync|async; non-coroutine branch is a sync HookContext
