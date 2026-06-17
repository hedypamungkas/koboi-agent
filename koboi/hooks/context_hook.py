"""koboi/hooks/context_hook.py -- Hook for context management at POST_COMPACT.

Applies context window management strategies when context compaction occurs.
"""
from __future__ import annotations

from koboi.hooks.chain import Hook, HookContext, HookEvent


class ContextHook(Hook):
    """Hook that applies context management at POST_COMPACT.

    Manages the message history after compaction by ensuring key context
    is preserved and properly structured for the next LLM call.
    """

    def __init__(
        self,
        max_messages: int = 50,
        preserve_system: bool = True,
        preserve_recent: int = 5,
    ):
        self.max_messages = max_messages
        self.preserve_system = preserve_system
        self.preserve_recent = preserve_recent

    def handles(self) -> list[HookEvent]:
        return [HookEvent.POST_COMPACT]

    async def execute(self, ctx: HookContext) -> HookContext:
        if not ctx.messages:
            return ctx

        messages = ctx.messages
        managed: list[dict] = []

        # Always preserve system messages
        if self.preserve_system:
            for msg in messages:
                if msg.get("role") == "system":
                    managed.append(msg)

        # Preserve the most recent messages
        recent = messages[-self.preserve_recent:] if len(messages) > self.preserve_recent else messages
        for msg in recent:
            if msg not in managed:
                managed.append(msg)

        # If we still have room, include middle messages up to max_messages
        if len(managed) < self.max_messages:
            remaining_slots = self.max_messages - len(managed)
            middle = [
                msg for msg in messages
                if msg not in managed
            ]
            managed.extend(middle[:remaining_slots])

        # Record context management in metadata
        ctx.metadata["context_managed"] = True
        ctx.metadata["context_before"] = len(messages)
        ctx.metadata["context_after"] = len(managed)

        # Update messages if we trimmed
        if len(managed) < len(messages):
            ctx.messages = managed

        return ctx
