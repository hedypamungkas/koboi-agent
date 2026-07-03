"""koboi/hooks/read_before_write_reset_hook.py -- Reset read-before-write tracker.

Clears the filesystem tool's read-path set at SESSION_START (fresh session =
fresh tracking) and after a REAL context compaction (POST_COMPACT). This
prevents stale read-memory from suppressing advisory write/delete notes on
files whose prior reads were truncated away.

The loop emits POST_COMPACT on EVERY iteration (loop.py), so an unconditional
reset would wipe valid inter-iteration tracking. This hook therefore gates the
POST_COMPACT reset on ``ctx.metadata["compacted"]``, which AgentCore sets only
when ``ContextManager.manage()`` actually trimmed messages.
"""

from __future__ import annotations

from koboi.hooks.chain import Hook, HookContext, HookEvent


class ReadBeforeWriteResetHook(Hook):
    """Reset the read-before-write tracker at session start and on real compaction.

    Priority 44: sits in the persistence band alongside SkillPersistenceHook
    (45) and TaskPersistenceHook (46). The relative order among these three is
    incidental -- clearing _read_paths is independent of re-injecting skill
    bodies or task summaries, so no data dependency forces a specific sequence.
    """

    priority = 44

    def handles(self) -> list[HookEvent]:
        return [HookEvent.SESSION_START, HookEvent.POST_COMPACT]

    async def execute(self, ctx: HookContext) -> HookContext:
        if ctx.event is HookEvent.SESSION_START:
            # Fresh session: always start with a clean tracker.
            self._reset()
            return ctx

        # POST_COMPACT: only reset if a real compaction occurred. The loop
        # stamps this flag; absent/False means "no trim -- keep tracking".
        if ctx.metadata.get("compacted", False):
            self._reset()
        return ctx

    @staticmethod
    def _reset() -> None:
        # Imported lazily so the hook is constructible without filesystem tools
        # being registered (e.g. an agent with no file tools).
        from koboi.tools.builtin import filesystem

        filesystem.reset_read_before_write()
