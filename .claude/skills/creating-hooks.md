---
name: creating-hooks
description: Guide for creating new hooks in the koboi hook system
---

# Creating Hooks

## Pattern
1. Create a file in `koboi/hooks/` named `<name>_hook.py`
2. Import from `koboi.hooks.chain`: `Hook`, `HookContext`, `HookEvent`
3. Use `TYPE_CHECKING` guard for logger imports
4. Subclass `Hook`, implement `handles()` and `execute()`

## Template
```python
"""koboi/hooks/my_hook.py -- Short description."""
from __future__ import annotations

from typing import TYPE_CHECKING
from koboi.hooks.chain import Hook, HookContext, HookEvent

if TYPE_CHECKING:
    from koboi.logger import AgentLogger


class MyHook(Hook):
    def __init__(self, logger: AgentLogger | None = None):
        self.logger = logger

    def handles(self) -> list[HookEvent]:
        return [HookEvent.PRE_TOOL_USE, HookEvent.POST_TOOL_USE]

    async def execute(self, ctx: HookContext) -> HookContext:
        # Access: ctx.tool_name, ctx.tool_arguments, ctx.tool_result,
        #         ctx.iteration, ctx.messages, ctx.user_message
        # Set ctx.abort = True to halt execution
        # Set ctx.inject_message to inject text
        return ctx
```

## Registration
Add to `facade.py:_build_hooks()` with a config flag, or use `agent.add_hook()` programmatically.

## Existing hooks for reference
- `builtin.py`: LoggingHook (simplest), AuditHook (tool events)
- `callback_hook.py`: Wraps plain functions
- `doom_loop_hook.py`: Shows abort pattern
- `carryover_hook.py`: Shows event dispatch via dict mapping
