"""koboi/subagent.py -- SubAgentManager for agent-driven subagent spawning.

Allows an agent to dynamically spawn child agents to handle subtasks in parallel.
Child agents inherit the parent's tools and receive a summary of the parent's conversation.

Lifecycle features:
- Per-task timeout (configurable)
- Running task tracking for cancellation
- Explicit cancel_task / cancel_all methods
- Resource cleanup after each task completes
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from koboi.client import Client
    from koboi.hooks.chain import HookChain
    from koboi.logger import AgentLogger
    from koboi.tools.registry import ToolRegistry


@dataclass
class SubagentTask:
    """A single task to delegate to a child agent."""
    task: str
    label: str = ""


@dataclass
class SubagentResult:
    """Result from a single subagent execution."""
    label: str
    task: str
    answer: str
    elapsed_seconds: float
    iterations_used: int
    success: bool
    error: str | None = None


def _build_conversation_summary(messages: list[dict], max_chars: int = 2000) -> str:
    """Build a brief summary of the conversation for subagent context."""
    parts: list[str] = []
    total = 0
    # Walk messages in reverse (most recent first), skip system messages
    for msg in reversed(messages):
        role = msg.get("role", "")
        if role == "system":
            continue
        content = msg.get("content", "") or ""
        if not content:
            continue
        entry = f"[{role}]: {content[:300]}"
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)
    parts.reverse()
    return "\n".join(parts) if parts else "(no prior context)"


class SubAgentManager:
    """Manages spawning and running child agents for task delegation.

    Supports:
    - Per-task timeout (asyncio.wait_for)
    - Running task tracking via _running_tasks dict
    - Cancel specific or all running subagents
    - Resource cleanup after each task
    """

    def __init__(
        self,
        client: Client,
        tools: ToolRegistry,
        hook_chain: HookChain,
        logger: AgentLogger | None = None,
        max_iterations: int = 5,
        timeout: float = 60.0,
    ):
        self.client = client
        self.tools = tools
        self.hook_chain = hook_chain
        self.logger = logger
        self.max_iterations = max_iterations
        self.timeout = timeout
        self._running_tasks: dict[str, asyncio.Task] = {}

    def _make_child_logger(self, label: str) -> AgentLogger | None:
        if not self.logger:
            return None
        from koboi.logger import AgentLogger
        return AgentLogger(
            log_dir=self.logger.log_dir,
            session_id=f"{self.logger.session_id}_sub_{label}",
        )

    def _build_child_tools(self) -> ToolRegistry:
        """Build a filtered copy of tools that excludes delegate_tasks (prevents recursion)."""
        from koboi.tools.registry import ToolRegistry
        child_tools = ToolRegistry()
        for name, defn in self.tools._tools.items():
            if name == "delegate_tasks":
                continue
            handler = self.tools._handlers.get(name)
            if handler is not None:
                child_tools.register(
                    name=name,
                    description=defn.description,
                    parameters=defn.parameters,
                    fn=handler,
                    risk_level=defn.risk_level,
                    timeout=defn.timeout,
                )
        return child_tools

    async def run_tasks(
        self,
        tasks: list[SubagentTask],
        parent_messages: list[dict] | None = None,
    ) -> list[SubagentResult]:
        """Run multiple subagent tasks in parallel with tracking."""
        summary = ""
        if parent_messages:
            summary = _build_conversation_summary(parent_messages)

        # Create tracked tasks
        async_tasks: list[asyncio.Task] = []
        for i, task in enumerate(tasks):
            label = task.label or f"task_{i}"
            coro = self._run_single(task, summary, index=i, total=len(tasks))
            atask = asyncio.create_task(coro)
            self._running_tasks[label] = atask
            async_tasks.append(atask)

        try:
            results = await asyncio.gather(*async_tasks, return_exceptions=True)
        finally:
            # Clean up tracking dict
            for task in tasks:
                label = task.label or f"task_{tasks.index(task)}"
                self._running_tasks.pop(label, None)

        final: list[SubagentResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final.append(SubagentResult(
                    label=tasks[i].label or f"task_{i}",
                    task=tasks[i].task,
                    answer="",
                    elapsed_seconds=0,
                    iterations_used=0,
                    success=False,
                    error=str(result),
                ))
            else:
                final.append(result)
        return final

    async def _run_single(
        self,
        task: SubagentTask,
        summary: str,
        index: int = 0,
        total: int = 1,
    ) -> SubagentResult:
        """Run a single subagent task with timeout and cleanup."""
        from koboi.loop import AgentCore
        from koboi.hooks.chain import HookContext, HookEvent

        label = task.label or f"task_{index}"
        child_logger = self._make_child_logger(label)

        # Emit dispatch event through parent's hook chain
        dispatch_ctx = HookContext(
            event=HookEvent.AGENT_DISPATCHED,
            metadata={
                "subagent_label": label,
                "subagent_task": task.task,
                "subagent_index": index,
                "subagent_total": total,
            },
        )
        await self.hook_chain.emit(dispatch_ctx)

        # Build child agent with parent's tools (excluding delegate_tasks to prevent recursion)
        system_prompt = (
            "You are a subagent handling a specific task. "
            "Complete the task thoroughly and provide a clear answer.\n\n"
        )
        if summary:
            system_prompt += f"Conversation context:\n{summary}\n\n"
        system_prompt += f"Your task: {task.task}"

        child_tools = self._build_child_tools()

        # Build a child-specific hook chain with its own DoomLoopHook
        # so parallel children don't interleave their tool-call histories.
        from koboi.hooks.chain import HookChain
        from koboi.hooks.doom_loop_hook import DoomLoopHook
        child_hooks = HookChain()
        for h in self.hook_chain._hooks:
            if isinstance(h, DoomLoopHook):
                continue  # skip parent's doom loop hook
            child_hooks.add(h)
        # Add a child-specific DoomLoopHook with lower thresholds
        try:
            from koboi.harness.doom_loop import DoomLoopConfig
            child_doom = DoomLoopHook(
                config=DoomLoopConfig(
                    consecutive_identical_threshold=3,
                    error_retry_threshold=2,
                ),
            )
            child_hooks.add(child_doom)
        except ImportError:
            pass

        child = AgentCore(
            client=self.client,
            tools=child_tools,
            max_iterations=self.max_iterations,
            logger=child_logger,
            system_prompt=system_prompt,
            hook_chain=child_hooks,
        )

        start = time.monotonic()
        try:
            # Wrap in timeout
            result = await asyncio.wait_for(
                child.run(task.task),
                timeout=self.timeout,
            )
            elapsed = time.monotonic() - start

            # Emit completion event
            complete_ctx = HookContext(
                event=HookEvent.AGENT_COMPLETED,
                metadata={
                    "subagent_label": label,
                    "subagent_task": task.task,
                    "subagent_index": index,
                    "subagent_total": total,
                    "subagent_elapsed": elapsed,
                    "subagent_success": result.success,
                },
            )
            await self.hook_chain.emit(complete_ctx)

            return SubagentResult(
                label=label,
                task=task.task,
                answer=result.content,
                elapsed_seconds=elapsed,
                iterations_used=result.iterations_used,
                success=result.success,
            )
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start
            error_msg = f"Subagent timed out after {self.timeout}s"

            # Emit timeout event
            fail_ctx = HookContext(
                event=HookEvent.AGENT_COMPLETED,
                metadata={
                    "subagent_label": label,
                    "subagent_task": task.task,
                    "subagent_index": index,
                    "subagent_total": total,
                    "subagent_elapsed": elapsed,
                    "subagent_success": False,
                    "subagent_error": error_msg,
                },
            )
            await self.hook_chain.emit(fail_ctx)

            return SubagentResult(
                label=label,
                task=task.task,
                answer="",
                elapsed_seconds=elapsed,
                iterations_used=0,
                success=False,
                error=error_msg,
            )
        except asyncio.CancelledError:
            elapsed = time.monotonic() - start
            error_msg = "Subagent cancelled"

            # Emit cancel event
            cancel_ctx = HookContext(
                event=HookEvent.AGENT_COMPLETED,
                metadata={
                    "subagent_label": label,
                    "subagent_task": task.task,
                    "subagent_index": index,
                    "subagent_total": total,
                    "subagent_elapsed": elapsed,
                    "subagent_success": False,
                    "subagent_error": error_msg,
                },
            )
            await self.hook_chain.emit(cancel_ctx)

            return SubagentResult(
                label=label,
                task=task.task,
                answer="",
                elapsed_seconds=elapsed,
                iterations_used=0,
                success=False,
                error=error_msg,
            )
        except Exception as e:
            elapsed = time.monotonic() - start

            # Emit failure event
            fail_ctx = HookContext(
                event=HookEvent.AGENT_COMPLETED,
                metadata={
                    "subagent_label": label,
                    "subagent_task": task.task,
                    "subagent_index": index,
                    "subagent_total": total,
                    "subagent_elapsed": elapsed,
                    "subagent_success": False,
                    "subagent_error": str(e),
                },
            )
            await self.hook_chain.emit(fail_ctx)

            return SubagentResult(
                label=label,
                task=task.task,
                answer="",
                elapsed_seconds=elapsed,
                iterations_used=0,
                success=False,
                error=str(e),
            )
        finally:
            # Resource cleanup
            child.memory.clear()
            if child_logger and hasattr(child_logger, "close"):
                try:
                    child_logger.close()
                except Exception:
                    pass

    # -- Cancel / status methods -----------------------------------------------

    def cancel_task(self, label: str) -> bool:
        """Cancel a specific running subagent by label. Returns True if found."""
        task = self._running_tasks.get(label)
        if task and not task.done():
            task.cancel()
            return True
        return False

    async def cancel_all(self) -> int:
        """Cancel all running subagents. Returns count cancelled."""
        count = 0
        for label, task in list(self._running_tasks.items()):
            if not task.done():
                task.cancel()
                count += 1
        self._running_tasks.clear()
        return count

    def list_running(self) -> list[str]:
        """Return labels of currently running subagents."""
        return [label for label, task in self._running_tasks.items() if not task.done()]
