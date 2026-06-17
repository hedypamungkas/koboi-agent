"""koboi/hooks/chain.py -- Observer-pattern hook system for agent lifecycle.

Provides structured intervention points at every stage of the agent loop:
before/after tool execution, LLM calls, compaction, session boundaries,
input processing, and output generation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


class HookEvent(Enum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    PRE_INPUT = "pre_input"
    POST_OUTPUT = "post_output"
    PRE_COMPACT = "pre_compact"
    POST_COMPACT = "post_compact"
    PRE_LLM_CALL = "pre_llm_call"
    POST_LLM_CALL = "post_llm_call"
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    DOOM_LOOP_DETECTED = "doom_loop_detected"
    PRE_ROUTING = "pre_routing"
    POST_ROUTING = "post_routing"
    AGENT_DISPATCHED = "agent_dispatched"
    AGENT_COMPLETED = "agent_completed"


class HookOutcome(Enum):
    """Result of a single hook execution within a chain emit."""

    SUCCESS = "success"
    ERRORED = "errored"


@dataclass
class AgentInfo:
    """Narrow view of agent state exposed to hooks.

    Prevents bidirectional coupling between hooks and AgentCore.
    Add fields here only when a hook genuinely needs them.
    """

    model: str = ""
    agent_name: str = ""
    iteration: int = 0


class MetadataBag(dict):
    """Typed metadata container for HookContext.

    Supports both known typed fields (via properties) and arbitrary
    custom keys (via dict interface). Fully backward-compatible with
    code that reads/writes ctx.metadata["key"].
    """

    # -- Mode enforcement --

    @property
    def mode_blocked(self) -> bool:
        return self.get("mode_blocked", False)

    @mode_blocked.setter
    def mode_blocked(self, value: bool) -> None:
        self["mode_blocked"] = value

    @property
    def mode_block_reason(self) -> str:
        return self.get("mode_block_reason", "")

    # -- Policy engine --

    @property
    def policy_decision(self) -> dict | None:
        return self.get("policy_decision")

    @property
    def policy_needs_confirmation(self) -> bool:
        return self.get("policy_needs_confirmation", False)

    # -- Guardrails --

    @property
    def guardrail_blocked(self) -> bool:
        return self.get("guardrail_blocked", False)

    @property
    def input_guardrail_result(self) -> dict | None:
        return self.get("input_guardrail_result")

    @property
    def output_guardrail_result(self) -> dict | None:
        return self.get("output_guardrail_result")

    @property
    def output_warning(self) -> str:
        return self.get("output_warning", "")

    # -- Doom loop --

    @property
    def doom_loop_detected(self) -> bool:
        return self.get("doom_loop_detected", False)

    @property
    def doom_loop_info(self) -> dict | None:
        return self.get("doom_loop")

    # -- Subagent --

    @property
    def subagent_label(self) -> str | None:
        return self.get("subagent_label")

    @property
    def subagent_task(self) -> str | None:
        return self.get("subagent_task")

    @property
    def subagent_success(self) -> bool | None:
        return self.get("subagent_success")

    @property
    def subagent_error(self) -> str | None:
        return self.get("subagent_error")

    # -- RAG --

    @property
    def rag_strategy(self) -> str:
        return self.get("rag_strategy", "")

    @property
    def rag_augmentation(self) -> str:
        return self.get("rag_augmentation", "")

    # -- Skills --

    @property
    def skills_detected(self) -> list:
        return self.get("skills_detected", [])

    # -- Context management --

    @property
    def context_managed(self) -> bool:
        return self.get("context_managed", False)


@dataclass(init=False)
class HookContext:
    event: HookEvent
    agent: AgentInfo | None = None
    iteration: int = 0
    tool_name: str | None = None
    tool_arguments: str | None = None
    tool_result: str | None = None
    messages: list[dict] | None = None
    user_message: str | None = None
    llm_response: Any = None
    carryover: Any = None
    metadata: MetadataBag = field(default_factory=MetadataBag)
    abort: bool = False
    inject_messages: list[str] = field(default_factory=list)
    hook_outcomes: list[tuple[str, HookOutcome, str | None]] = field(default_factory=list)

    def __init__(
        self,
        event: HookEvent,
        *,
        agent: AgentInfo | None = None,
        iteration: int = 0,
        tool_name: str | None = None,
        tool_arguments: str | None = None,
        tool_result: str | None = None,
        messages: list[dict] | None = None,
        user_message: str | None = None,
        llm_response: Any = None,
        carryover: Any = None,
        metadata: dict | MetadataBag | None = None,
        abort: bool = False,
        inject_message: str | None = None,
        inject_messages: list[str] | None = None,
        hook_outcomes: list[tuple[str, HookOutcome, str | None]] | None = None,
    ):
        self.event = event
        self.agent = agent
        self.iteration = iteration
        self.tool_name = tool_name
        self.tool_arguments = tool_arguments
        self.tool_result = tool_result
        self.messages = messages
        self.user_message = user_message
        self.llm_response = llm_response
        self.carryover = carryover
        self.metadata = MetadataBag(metadata) if metadata is not None else MetadataBag()
        self.abort = abort
        self.inject_messages = list(inject_messages) if inject_messages else []
        if inject_message is not None:
            self.inject_messages.append(inject_message)
        self.hook_outcomes = list(hook_outcomes) if hook_outcomes else []

    @property
    def inject_message(self) -> str | None:
        """Backward-compatible: returns the last injected message, or None."""
        return self.inject_messages[-1] if self.inject_messages else None

    @inject_message.setter
    def inject_message(self, value: str | None) -> None:
        """Backward-compatible setter: appends to the message queue."""
        if value is not None:
            self.inject_messages.append(value)


class Hook(ABC):
    """Base class for lifecycle hooks.

    Priority conventions (lower = runs first):
        0-19:  Infrastructure (logging, telemetry)
        20-39: Security (guardrails, policy)
        40-59: Business logic (default)
        60-79: Post-processing (audit, carryover)
        80-100: Cleanup (notifications)
    """

    priority: int = 50

    @abstractmethod
    def handles(self) -> list[HookEvent]: ...

    @abstractmethod
    async def execute(self, ctx: HookContext) -> HookContext: ...


class HookChain:
    def __init__(self, hooks: list[Hook] | None = None):
        self._hooks: list[Hook] = hooks or []
        self._subscribers: dict[HookEvent, list[Hook]] = {}
        self._build_index()

    def add(self, hook: Hook) -> None:
        self._hooks.append(hook)
        self._build_index()

    async def emit(self, ctx: HookContext) -> HookContext:
        for hook in self._subscribers.get(ctx.event, []):
            hook_name = type(hook).__name__
            try:
                ctx = await hook.execute(ctx)
                ctx.hook_outcomes.append((hook_name, HookOutcome.SUCCESS, None))
            except Exception as exc:
                _logger.warning(
                    "Hook %s raised %s: %s -- setting abort",
                    hook_name,
                    type(exc).__name__,
                    exc,
                )
                ctx.hook_outcomes.append((hook_name, HookOutcome.ERRORED, str(exc)))
                ctx.abort = True
            if ctx.abort:
                break
        return ctx

    def list_hooks(self) -> list[dict]:
        return [
            {
                "hook": hook,
                "name": type(hook).__name__,
                "events": [e.value for e in hook.handles()],
            }
            for hook in self._hooks
        ]

    def find_hook(self, predicate: Callable[[Hook], bool]) -> Hook | None:
        for hook in self._hooks:
            if predicate(hook):
                return hook
        return None

    def _build_index(self) -> None:
        self._hooks.sort(key=lambda h: h.priority)
        self._subscribers = {}
        for hook in self._hooks:
            for event in hook.handles():
                self._subscribers.setdefault(event, []).append(hook)
