"""koboi/modes.py -- Agent interaction modes (Chat/Plan/Act/Auto).

Centralizes mode types, per-mode configuration, and runtime mode switching.
Used by the hook system, TUI, and agent loop to enforce mode-aware behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AgentMode(Enum):
    """Interaction mode for the agent."""

    CHAT = "chat"  # Read-only exploration, no file modifications
    PLAN = "plan"  # Analyze codebase, produce numbered step plan
    ACT = "act"  # Execute with per-action permission prompts
    AUTO = "auto"  # Execute with graduated trust (learns from approvals)
    YOLO = "yolo"  # Bypass rate limit, approval, mode blocks; hardcoded safety only


@dataclass(frozen=True)
class ModeConfig:
    """Per-mode behavior configuration."""

    allow_file_write: bool
    allow_shell: bool
    require_plan_approval: bool
    permission_level: str  # "always_ask", "session_allow", "graduated"
    system_prompt_suffix: str


_MODE_CONFIGS: dict[AgentMode, ModeConfig] = {
    AgentMode.CHAT: ModeConfig(
        allow_file_write=False,
        allow_shell=False,
        require_plan_approval=False,
        permission_level="always_ask",
        system_prompt_suffix=(
            "You are in CHAT mode. You can read files and analyze code, "
            "but you MUST NOT modify any files or execute shell commands. "
            "Provide analysis, explanations, and suggestions only."
        ),
    ),
    AgentMode.PLAN: ModeConfig(
        allow_file_write=False,
        allow_shell=False,
        require_plan_approval=True,
        permission_level="always_ask",
        system_prompt_suffix=(
            "You are in PLAN mode. Analyze the user's request and the codebase, "
            "then produce a numbered step plan. Each step should be specific and "
            "actionable. Do NOT execute any file modifications or state-changing "
            "tool calls. Only use read-only tools (read, search, grep). "
            "Format your plan as a numbered list with clear descriptions."
        ),
    ),
    AgentMode.ACT: ModeConfig(
        allow_file_write=True,
        allow_shell=True,
        require_plan_approval=False,
        permission_level="always_ask",
        system_prompt_suffix=(
            "You are in ACT mode. Execute the user's request step by step. "
            "You may modify files and run shell commands, but each destructive "
            "action will require explicit user approval."
        ),
    ),
    AgentMode.AUTO: ModeConfig(
        allow_file_write=True,
        allow_shell=True,
        require_plan_approval=False,
        permission_level="graduated",
        system_prompt_suffix=(
            "You are in AUTO mode. Execute the user's request efficiently. "
            "The system uses graduated trust — actions you've been approved for "
            "before will be auto-approved. New or risky actions will still prompt."
        ),
    ),
    AgentMode.YOLO: ModeConfig(
        allow_file_write=True,
        allow_shell=True,
        require_plan_approval=False,
        permission_level="yolo",
        system_prompt_suffix=(
            "You are in YOLO mode. All tools are available without per-action "
            "confirmation. Rate limiting and approval prompts are disabled. "
            "Hardcoded safety checks (sensitive paths, dangerous commands) "
            "are still enforced. Use with caution."
        ),
    ),
}

_MODE_CYCLE = [AgentMode.CHAT, AgentMode.PLAN, AgentMode.ACT, AgentMode.AUTO, AgentMode.YOLO]


class ModeManager:
    """Manages the current agent mode and mode switching."""

    def __init__(self, initial_mode: AgentMode = AgentMode.CHAT):
        self._mode = initial_mode
        self._listeners: list = []

    @property
    def current_mode(self) -> AgentMode:
        return self._mode

    @property
    def config(self) -> ModeConfig:
        return _MODE_CONFIGS[self._mode]

    def switch_mode(self, mode: AgentMode) -> None:
        """Switch to a specific mode."""
        if mode == self._mode:
            return
        old = self._mode
        self._mode = mode
        for listener in self._listeners:
            listener(old, mode)

    def cycle_mode(self) -> AgentMode:
        """Cycle to the next mode: CHAT -> PLAN -> ACT -> AUTO -> YOLO -> CHAT."""
        idx = _MODE_CYCLE.index(self._mode)
        next_mode = _MODE_CYCLE[(idx + 1) % len(_MODE_CYCLE)]
        self.switch_mode(next_mode)
        return next_mode

    def on_mode_change(self, listener) -> None:
        """Register a callback: listener(old_mode, new_mode)."""
        self._listeners.append(listener)

    @staticmethod
    def get_config(mode: AgentMode) -> ModeConfig:
        """Get the config for a specific mode."""
        return _MODE_CONFIGS[mode]

    @staticmethod
    def from_string(value: str) -> AgentMode:
        """Parse a mode string (case-insensitive)."""
        try:
            return AgentMode(value.lower())
        except ValueError:
            valid = [m.value for m in AgentMode]
            raise ValueError(f"Unknown mode '{value}'. Valid modes: {valid}") from None
