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

# Tools always treated as read-only (permitted in CHAT/PLAN). Single source of truth
# shared by ModeManager.is_tool_allowed and ModeHook so the pipeline's pre-approval
# mode gate and the hook's metadata flag can never disagree.
_READ_ONLY_TOOLS: set[str] = {
    "read",
    "search",
    "grep",
    "find",
    "list",
    "glob",
    "web_search",
    "web_fetch",
    "calculator",
    "delegate_tasks",
    # Exact names, NOT a "git" prefix: a future git_commit/git_push must not
    # inherit read-only status.
    "git_status",
    "git_log",
    "git_diff",
    "repo_map",
    "github_list_prs",
    "github_get_pr",
    "run_typecheck",  # read-only diagnostic (ruff/mypy/pyright on a validated path)
}


def is_read_only_tool(tool_name: str) -> bool:
    """Check if a tool is read-only (safe for CHAT/PLAN modes).

    Exact match, or prefix match for namespaced (``filesystem.read``) and
    snake_case builtin (``read_file``, ``grep_search``) tool names. Without the
    ``_`` separator no builtin ever matched, so CHAT/PLAN blocked even
    ``read_file`` -- contradicting the documented behavior.

    The ``prefix + "_"`` rule is intentionally permissive (the short bases
    ``read``/``list``/``grep``/``glob``/``find`` are what qualify the snake_case
    builtins), so a HYPOTHETICAL destructive tool whose name starts with a
    read-only base (``git_diff_apply``) would also match. No such tool is
    registered today, and ``tests/test_modes.py`` has a registry-snapshot test
    that fails at CI if any DESTRUCTIVE/MODERATE builtin ever collides -- that
    test is the durable guard, not this function.
    """
    name_lower = tool_name.lower()
    if name_lower in _READ_ONLY_TOOLS:
        return True
    for prefix in _READ_ONLY_TOOLS:
        if name_lower.startswith(prefix + ".") or name_lower.startswith(prefix + "_"):
            return True
    return False


def _mode_block_reason(mode: AgentMode, tool_name: str) -> str:
    """Human-readable reason for why ``tool_name`` is blocked in ``mode``."""
    if mode == AgentMode.CHAT:
        return (
            f"CHAT mode: tool '{tool_name}' is not allowed. Switch to ACT or AUTO mode to execute state-changing tools."
        )
    return f"PLAN mode: tool '{tool_name}' is not allowed. Only read-only tools are permitted in PLAN mode."


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

    def is_tool_allowed(self, tool_name: str) -> tuple[bool, str]:
        """Whether ``tool_name`` may run in the current mode.

        Returns ``(allowed, reason)`` -- ``reason`` is ``""`` when allowed. CHAT/PLAN
        permit only read-only tools; ACT/AUTO/YOLO permit all. This is the single
        source of truth consulted BOTH by the pipeline (pre-approval mode gate, so a
        blocked tool never prompts the user) and by ModeHook (metadata flag for audit).
        """
        if self._mode in (AgentMode.CHAT, AgentMode.PLAN) and not is_read_only_tool(tool_name):
            return False, _mode_block_reason(self._mode, tool_name)
        return True, ""

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
