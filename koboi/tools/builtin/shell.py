"""koboi/tools/builtin/shell -- Shell command execution."""

from __future__ import annotations

import functools
import logging
import os
import re
import subprocess

from koboi.tools.registry import tool, truncate_text
from koboi.types import RiskLevel
from koboi.harness.policy import COMMAND_DENY_PATTERNS, SENSITIVE_PATHS

MAX_OUTPUT = 10000
TIMEOUT = 30
_logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _get_npm_root() -> str:
    try:
        result = subprocess.run(
            ["npm", "root", "-g"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _build_env() -> dict:
    """Ensure global npm modules are in NODE_PATH."""
    env = os.environ.copy()
    npm_root = _get_npm_root()
    if npm_root and os.path.isdir(npm_root):
        existing = env.get("NODE_PATH", "")
        if npm_root not in existing:
            env["NODE_PATH"] = f"{npm_root}:{existing}" if existing else npm_root
    return env


def _check_command_blocked(command: str) -> str | None:
    """Quick inline safety check. Returns error message if blocked, None if OK."""
    cmd_lower = command.lower()
    for path in SENSITIVE_PATHS:
        if path.lower() in cmd_lower:
            return f"Blocked: command references sensitive path ({path})"
    for pattern in COMMAND_DENY_PATTERNS:
        match = pattern.search(cmd_lower)
        if match:
            return f"Blocked: command matches deny pattern ({match.group()[:50]})"
    return None


@tool(
    name="run_shell",
    group="system",
    description="Run shell command and return output",
    risk_level=RiskLevel.MODERATE,
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to run",
            },
            "cwd": {
                "type": "string",
                "description": "Working directory (optional, default: current directory)",
            },
        },
        "required": ["command"],
    },
)
def run_shell(command: str, cwd: str = "", _tool_config: dict | None = None) -> str:
    cfg = _tool_config or {}
    max_output = cfg.get("max_output", MAX_OUTPUT)
    timeout = cfg.get("timeout", TIMEOUT)
    # shim: macOS doesn't have `python` binary, only `python3`
    substituted = re.sub(r"\bpython\b(?!3)", "python3", command)
    if substituted != command:
        _logger.debug("Shell shim: substituted 'python' with 'python3' in command: %s", command)
    command = substituted
    blocked_reason = _check_command_blocked(command)
    if blocked_reason:
        return f"Error: {blocked_reason}"
    try:
        # shell=True is required for pipe/redirect/&& chaining in agent commands.
        # The policy engine provides deny-list filtering for dangerous patterns.
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or None,
            env=_build_env(),
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            if output:
                output += "\n"
            output += result.stderr
        if not output:
            output = "(no output)"
        if result.returncode != 0:
            output = f"[exit code: {result.returncode}]\n{output}"
        return truncate_text(output, max_output)
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    except FileNotFoundError as e:
        if cwd and not os.path.isdir(cwd):
            return f"Error: working directory '{cwd}' not found"
        return f"Error: command not found — {e}"
    except Exception as e:
        return f"Error: {e}"
