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
from koboi.harness.env import build_safe_env

MAX_OUTPUT = 10000
TIMEOUT = 30
_logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _get_npm_root() -> str:
    try:
        result = subprocess.run(  # nosec B607 - intentional PATH-based launch of a user tool/editor
            ["npm", "root", "-g"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _build_env(cfg: dict | None = None, sandbox=None) -> dict:
    """Sanitized env for subprocess.run, with npm NODE_PATH preserved.

    When a sandbox is wired, env hygiene flows through ``sandbox.build_env``
    (the restricted backend layers PATH/network stripping on top); otherwise we
    use ``build_safe_env`` directly. Either way NODE_PATH is prepended.
    """
    if sandbox is not None:
        env = sandbox.build_env(cfg)
    else:
        env = build_safe_env(cfg)
    npm_root = _get_npm_root()
    if npm_root and os.path.isdir(npm_root):
        existing = env.get("NODE_PATH", "")
        if npm_root not in existing:
            env["NODE_PATH"] = f"{npm_root}:{existing}" if existing else npm_root
    return env


def _format_result(result, timeout: int) -> str:
    """Render a subprocess result (SandboxResult or CompletedProcess)."""
    if getattr(result, "timed_out", False):
        return f"Error: command timed out after {timeout}s"
    stdout = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    returncode = getattr(result, "returncode", 0)
    output = ""
    if stdout:
        output += stdout
    if stderr:
        output = ("\n" + stderr) if output else stderr
    if not output:
        output = "(no output)"
    if returncode != 0:
        output = f"[exit code: {returncode}]\n{output}"
    return output


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
    deps=["sandbox"],
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
def run_shell(command: str, cwd: str = "", _tool_config: dict | None = None, _deps: dict | None = None) -> str:
    cfg = _tool_config or {}
    sandbox = (_deps or {}).get("sandbox")
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
        if sandbox is not None:
            # shell=True is required for pipe/redirect/&& chaining. The sandbox
            # applies cwd containment, env hygiene, network/rlimit policy, and
            # process-group kill on timeout.
            result = sandbox.run(
                command,
                cwd=cwd or None,
                env=_build_env(cfg, sandbox=sandbox),
                timeout=timeout,
                shell=True,
            )
            return truncate_text(_format_result(result, timeout), max_output)
        # Legacy path (no sandbox wired) -- preserves exact pre-P0b behavior.
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or None,
            env=_build_env(cfg),
        )
        return truncate_text(_format_result(result, timeout), max_output)
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    except PermissionError as e:
        # Restricted sandbox rejects cwd/paths outside the workdir.
        return f"Error: {e}"
    except FileNotFoundError as e:
        if cwd and not os.path.isdir(cwd):
            return f"Error: working directory '{cwd}' not found"
        return f"Error: command not found — {e}"
    except Exception as e:
        return f"Error: {e}"
