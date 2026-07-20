"""koboi/tools/builtin/typecheck -- Run a Python type/lint checker on a path.

A read-only diagnostic wrapper around ``ruff``/``mypy``/``pyright`` chosen from a
FIXED allowlist -- the tool NEVER runs an arbitrary user command, so (unlike
``run_shell``) there is no shell-injection surface; only the validated ``path``
is interpolated, and that is ``shlex.quote``-d. On a non-zero exit the output is
prefixed ``[exit code: N]`` (the same authoritative failure token ``run_shell``
emits) so the Wave 2.3 exit-code signal still works as a fallback. When the
self-healing ``TypecheckHook`` is wired (``self_healing.enabled``), that hook
refines ``error_kind`` from ``command_failed`` to ``typecheck_failed`` and
attaches structured ``{file,line,severity,message}`` diagnostics for the
reflection loop -- but this tool is fully functional without it.
"""

from __future__ import annotations

import logging
import shlex
import subprocess

from koboi.harness.env import build_safe_env
from koboi.tools.builtin.filesystem import _validate_path
from koboi.tools.builtin.shell import _format_result as _render_shell
from koboi.tools.registry import tool, truncate_text
from koboi.types import RiskLevel

_logger = logging.getLogger(__name__)

# Fixed allowlist -- never a user-supplied command. Adding a checker here is the
# only way to extend what run_typecheck will execute.
_CHECKERS: tuple[str, ...] = ("ruff", "mypy", "pyright")

# Checker -> argv template. The single validated, shlex-quoted ``path`` fills
# ``{paths}``. Concise/junit-free human output keeps the tool result small and
# is what TypecheckHook parses.
_DEFAULT_CMD: dict[str, str] = {
    "ruff": "ruff check --output-format=concise {paths}",
    "mypy": "mypy {paths}",
    "pyright": "pyright {paths}",
}

MAX_OUTPUT = 10000
TIMEOUT = 60  # typecheckers can be markedly slower than a shell one-liner


@tool(
    name="run_typecheck",
    group="file",
    deps=["sandbox"],
    description=(
        "Run a Python type/lint checker (ruff, mypy, or pyright) on a file or "
        "directory and return its output. Read-only diagnostic -- safe in any "
        "mode. On failure the output is prefixed `[exit code: N]` (same "
        "convention as run_shell). Defaults to ruff; override per-call or via "
        "tools.overrides.run_typecheck.checker. After editing code, run this to "
        "get precise file:line diagnostics before re-running tests."
    ),
    risk_level=RiskLevel.SAFE,
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File or directory to check",
            },
            "checker": {
                "type": "string",
                "enum": list(_CHECKERS),
                "description": "Checker to run (default: ruff, or tools.overrides.run_typecheck.checker)",
            },
        },
        "required": ["path"],
    },
)
def run_typecheck(
    path: str,
    checker: str | None = None,
    _tool_config: dict | None = None,
    _deps: dict | None = None,
) -> str:
    cfg = _tool_config or {}
    sandbox = (_deps or {}).get("sandbox")
    checker = (checker or cfg.get("checker") or "ruff").strip().lower()
    if checker not in _CHECKERS:
        return f"Error: unknown checker {checker!r}; choose one of {', '.join(_CHECKERS)}"
    timeout = float(cfg.get("timeout", TIMEOUT))
    max_output = int(cfg.get("max_output", MAX_OUTPUT))
    try:
        path = _validate_path(path, sandbox=sandbox)
    except PermissionError:
        return f"Error: no access to '{path}'"

    command = _DEFAULT_CMD[checker].format(paths=shlex.quote(path))
    try:
        if sandbox is not None:
            result = sandbox.run(
                command,
                env=sandbox.build_env(cfg),
                timeout=timeout,
                shell=True,
            )
        else:
            # Legacy path (no sandbox wired) -- mirrors run_shell's pre-P0b behavior.
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=build_safe_env(cfg),
            )
    except subprocess.TimeoutExpired:
        return f"Error: {checker} timed out after {int(timeout)}s"
    except FileNotFoundError:
        return f"Error: {checker} is not installed -- install it (e.g. `pip install {checker}`) or pick another checker"
    except Exception as exc:
        return f"Error: {exc}"

    output = _render_shell(result, int(timeout))
    # A clean typecheck run prints nothing; surface that as a positive signal
    # rather than the generic "(no output)" shell fallback.
    if output == "(no output)":
        output = "No issues found."
    return truncate_text(output, max_output)
