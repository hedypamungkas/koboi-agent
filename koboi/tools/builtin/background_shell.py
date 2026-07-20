"""koboi/tools/builtin/background_shell -- start/poll/kill long-running shell processes.

Opt-in via ``agent.background_shell.enabled`` (default off). Unlike ``run_shell``
(one bounded, approval-gated invocation), ``submit_background_shell``'s live
process runs OUTSIDE the tool-execution pipeline for its ENTIRE lifetime once
started -- only the start/check/kill calls are gated (approval, policy, audit).
``check_command_blocked`` only evaluates the *launch* command, not anything the
process does afterward. ``max_lifetime_seconds`` is the mitigating cap. See
``koboi/harness/background_shell.py`` for the manager + full risk discussion.
"""

from __future__ import annotations

from koboi.tools.registry import tool
from koboi.types import RiskLevel


def _manager_or_error(_deps: dict | None):
    manager = (_deps or {}).get("background_shell_manager")
    if manager is None:
        return None, "Error: background shell is not configured (set agent.background_shell.enabled: true)."
    return manager, None


@tool(
    name="submit_background_shell",
    group="system",
    description=(
        "Start a long-running shell command (e.g. a dev server or watch-mode test runner) in the "
        "background and return a job id immediately -- does not block waiting for it to finish. "
        "Use check_background_shell to poll status/output and kill_background_shell to stop it."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run in the background."},
            "cwd": {"type": "string", "description": "Working directory. Default: current directory."},
            "max_lifetime_seconds": {
                "type": "integer",
                "description": "Hard cap on how long the process may run before it is auto-killed. Default: 1800.",
            },
        },
        "required": ["command"],
    },
    # DESTRUCTIVE: same bar as run_shell -- arguably deserves MORE scrutiny since
    # one approval now buys an UNBOUNDED duration of unsupervised process activity
    # (run_shell's one approval buys one bounded action). max_lifetime_seconds is
    # the mitigating control.
    risk_level=RiskLevel.DESTRUCTIVE,
    idempotent=False,
    deps=["sandbox", "background_shell_manager"],
)
async def submit_background_shell(
    command: str,
    cwd: str = "",
    max_lifetime_seconds: int | None = None,
    _tool_config: dict | None = None,
    _deps: dict | None = None,
) -> str:
    manager, err = _manager_or_error(_deps)
    if err:
        return err
    sandbox = (_deps or {}).get("sandbox")
    cfg = _tool_config or {}
    try:
        resolved_cwd = sandbox.validate_path(cwd or ".") if sandbox is not None else (cwd or ".")
    except PermissionError as e:
        return f"Error: {e}"
    env = sandbox.build_env(cfg) if sandbox is not None else None
    try:
        job = await manager.start(command, cwd=resolved_cwd, env=env, max_lifetime_seconds=max_lifetime_seconds)
    except ValueError as e:
        return f"Error: {e}"
    return f"Background job started: id={job.job_id} pid={job.pid}"


@tool(
    name="check_background_shell",
    group="system",
    description="Check the status and recent output of a background shell job.",
    parameters={
        "type": "object",
        "properties": {
            "job_id": {"type": "string", "description": "Job id returned by submit_background_shell."},
            "tail_lines": {
                "type": "integer",
                "description": "Number of recent output lines to include. Default: 50.",
            },
        },
        "required": ["job_id"],
    },
    risk_level=RiskLevel.SAFE,
    idempotent=True,
    deps=["background_shell_manager"],
)
async def check_background_shell(job_id: str, tail_lines: int = 50, _deps: dict | None = None) -> str:
    manager, err = _manager_or_error(_deps)
    if err:
        return err
    job = await manager.poll(job_id)
    if job is None:
        return f"Error: no background job with id={job_id}"
    output = manager.tail(job_id, tail_lines)
    header = f"status={job.status} pid={job.pid}"
    if job.returncode is not None:
        header += f" returncode={job.returncode}"
    return f"{header}\n{output}" if output else header


@tool(
    name="kill_background_shell",
    group="system",
    description="Stop a background shell job (SIGTERM, escalating to SIGKILL if it doesn't exit in time).",
    parameters={
        "type": "object",
        "properties": {
            "job_id": {"type": "string", "description": "Job id returned by submit_background_shell."},
            "force": {
                "type": "boolean",
                "description": "Skip the SIGTERM grace period and SIGKILL immediately. Default: false.",
            },
        },
        "required": ["job_id"],
    },
    risk_level=RiskLevel.MODERATE,
    idempotent=True,  # killing an already-exited job is a no-op
    deps=["background_shell_manager"],
)
async def kill_background_shell(job_id: str, force: bool = False, _deps: dict | None = None) -> str:
    manager, err = _manager_or_error(_deps)
    if err:
        return err
    job = await manager.kill(job_id, force=force)
    if job is None:
        return f"Error: no background job with id={job_id}"
    return f"status={job.status} pid={job.pid}"
