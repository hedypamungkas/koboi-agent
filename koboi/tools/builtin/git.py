"""koboi/tools/builtin/git -- Git repository operations (status, log, diff, add, commit, checkout, push)."""

from __future__ import annotations

import os
import re
import subprocess

from koboi.tools.registry import tool, truncate_text
from koboi.types import RiskLevel
from koboi.harness.env import build_safe_env

GIT_TIMEOUT = 15
MAX_OUTPUT = 10000
MAX_LOG_COUNT = 50

SAFE_TARGET_RE = re.compile(r"^[a-zA-Z0-9._/@-]+$")


def _run_git(args: list[str], repo_path: str, tool_config: dict | None = None, sandbox=None) -> str:
    cfg = tool_config or {}
    timeout = cfg.get("timeout", GIT_TIMEOUT)
    max_output = cfg.get("max_output", MAX_OUTPUT)
    try:
        if sandbox is not None:
            # Restricted sandbox contains repo access; passthrough is a no-op
            # unless KOBOI_SANDBOX_DIR is set.
            repo_path = sandbox.validate_path(repo_path)
    except PermissionError as exc:
        return f"Error: {exc}"
    if not os.path.isdir(repo_path):
        return f"Error: path '{repo_path}' is not a directory or not found"

    try:
        if sandbox is not None:
            result = sandbox.run(
                ["git", "-C", repo_path] + args,
                env=sandbox.build_env(cfg),
                timeout=timeout,
                shell=False,
            )
            if result.timed_out:
                return f"Error: git command timed out after {timeout}s"
            rc, stdout, stderr = result.returncode, result.stdout, result.stderr
        else:
            proc = subprocess.run(
                ["git", "-C", repo_path] + args,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=build_safe_env(cfg),
            )
            rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return "Error: git not found in system"
    except subprocess.TimeoutExpired:
        return f"Error: git command timed out after {timeout}s"

    output = ""
    if stdout:
        output += stdout
    if stderr and rc != 0:
        if output:
            output += "\n"
        output += stderr

    if rc != 0 and not stdout:
        return f"Error: git command failed [exit code {rc}]: {output}"

    output = truncate_text(output, max_output)
    return output if output.strip() else "(no output)"


STATUS_MAP = {
    "M": "modified",
    "A": "added",
    "D": "deleted",
    "R": "renamed",
    "C": "copied",
    "?": "untracked",
    "!": "ignored",
}


def _parse_status(raw: str) -> str:
    cleaned = raw.strip()
    if not cleaned or cleaned == "(no output)":
        return "Working tree clean, no changes."

    groups: dict[str, list[str]] = {}
    for line in cleaned.split("\n"):
        if not line or line == "(no output)":
            continue
        xy = line[:2]
        filepath = line[3:]
        code = xy[0] if xy[0] != " " else xy[1]
        label = STATUS_MAP.get(code, "other")
        groups.setdefault(label, []).append(filepath)

    parts = []
    for label, files in groups.items():
        parts.append(f"[{label}]")
        for f in files:
            parts.append(f"  {f}")
    return "\n".join(parts)


@tool(
    name="git_status",
    group="git",
    deps=["sandbox"],
    description="Get git repository status — modified, added, deleted, untracked files.",
    parameters={
        "type": "object",
        "properties": {
            "repo_path": {
                "type": "string",
                "description": "Path to git repository. Default: current directory.",
            },
        },
        "required": [],
    },
)
def git_status(repo_path: str = ".", _tool_config: dict | None = None, _deps: dict | None = None) -> str:
    raw = _run_git(["status", "--porcelain"], repo_path, _tool_config, sandbox=(_deps or {}).get("sandbox"))
    if raw.startswith("Error"):
        return raw
    return _parse_status(raw)


@tool(
    name="git_log",
    group="git",
    deps=["sandbox"],
    description="Get git commit log.",
    parameters={
        "type": "object",
        "properties": {
            "repo_path": {
                "type": "string",
                "description": "Path to git repository. Default: current directory.",
            },
            "count": {
                "type": "integer",
                "description": "Number of commits to show. Default: 10. Max: 50.",
            },
        },
        "required": [],
    },
)
def git_log(repo_path: str = ".", count: int = 10, _tool_config: dict | None = None, _deps: dict | None = None) -> str:
    cfg = _tool_config or {}
    max_log_count = cfg.get("max_log_count", MAX_LOG_COUNT)
    count = max(1, min(count, max_log_count))
    return _run_git(
        ["log", f"-{count}", "--oneline", "--decorate"],
        repo_path,
        _tool_config,
        sandbox=(_deps or {}).get("sandbox"),
    )


@tool(
    name="git_diff",
    group="git",
    deps=["sandbox"],
    description="Get git diff — uncommitted changes or diff against branch/commit.",
    parameters={
        "type": "object",
        "properties": {
            "repo_path": {
                "type": "string",
                "description": "Path to git repository. Default: current directory.",
            },
            "target": {
                "type": "string",
                "description": "Branch or commit to diff against. If empty, shows unstaged changes.",
            },
        },
        "required": [],
    },
)
def git_diff(
    repo_path: str = ".", target: str = "", _tool_config: dict | None = None, _deps: dict | None = None
) -> str:
    sandbox = (_deps or {}).get("sandbox")
    if target:
        if target.startswith("-"):
            return "Error: target cannot start with '-' (option injection guard)"
        if not SAFE_TARGET_RE.match(target):
            return "Error: target contains disallowed characters"
        return _run_git(["diff", target], repo_path, _tool_config, sandbox=sandbox)

    unstaged = _run_git(["diff"], repo_path, _tool_config, sandbox=sandbox)
    staged = _run_git(["diff", "--cached"], repo_path, _tool_config, sandbox=sandbox)

    parts = []
    if unstaged and not unstaged.startswith("Error") and unstaged.strip() != "(no output)":
        parts.append("[Unstaged changes]")
        parts.append(unstaged)
    if staged and not staged.startswith("Error") and staged.strip() != "(no output)":
        parts.append("[Staged changes]")
        parts.append(staged)

    if not parts:
        return "No changes (unstaged or staged)."
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Write tools (Wave 3): add / commit / checkout / push. All reuse _run_git's
# argv path (shell=False, "-C repo") -- no shell interpolation surface.
# --------------------------------------------------------------------------- #

_FALLBACK_IDENTITY = ("koboi-agent", "agent@koboi.local")  # matches pool._git_init_workdir


def _validate_ref(value: str, arg_name: str) -> str | None:
    """Option-injection + charset guard for ref-like args (git_diff precedent)."""
    if value.startswith("-"):
        return f"Error: {arg_name} cannot start with '-' (option injection guard)"
    if not SAFE_TARGET_RE.match(value):
        return f"Error: {arg_name} contains disallowed characters"
    return None


def _identity_args(repo_path: str, tool_config: dict | None, sandbox) -> list[str]:
    """``-c user.name/email`` fallback when the repo has no commit identity.

    build_safe_env strips GIT_* vars, so commits rely on git config. When the
    repo (or allowed global config) already has user.email, return [] -- never
    override an existing identity.
    """
    existing = _run_git(["config", "user.email"], repo_path, tool_config, sandbox=sandbox)
    if existing.startswith("Error") or existing.strip() in ("", "(no output)"):
        name, email = _FALLBACK_IDENTITY
        return ["-c", f"user.name={name}", "-c", f"user.email={email}"]
    return []


@tool(
    name="git_add",
    group="git",
    deps=["sandbox"],
    description="Stage files for commit (git add). Defaults to all changes.",
    risk_level=RiskLevel.MODERATE,
    parameters={
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Paths to stage. Default: ['.'] (all changes).",
            },
            "repo_path": {
                "type": "string",
                "description": "Path to git repository. Default: current directory.",
            },
        },
        "required": [],
    },
)
def git_add(
    paths: list[str] | None = None,
    repo_path: str = ".",
    _tool_config: dict | None = None,
    _deps: dict | None = None,
) -> str:
    # Filter empties (``[""]`` used to collapse to ``[]`` -> ``git add --`` no-op);
    # fall back to the documented default ``['.']`` when nothing remains.
    paths = [p for p in (paths or ["."]) if p] or ["."]
    for p in paths:
        if p.startswith("-"):
            return "Error: paths cannot start with '-' (option injection guard)"
    # "--" separator: path args can never be parsed as options.
    return _run_git(["add", "--"] + paths, repo_path, _tool_config, sandbox=(_deps or {}).get("sandbox"))


@tool(
    name="git_commit",
    group="git",
    deps=["sandbox"],
    description="Commit staged changes (git commit -m). Stage first with git_add.",
    risk_level=RiskLevel.MODERATE,
    # Committing twice creates two commits; crash-resume must not silently
    # replay it (issue #48 semantics), and the Wave-2 checkpointer keys its
    # per-call commits on idempotent=False.
    idempotent=False,
    parameters={
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Commit message",
            },
            "repo_path": {
                "type": "string",
                "description": "Path to git repository. Default: current directory.",
            },
        },
        "required": ["message"],
    },
)
def git_commit(
    message: str,
    repo_path: str = ".",
    _tool_config: dict | None = None,
    _deps: dict | None = None,
) -> str:
    if not message or not message.strip():
        return "Error: commit message must not be empty"
    sandbox = (_deps or {}).get("sandbox")
    identity = _identity_args(repo_path, _tool_config, sandbox)
    result = _run_git(identity + ["commit", "-m", message], repo_path, _tool_config, sandbox=sandbox)
    # git prints "nothing to commit" (clean tree), "no changes added to commit"
    # (unstaged modifications), or "nothing added to commit but untracked files
    # present" (untracked only) -- all mean NO commit was created, but the clean
    # case lands on stdout and reads as success. Surface a clear signal instead.
    # Match the SPECIFIC phrases: a bare "added to commit" substring would false-
    # positive on a successful commit whose message contains those words (git
    # echoes "[branch hash] <message>" on success).
    low = result.lower()
    if "nothing to commit" in low or "no changes added to commit" in low or "nothing added to commit" in low:
        return "No staged changes to commit (stage changes with git_add first)"
    return result


@tool(
    name="git_checkout",
    group="git",
    deps=["sandbox"],
    description="Switch branches or create one (git checkout [-b]).",
    risk_level=RiskLevel.MODERATE,
    parameters={
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "Branch, tag, or commit to check out",
            },
            "create": {
                "type": "boolean",
                "description": "Create the branch (checkout -b). Default: false.",
            },
            "repo_path": {
                "type": "string",
                "description": "Path to git repository. Default: current directory.",
            },
        },
        "required": ["target"],
    },
)
def git_checkout(
    target: str,
    create: bool = False,
    repo_path: str = ".",
    _tool_config: dict | None = None,
    _deps: dict | None = None,
) -> str:
    err = _validate_ref(target, "target")
    if err:
        return err
    args = ["checkout", "-b", target] if create else ["checkout", target]
    return _run_git(args, repo_path, _tool_config, sandbox=(_deps or {}).get("sandbox"))


@tool(
    name="git_push",
    group="git",
    deps=["sandbox"],
    description="Push commits to a remote (git push). No force push.",
    # DESTRUCTIVE: mutates remote state (irreversible for collaborators, can
    # trigger CI/deploys) -- must never be silently auto-approved in autonomous
    # jobs the way MODERATE tools are; enable via Trust-DB / explicit intent.
    risk_level=RiskLevel.DESTRUCTIVE,
    idempotent=False,
    parameters={
        "type": "object",
        "properties": {
            "remote": {
                "type": "string",
                "description": "Remote name. Default: origin.",
            },
            "branch": {
                "type": "string",
                "description": "Branch to push. Default: the current branch (git push <remote> HEAD).",
            },
            "repo_path": {
                "type": "string",
                "description": "Path to git repository. Default: current directory.",
            },
        },
        "required": [],
    },
)
def git_push(
    remote: str = "origin",
    branch: str = "",
    repo_path: str = ".",
    _tool_config: dict | None = None,
    _deps: dict | None = None,
) -> str:
    err = _validate_ref(remote, "remote")
    if err:
        return err
    if branch:
        err = _validate_ref(branch, "branch")
        if err:
            return err
    args = ["push", remote, branch or "HEAD"]
    return _run_git(args, repo_path, _tool_config, sandbox=(_deps or {}).get("sandbox"))
