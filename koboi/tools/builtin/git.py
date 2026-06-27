"""koboi/tools/builtin/git -- Git repository operations (status, log, diff)."""

from __future__ import annotations

import os
import re
import subprocess

from koboi.tools.registry import tool, truncate_text
from koboi.harness.env import build_safe_env

GIT_TIMEOUT = 15
MAX_OUTPUT = 10000
MAX_LOG_COUNT = 50

SAFE_TARGET_RE = re.compile(r"^[a-zA-Z0-9._/@-]+$")


def _run_git(args: list[str], repo_path: str, tool_config: dict | None = None) -> str:
    cfg = tool_config or {}
    timeout = cfg.get("timeout", GIT_TIMEOUT)
    max_output = cfg.get("max_output", MAX_OUTPUT)
    if not os.path.isdir(repo_path):
        return f"Error: path '{repo_path}' is not a directory or not found"

    try:
        result = subprocess.run(
            ["git", "-C", repo_path] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=build_safe_env(cfg),
        )
    except FileNotFoundError:
        return "Error: git not found in system"
    except subprocess.TimeoutExpired:
        return f"Error: git command timed out after {timeout}s"

    output = ""
    if result.stdout:
        output += result.stdout
    if result.stderr and result.returncode != 0:
        if output:
            output += "\n"
        output += result.stderr

    if result.returncode != 0 and not result.stdout:
        return f"Error: git command failed [exit code {result.returncode}]: {output}"

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
def git_status(repo_path: str = ".", _tool_config: dict | None = None) -> str:
    raw = _run_git(["status", "--porcelain"], repo_path, _tool_config)
    if raw.startswith("Error"):
        return raw
    return _parse_status(raw)


@tool(
    name="git_log",
    group="git",
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
def git_log(repo_path: str = ".", count: int = 10, _tool_config: dict | None = None) -> str:
    cfg = _tool_config or {}
    max_log_count = cfg.get("max_log_count", MAX_LOG_COUNT)
    count = max(1, min(count, max_log_count))
    return _run_git(["log", f"-{count}", "--oneline", "--decorate"], repo_path, _tool_config)


@tool(
    name="git_diff",
    group="git",
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
def git_diff(repo_path: str = ".", target: str = "", _tool_config: dict | None = None) -> str:
    if target:
        if not SAFE_TARGET_RE.match(target):
            return "Error: target contains disallowed characters"
        return _run_git(["diff", target], repo_path, _tool_config)

    unstaged = _run_git(["diff"], repo_path, _tool_config)
    staged = _run_git(["diff", "--cached"], repo_path, _tool_config)

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
