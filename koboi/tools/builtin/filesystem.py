"""koboi/tools/builtin/filesystem -- File system operations (read, write, list, delete)."""

from __future__ import annotations

import os
from fnmatch import fnmatch

from koboi.tools.registry import tool
from koboi.types import RiskLevel

_SANDBOX_DIR: str | None = os.environ.get("KOBOI_SANDBOX_DIR")
_MAX_READ_SIZE = 50000

# P3b: read-before-write tracking (advisory, never blocks).
# Module-global mirrors the _SANDBOX_DIR pattern. Populated by read_file and
# consulted by write_file/delete_file to emit an advisory note when writing to
# a path that was never read. Cleared by ReadBeforeWriteResetHook at
# SESSION_START and after a real context compaction.
_read_paths: set[str] = set()


def reset_read_before_write() -> None:
    """Clear the read-before-write tracker (hook: SESSION_START / real compaction)."""
    _read_paths.clear()


def get_read_paths() -> set[str]:
    """Return a copy of the paths recorded by read_file (test/debug visibility)."""
    return set(_read_paths)


def _validate_path(path: str) -> str:
    """Resolve and validate path stays within sandbox (when configured)."""
    resolved = os.path.realpath(path)
    if _SANDBOX_DIR is None:
        return resolved
    sandbox = os.path.realpath(_SANDBOX_DIR)
    if not (resolved.startswith(sandbox + os.sep) or resolved == sandbox):
        raise PermissionError(f"Path '{path}' is outside the sandbox directory")
    return resolved


@tool(
    name="list_files",
    group="file",
    description="List files and directories in a path. Optionally filter by glob pattern.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list",
            },
            "pattern": {
                "type": "string",
                "description": "Optional glob pattern to filter entries, e.g. '*.py' or '*.md'. Default: show all.",
            },
        },
        "required": ["path"],
    },
)
def list_files(path: str, pattern: str | None = None) -> str:
    try:
        path = _validate_path(path)
        entries = os.listdir(path)
        if pattern:
            entries = [e for e in entries if fnmatch(e, pattern)]
        if not entries:
            return f"Directory '{path}' is empty."
        lines = [f"{path}/"]
        for entry in sorted(entries):
            full = os.path.join(path, entry)
            prefix = "\U0001f4c1" if os.path.isdir(full) else "\U0001f4c4"
            lines.append(f"  {prefix} {entry}")
        return "\n".join(lines)
    except FileNotFoundError:
        return f"Error: path '{path}' not found"
    except PermissionError:
        return f"Error: no access to '{path}'"


@tool(
    name="read_file",
    group="file",
    description="Read text file content",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path of the file to read",
            },
        },
        "required": ["path"],
    },
)
def read_file(path: str, _tool_config: dict | None = None) -> str:
    cfg = _tool_config or {}
    max_read_size = cfg.get("max_read_size", _MAX_READ_SIZE)
    try:
        path = _validate_path(path)
        _read_paths.add(path)
        with open(path) as f:
            content = f.read(max_read_size)
        if len(content) == max_read_size:
            content += "\n... (file truncated, too long)"
        return content
    except FileNotFoundError:
        return f"Error: file '{path}' not found"
    except PermissionError:
        return f"Error: no access to '{path}'"
    except IsADirectoryError:
        return f"Error: '{path}' is a directory, not a file"


@tool(
    name="write_file",
    group="file",
    description="Write/create text file",
    risk_level=RiskLevel.DESTRUCTIVE,
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path of the file to write",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file",
            },
        },
        "required": ["path", "content"],
    },
)
def write_file(path: str, content: str) -> str:
    try:
        path = _validate_path(path)
        note = ""
        if path not in _read_paths:
            note = f"\nNote: writing to '{path}' without having read it first -- verify the path is correct."
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to '{path}'{note}"
    except PermissionError:
        return f"Error: no access to write to '{path}'"
    except IsADirectoryError:
        return f"Error: '{path}' is a directory, not a file"


@tool(
    name="delete_file",
    group="file",
    description="Delete file",
    risk_level=RiskLevel.DESTRUCTIVE,
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path of the file to delete",
            },
        },
        "required": ["path"],
    },
)
def delete_file(path: str) -> str:
    try:
        path = _validate_path(path)
        note = ""
        if path not in _read_paths:
            note = f"\nNote: deleting '{path}' without having read it first -- verify the path is correct."
        os.remove(path)
        return f"Successfully deleted '{path}'{note}"
    except FileNotFoundError:
        return f"Error: file '{path}' not found"
    except PermissionError:
        return f"Error: no access to delete '{path}'"
    except IsADirectoryError:
        return f"Error: '{path}' is a directory, not a file"
