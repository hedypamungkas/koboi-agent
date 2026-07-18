"""koboi/tools/builtin/filesystem -- File system operations (read, write, list, delete)."""

from __future__ import annotations

import os
import tempfile
from fnmatch import fnmatch

from koboi.tools.registry import tool
from koboi.types import RiskLevel

# P0b: kept as a legacy back-compat fallback only. Containment is now driven
# per-agent via the ``sandbox`` dep (sandbox.validate_path), which the facade
# always wires. KOBOI_SANDBOX_DIR still works for callers that invoke the
# tools directly without _deps (e.g. some tests).
_SANDBOX_DIR: str | None = os.environ.get("KOBOI_SANDBOX_DIR")
_MAX_READ_SIZE = 50000

# P3b: read-before-write tracking (advisory, never blocks).
# Module-global mirrors the _SANDBOX_DIR pattern. Populated by read_file and
# consulted by write_file/delete_file to emit an advisory note when writing to
# a path that was never read. Cleared by ReadBeforeWriteResetHook at
# SESSION_START and after a real context compaction.
_read_paths: set[str] = set()


def reset_read_before_write(tool_state=None) -> None:
    """Clear the read-before-write tracker (hook: SESSION_START / real compaction).

    If ``tool_state`` is given, clear only that per-session set; otherwise clear
    the module global (direct/test callers + the legacy reset hook).
    """
    if tool_state is not None:
        tool_state.read_paths.clear()
    _read_paths.clear()


def get_read_paths() -> set[str]:
    """Return a copy of the paths recorded by read_file (test/debug visibility)."""
    return set(_read_paths)


def _read_paths_for(_deps: dict | None) -> set[str]:
    """M6: the per-session read-path set when wired, else the module global.

    Direct callers (e.g. unit tests) pass no _deps and keep using the legacy
    module-global set; per-session agents get an isolated ToolState.
    """
    ts = (_deps or {}).get("tool_state")
    return ts.read_paths if ts is not None else _read_paths


def _validate_path(path: str, sandbox=None) -> str:
    """Resolve and validate path stays within the sandbox.

    Prefers the per-registry sandbox handle (delegates to
    ``sandbox.validate_path``); falls back to the legacy KOBOI_SANDBOX_DIR env
    var when no sandbox is wired so existing setups keep working.
    """
    if sandbox is not None:
        return sandbox.validate_path(path)
    resolved = os.path.realpath(path)
    if _SANDBOX_DIR is None:
        return resolved
    sandbox_dir = os.path.realpath(_SANDBOX_DIR)
    if not (resolved.startswith(sandbox_dir + os.sep) or resolved == sandbox_dir):
        raise PermissionError(f"Path '{path}' is outside the sandbox directory")
    return resolved


@tool(
    name="list_files",
    group="file",
    deps=["sandbox"],
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
def list_files(path: str, pattern: str | None = None, _deps: dict | None = None) -> str:
    try:
        path = _validate_path(path, sandbox=(_deps or {}).get("sandbox"))
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
    deps=["sandbox"],
    description=(
        "Read text file content. For large files, pass offset/limit to read a line range (returned with line numbers)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path of the file to read",
            },
            "offset": {
                "type": "integer",
                "description": "1-based line number to start reading from (optional)",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read (optional; default: to end of file)",
            },
        },
        "required": ["path"],
    },
)
def read_file(
    path: str,
    offset: int | None = None,
    limit: int | None = None,
    _tool_config: dict | None = None,
    _deps: dict | None = None,
) -> str:
    cfg = _tool_config or {}
    max_read_size = cfg.get("max_read_size", _MAX_READ_SIZE)
    try:
        path = _validate_path(path, sandbox=(_deps or {}).get("sandbox"))
        _read_paths.add(path)  # legacy global (back-compat / test visibility)
        ts = (_deps or {}).get("tool_state")  # M6: per-session tracking
        if ts is not None:
            ts.read_paths.add(path)
        if offset is not None or limit is not None:
            return _read_file_range(path, offset, limit, max_read_size)
        with open(path) as f:
            content = f.read(max_read_size)
        if len(content) == max_read_size:
            content += f"\n... (truncated at {max_read_size} chars -- use offset/limit to read the rest)"
        return content
    except FileNotFoundError:
        return f"Error: file '{path}' not found"
    except PermissionError:
        return f"Error: no access to '{path}'"
    except IsADirectoryError:
        return f"Error: '{path}' is a directory, not a file"


def _read_file_range(path: str, offset: int | None, limit: int | None, max_read_size: int) -> str:
    """Return a numbered line range of ``path`` (1-based ``offset``, ``limit`` lines)."""
    start = max(1, offset or 1)
    with open(path) as f:
        lines = f.readlines()
    total = len(lines)
    if start > total:
        return f"Error: offset {start} is beyond end of file '{path}' ({total} lines)"
    end = total if limit is None else min(total, start + max(1, limit) - 1)
    selected = lines[start - 1 : end]
    body = "".join(f"{i:>6}\t{line}" for i, line in enumerate(selected, start=start))
    if len(body) > max_read_size:
        body = body[:max_read_size] + f"\n... (truncated at {max_read_size} chars -- narrow the range)"
    return f"(lines {start}-{end} of {total} in '{path}')\n{body}"


@tool(
    name="write_file",
    group="file",
    deps=["sandbox"],
    description="Write/create text file",
    risk_level=RiskLevel.DESTRUCTIVE,
    # Issue #48: writing mutates the fs; on crash-resume the loop's
    # _repair_interrupted_turn must NOT silently replay it (double writes).
    idempotent=False,
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
def write_file(path: str, content: str, _deps: dict | None = None) -> str:
    try:
        path = _validate_path(path, sandbox=(_deps or {}).get("sandbox"))
        note = ""
        if path not in _read_paths_for(_deps):  # M6: per-session when wired
            note = f"\nNote: writing to '{path}' without having read it first -- verify the path is correct."
        # M11: accepted residual risk -- validate_path resolves at validation time
        # and open() happens later (a TOCTOU via symlink swap). Low-risk in this
        # single-process agent (no concurrent attacker within the workdir). A future
        # hardening pass can switch the leaf to os.open(path, O_WRONLY|O_NOFOLLOW).
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to '{path}'{note}"
    except PermissionError:
        return f"Error: no access to write to '{path}'"
    except IsADirectoryError:
        return f"Error: '{path}' is a directory, not a file"


@tool(
    name="edit_file",
    group="file",
    deps=["sandbox"],
    description=(
        "Edit a text file by replacing an exact string. old_string must match "
        "exactly (including whitespace) and be unique in the file unless "
        "replace_all is set. Preferred over write_file for modifying existing files."
    ),
    risk_level=RiskLevel.DESTRUCTIVE,
    # Issue #48: editing mutates the fs; on crash-resume the loop's
    # _repair_interrupted_turn must NOT silently replay it (double edits).
    idempotent=False,
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path of the file to edit",
            },
            "old_string": {
                "type": "string",
                "description": "Exact text to replace (must be unique in the file unless replace_all)",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace every occurrence of old_string (default: false)",
            },
        },
        "required": ["path", "old_string", "new_string"],
    },
)
def edit_file(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    _deps: dict | None = None,
) -> str:
    if old_string == new_string:
        return "Error: old_string and new_string are identical -- nothing to change"
    try:
        path = _validate_path(path, sandbox=(_deps or {}).get("sandbox"))
        note = ""
        if path not in _read_paths_for(_deps):  # M6: per-session when wired
            note = f"\nNote: editing '{path}' without having read it first -- verify the path is correct."
        with open(path) as f:
            content = f.read()
        count = content.count(old_string)
        if count == 0:
            return f"Error: old_string not found in '{path}' -- re-read the file and match the exact text"
        if count > 1 and not replace_all:
            return (
                f"Error: old_string matched {count} times in '{path}'; provide more "
                "surrounding context to make it unique, or set replace_all"
            )
        new_content = content.replace(old_string, new_string)
        # Atomic swap: write to a temp file in the same directory, then os.replace,
        # so a crash mid-write can never leave a truncated file behind.
        dir_ = os.path.dirname(path) or "."
        fd, tmp = tempfile.mkstemp(dir=dir_, prefix=".edit_file_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(new_content)
            os.chmod(tmp, os.stat(path).st_mode)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        replaced = count if replace_all else 1
        return f"Successfully replaced {replaced} occurrence(s) in '{path}'{note}"
    except FileNotFoundError:
        return f"Error: file '{path}' not found"
    except PermissionError:
        return f"Error: no access to '{path}'"
    except IsADirectoryError:
        return f"Error: '{path}' is a directory, not a file"


@tool(
    name="delete_file",
    group="file",
    deps=["sandbox"],
    description="Delete file",
    risk_level=RiskLevel.DESTRUCTIVE,
    # Issue #48: deleting mutates the fs; on crash-resume the loop's
    # _repair_interrupted_turn must NOT silently replay it (double deletes).
    idempotent=False,
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
def delete_file(path: str, _deps: dict | None = None) -> str:
    try:
        path = _validate_path(path, sandbox=(_deps or {}).get("sandbox"))
        note = ""
        if path not in _read_paths_for(_deps):  # M6: per-session when wired
            note = f"\nNote: deleting '{path}' without having read it first -- verify the path is correct."
        os.remove(path)
        return f"Successfully deleted '{path}'{note}"
    except FileNotFoundError:
        return f"Error: file '{path}' not found"
    except PermissionError:
        return f"Error: no access to delete '{path}'"
    except IsADirectoryError:
        return f"Error: '{path}' is a directory, not a file"
