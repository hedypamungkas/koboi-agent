"""koboi/tools/builtin/search -- Text search (grep) and file finding (glob)."""

from __future__ import annotations

import os
import re
import glob as _glob
from fnmatch import fnmatch

from koboi.tools.registry import tool

# ── grep_search helpers ──

MAX_OUTPUT = 10000
BINARY_CHECK_BYTES = 8192


def _is_binary(filepath: str) -> bool:
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(BINARY_CHECK_BYTES)
        return b"\x00" in chunk
    except Exception:
        return False


def _expand_braces(pattern: str) -> list[str]:
    """Expand brace patterns like '*.{py,js}' into ['*.py', '*.js']."""
    match = re.search(r"\{([^}]+)\}", pattern)
    if not match:
        return [pattern]
    variants = match.group(1).split(",")
    return [pattern[: match.start()] + v + pattern[match.end() :] for v in variants]


def _match_glob(rel_path: str, patterns: list[str]) -> bool:
    """Match relative path against glob patterns, handling ``**`` correctly."""
    for pat in patterns:
        if fnmatch(rel_path, pat):
            return True
        if "/**/" in pat:
            if fnmatch(rel_path, pat.replace("/**/", "/")):
                return True
        if pat.startswith("**/"):
            if fnmatch(rel_path, pat[3:]):
                return True
    return False


@tool(
    name="grep_search",
    group="file",
    deps=["sandbox"],
    description="Search text in files using regex pattern. Like ripgrep/grep.",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search, e.g. 'def \\w+' or 'TODO'",
            },
            "path": {
                "type": "string",
                "description": "Directory path to search",
            },
            "file_filter": {
                "type": "string",
                "description": "Glob pattern to filter files, e.g. '*.py' or '*.{js,ts}'. Default: all files.",
            },
            "context_lines": {
                "type": "integer",
                "description": "Number of context lines above and below match. Default: 0.",
            },
            "output_mode": {
                "type": "string",
                "description": "Output mode: 'content' (default, file:line: content), 'files' (unique filenames), 'count' (match count per file).",
            },
        },
        "required": ["pattern", "path"],
    },
)
def grep_search(
    pattern: str,
    path: str,
    file_filter: str = "",
    context_lines: int = 0,
    output_mode: str = "content",
    _tool_config: dict | None = None,
    _deps: dict | None = None,
) -> str:
    cfg = _tool_config or {}
    max_output = cfg.get("max_output", MAX_OUTPUT)
    try:
        compiled = re.compile(pattern, re.MULTILINE)
    except re.error as e:
        return f"Error: invalid regex pattern: {e}"

    # Anchor the search root through the sandbox so grep_search shares the
    # same root as list_files/read_file/write_file (the per-session workdir).
    # Without this, grep_search walked the process CWD (repo root) while the
    # fs-tools used the workdir -- so the model could grep a file it then
    # couldn't read_file. Falls back to the raw path when no sandbox is wired
    # (direct/test callers), preserving pre-P0 behavior.
    sandbox = (_deps or {}).get("sandbox")
    try:
        search_root = sandbox.validate_path(path) if sandbox is not None else path
    except PermissionError:
        return f"Error: no access to '{path}'"

    if not os.path.isdir(search_root):
        return f"Error: path '{path}' is not a directory or not found"

    filters = _expand_braces(file_filter) if file_filter else []
    output_parts = []
    file_counts: dict[str, int] = {}
    total_matches = 0
    total_size = 0

    for root, dirs, files in os.walk(search_root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, search_root)

            if filters:
                if not _match_glob(rel, filters):
                    continue

            if _is_binary(fpath):
                continue

            try:
                with open(fpath) as f:
                    lines = f.readlines()
            except (PermissionError, UnicodeDecodeError):
                continue
            for i, line in enumerate(lines):
                if compiled.search(line):
                    total_matches += 1
                    file_counts[rel] = file_counts.get(rel, 0) + 1
                    if output_mode == "content":
                        if context_lines > 0:
                            start = max(0, i - context_lines)
                            end = min(len(lines), i + context_lines + 1)
                            for j in range(start, end):
                                prefix = ">" if j == i else " "
                                entry = f"{rel}:{j + 1}:{prefix} {lines[j].rstrip()}"
                                if total_size + len(entry) > max_output:
                                    output_parts.append(f"\n... (truncated, total {total_matches} matches)")
                                    return "\n".join(output_parts)
                                output_parts.append(entry)
                                total_size += len(entry)
                        else:
                            entry = f"{rel}:{i + 1}: {line.rstrip()}"
                            if total_size + len(entry) > max_output:
                                output_parts.append(f"\n... (truncated, total {total_matches} matches)")
                                return "\n".join(output_parts)
                            output_parts.append(entry)
                            total_size += len(entry)

    if not file_counts:
        return f"No match found for pattern '{pattern}' in '{path}'"

    if output_mode == "files":
        return "\n".join(sorted(file_counts.keys()))
    elif output_mode == "count":
        lines = [f"{fname}: {count} matches" for fname, count in sorted(file_counts.items())]
        lines.append(f"\nTotal: {total_matches} matches in {len(file_counts)} files")
        return "\n".join(lines)

    return "\n".join(output_parts)


# ── glob_find ──

MAX_RESULTS = 500


@tool(
    name="glob_find",
    group="file",
    deps=["sandbox"],
    description="Find files by name pattern. Like 'find' or 'glob'.",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to find files, e.g. '**/*.py' or 'src/**/*.ts'",
            },
            "path": {
                "type": "string",
                "description": "Base directory path for search. Default: current working directory.",
            },
        },
        "required": ["pattern"],
    },
)
def glob_find(pattern: str, path: str = "", _tool_config: dict | None = None, _deps: dict | None = None) -> str:
    cfg = _tool_config or {}
    max_results = cfg.get("max_results", MAX_RESULTS)
    raw_base = path or "."
    # Anchor the search root through the sandbox (per-session workdir) so
    # glob_find shares the same root as the other fs-tools. See grep_search for
    # rationale. Falls back to the raw base when no sandbox is wired.
    sandbox = (_deps or {}).get("sandbox")
    try:
        base = sandbox.validate_path(raw_base) if sandbox is not None else raw_base
    except PermissionError:
        return f"Error: no access to '{raw_base}'"
    if not os.path.isdir(base):
        return f"Error: path '{raw_base}' is not a directory or not found"

    full_pattern = os.path.join(base, pattern)
    try:
        results = sorted(_glob.glob(full_pattern, recursive=True))
    except OSError as e:
        return f"Error: invalid pattern: {e}"

    if not results:
        return f"No files matching pattern '{pattern}' in '{base}'"

    if len(results) > max_results:
        shown = results[:max_results]
        lines = [os.path.relpath(r, base) for r in shown]
        lines.append(f"\n... ({len(results)} total results, showing first {max_results})")
        return "\n".join(lines)

    return "\n".join(os.path.relpath(r, base) for r in results)
