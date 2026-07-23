"""koboi/tools/builtin/repo_map -- Directory tree + best-effort symbol outline."""

from __future__ import annotations

import ast
import os
import re

from koboi.tools.registry import tool

MAX_DEPTH = 3
MAX_ENTRIES = 300

# Cheap correctness win without a full .gitignore parser (deliberately out of
# scope -- see Wave 4 plan). Dot-prefixed dirs are skipped unconditionally.
_JUNK_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
}

# Best-effort, NOT language-aware -- a plain line-grep for common def keywords
# across non-Python files. Good enough for a "lay of the land" outline; do not
# over-trust it (misses multi-line signatures, decorators-as-names, etc).
_NON_PY_SYMBOL_RE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function|class|def|func|interface)\s+([A-Za-z_$][\w$]*)"
)


def _within_base(path: str, base_real: str) -> bool:
    """True if realpath(path) is ``base_real`` itself or a descendant of it.

    Mirrors the containment check in search.py's glob_find -- a symlinked file
    inside the walked tree must not have its content read from outside the
    sandbox root.
    """
    rr = os.path.realpath(path)
    return rr == base_real or rr.startswith(base_real + os.sep)


def _format_args(args: ast.arguments) -> str:
    """Render a function's parameter list without default-value expressions."""
    try:
        parts = [a.arg for a in args.posonlyargs] + [a.arg for a in args.args]
        defaults = args.defaults
        if defaults:
            offset = len(parts) - len(defaults)
            for i in range(len(defaults)):
                idx = offset + i
                if 0 <= idx < len(parts):
                    parts[idx] = f"{parts[idx]}=..."
        if args.vararg:
            parts.append(f"*{args.vararg.arg}")
        parts.extend(a.arg for a in args.kwonlyargs)
        if args.kwarg:
            parts.append(f"**{args.kwarg.arg}")
        return ", ".join(parts)
    except Exception:  # nosec - best-effort formatting, never fail the map
        return "..."


def _python_symbols(fpath: str) -> list[str]:
    """Top-level function/class names + signatures via ast.parse.

    Malformed Python (SyntaxError) or unreadable files return [] rather than
    raising -- a single bad file must not crash the whole map. Only
    module-level defs are captured, not nested methods (keeps the outline
    short and fast).
    """
    try:
        with open(fpath, encoding="utf-8") as f:
            source = f.read()
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    symbols: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(f"{node.name}({_format_args(node.args)})")
        elif isinstance(node, ast.ClassDef):
            symbols.append(node.name)
    return symbols


def _regex_symbols(fpath: str, max_lines: int = 2000) -> list[str]:
    symbols: list[str] = []
    try:
        with open(fpath, encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                m = _NON_PY_SYMBOL_RE.match(line)
                if m:
                    symbols.append(m.group(1))
    except OSError:
        return []
    return symbols


@tool(
    name="repo_map",
    group="file",
    deps=["sandbox"],
    description=(
        "Render a directory tree with a best-effort symbol outline (function/class names + "
        "signatures, no bodies). Python files get real AST-based extraction; other languages get "
        "a best-effort regex scan that is NOT language-aware -- don't over-trust it for non-Python "
        "files. Use this to get a fast lay-of-the-land before diving into read_file/grep_search."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Root directory to map. Default: '.'"},
            "max_depth": {"type": "integer", "description": "Max directory depth to descend. Default: 3."},
            "max_entries": {"type": "integer", "description": "Max total entries (files+dirs). Default: 300."},
            "include_symbols": {
                "type": "boolean",
                "description": "Include a function/class outline per file. Default: true.",
            },
        },
        "required": [],
    },
)
def repo_map(
    path: str = ".",
    max_depth: int = MAX_DEPTH,
    max_entries: int = MAX_ENTRIES,
    include_symbols: bool = True,
    _deps: dict | None = None,
) -> str:
    sandbox = (_deps or {}).get("sandbox")
    try:
        root = sandbox.validate_path(path) if sandbox is not None else path
    except PermissionError:
        return f"Error: no access to '{path}'"
    if not os.path.isdir(root):
        return f"Error: path '{path}' is not a directory or not found"

    base_real = os.path.realpath(root)
    header = os.path.basename(os.path.normpath(root)) or root
    lines: list[str] = [f"{header}/"]
    count = 1
    truncated = False

    for dirpath, dirnames, filenames in os.walk(root):
        if truncated:
            break

        rel_dir = os.path.relpath(dirpath, root)
        depth = 0 if rel_dir == "." else rel_dir.count(os.sep) + 1
        dirnames[:] = sorted(d for d in dirnames if d not in _JUNK_DIRS and not d.startswith("."))
        if depth >= max_depth:
            dirnames[:] = []

        if rel_dir != ".":
            lines.append(f"{'  ' * depth}{os.path.basename(dirpath)}/")
            count += 1
            if count >= max_entries:
                truncated = True
                continue

        file_depth = depth + 1
        for fname in sorted(filenames):
            if count >= max_entries:
                truncated = True
                break
            fpath = os.path.join(dirpath, fname)
            lines.append(f"{'  ' * file_depth}{fname}")
            count += 1
            if include_symbols and _within_base(fpath, base_real):
                symbols = _python_symbols(fpath) if fname.endswith(".py") else _regex_symbols(fpath)
                if symbols:
                    lines.append(f"{'  ' * file_depth}  symbols: {', '.join(symbols)}")

    if truncated:
        lines.append(f"... truncated at {max_entries} entries")

    return "\n".join(lines)
