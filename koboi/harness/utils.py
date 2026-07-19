"""koboi/harness/utils -- Shared helpers for harness hooks."""

from __future__ import annotations

import re

_ERROR_PREFIX_RE = re.compile(
    r"^(Error|Rate limited|Denied|Blocked)\b",
    re.IGNORECASE,
)
# run_shell's authoritative failure token (shell.py _format_result): the prefix
# is only ever emitted when the exit code is non-zero.
_EXIT_CODE_RE = re.compile(r"^\[exit code: (\d+)\]")


def parse_exit_code(tool_result: str | None) -> int | None:
    """Extract the shell exit code from a ``[exit code: N]``-prefixed result.

    Returns None when the result carries no exit-code prefix (which includes
    every successful shell run -- run_shell only stamps the prefix on N != 0).
    """
    if not tool_result:
        return None
    m = _EXIT_CODE_RE.match(tool_result)
    return int(m.group(1)) if m else None


def is_tool_error(tool_result: str | None) -> bool:
    """Detect whether a tool result string represents a tool failure.

    Signals, in order: the shell's ``[exit code: N]`` prefix (authoritative --
    N is non-zero whenever present), then explicit prefix matching (``Error:``,
    ``Rate limited:``, ``Denied``, ``Blocked``). The old word-boundary
    ``\\berror\\b`` scan was removed: it false-positived on successful output
    that merely *mentions* errors ("ran with 0 errors", pytest summaries),
    which made doom-loop/telemetry misread a coding agent's normal
    edit->test iterations. Builtin tools signal failure via the ``Error:``
    prefix convention.
    """
    if not tool_result:
        return False
    if parse_exit_code(tool_result) not in (None, 0):
        return True
    return bool(_ERROR_PREFIX_RE.search(tool_result))
