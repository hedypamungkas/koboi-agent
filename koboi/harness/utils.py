"""koboi/harness/utils -- Shared helpers for harness hooks."""

from __future__ import annotations

import re

_ERROR_PREFIX_RE = re.compile(
    r"^(Error|Rate limited|Denied|Blocked)\b",
    re.IGNORECASE,
)
_ERROR_WORD_RE = re.compile(r"\berror\b", re.IGNORECASE)


def is_tool_error(tool_result: str | None) -> bool:
    """Detect whether a tool result string represents a tool failure.

    Uses explicit prefix matching (``Error:``, ``Rate limited:``, ``Denied``,
    ``Blocked``) and word-boundary ``error`` matching to avoid false positives
    from substrings like ``errorHandler``, ``error_code``, or field names.
    """
    if not tool_result:
        return False
    if _ERROR_PREFIX_RE.search(tool_result):
        return True
    return bool(_ERROR_WORD_RE.search(tool_result))
