"""koboi/orchestration/_utils.py -- Shared helpers for orchestration subpackage."""
from __future__ import annotations

import json


def extract_json(content: str) -> dict | None:
    """Extract JSON from LLM response, handling nested objects."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    start = content.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(content[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if depth == 0:
            try:
                return json.loads(content[start : i + 1])
            except json.JSONDecodeError:
                return None
    return None
