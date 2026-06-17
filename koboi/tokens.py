from __future__ import annotations

import json


def estimate_single(msg: dict) -> int:
    total = 0
    for val in msg.values():
        if isinstance(val, str):
            total += len(val)
        elif isinstance(val, (list, dict)):
            total += len(json.dumps(val, ensure_ascii=False))
        elif val:
            total += len(str(val))
    return max(total // 3, 1)


def estimate_tokens(messages: list[dict]) -> int:
    return sum(estimate_single(m) for m in messages)
