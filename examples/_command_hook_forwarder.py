#!/usr/bin/env python3
"""Standalone forwarder for ``examples/33_command_hook_messaging.py``.

Reads the koboi ``HookContext`` JSON from **stdin** and "forwards" the LLM
response to the file given as **argv[1]** -- a stand-in for a real WhatsApp /
Telegram / Slack webhook. In production this script would POST to your messaging
API; here it appends a line so the example can show the side effect with no
network credentials.

This is exactly the kind of separate, executable script koboi command hooks are
designed for. Run it via::

    uv run examples/_command_hook_forwarder.py OUTFILE      # local file + inline deps
    uvx my-wa-forwarder OUTFILE                             # a published, zero-install tool

Protocol (see ``docs/custom-hooks.md``): koboi sends JSON on stdin like
``{"event": "post_output", "llm_response": {"content": "..."}, ...}``.
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    payload = json.load(sys.stdin)
    content = (payload.get("llm_response") or {}).get("content") or payload.get("user_message") or ""
    event = payload.get("event", "?")
    outfile = sys.argv[1]
    with open(outfile, "a") as f:
        f.write(f"[{event}] {content}\n")


if __name__ == "__main__":
    main()
