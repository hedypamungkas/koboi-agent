"""Example 31: tool selection + secret hygiene.

Demonstrates (no API key required -- exercises config/tool/env machinery only):
  - tools.disabled  : denylist that removes a tool from BOTH the LLM view and execution
  - tools.groups    : hide tool groups from the LLM while keeping them executable
  - tools.defaults env_* : secret-hygiene env filtering for subprocess tools

Run:
    python examples/31_tool_selection.py
"""

from __future__ import annotations

import asyncio
import os

from koboi.config import Config
from koboi.facade import _build_tools
from koboi.harness.env import build_safe_env


async def main() -> None:
    config = Config.from_yaml("examples/31_tool_selection.yaml")
    registry = _build_tools(config)

    # 1) tools.disabled + tools.groups shape what the LLM can see.
    advertised = sorted(d["function"]["name"] for d in registry.get_definitions())
    print("1) Tools advertised to the LLM:", advertised)
    print("   run_shell removed by tools.disabled; web_search hidden by tools.groups=[math, file]")

    # 2) A disabled tool is gone from execution too (defense in depth).
    result = await registry.execute("run_shell", '{"command": "echo hi"}')
    print("\n2) run_shell.execute ->", result)

    # 3) Secret hygiene: the subprocess env is sanitized by default.
    os.environ["MY_SERVICE_TOKEN"] = "tk-would-leak-123"  # secret-shaped (*_TOKEN)
    os.environ["CARGO_HOME"] = "/fake/cargo"  # allow-listed in the YAML
    env = build_safe_env(config.get("tools", "defaults", default={}))
    print("\n3) build_safe_env():")
    print("   MY_SERVICE_TOKEN stripped (*_TOKEN): ", "MY_SERVICE_TOKEN" not in env)
    print("   CARGO_HOME allowed (env_allowlist):  ", env.get("CARGO_HOME") == "/fake/cargo")
    print("   PATH preserved (default allow-list): ", "PATH" in env)


if __name__ == "__main__":
    asyncio.run(main())
