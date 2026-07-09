"""koboi/mcp/tool_server.py -- Expose koboi's own tools over MCP (G9).

``koboi mcp-serve <config>`` builds the agent's ``ToolRegistry`` from a config and
serves its tools as an MCP server over stdio, so external MCP clients (Claude
Desktop, Cursor, another koboi instance) can invoke koboi's tools.

Gating (security): the MCP ``tools/call`` handler drives ``ToolRegistry.execute()``
directly, which BYPASSES the risk/approval/mode/audit pipeline. So the default
exposure is **SAFE-only**; an explicit ``--allow`` list can add named MODERATE
tools; ``--allow-all`` is the only way to expose DESTRUCTIVE tools (dangerous).
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from typing import TYPE_CHECKING

from koboi.config import Config
from koboi.mcp.server import MCPServer
from koboi.types import RiskLevel

if TYPE_CHECKING:
    from koboi.tools.registry import ToolDefinition, ToolRegistry

#: Per-call timeout (seconds) for driving an async koboi tool from the sync MCP handler.
TOOL_CALL_TIMEOUT = 60.0


def select_exposed_tools(
    registry: ToolRegistry, allow: list[str] | None = None, allow_all: bool = False
) -> dict[str, ToolDefinition]:
    """Pick which registered tools to expose, applying the SAFE-only default gate.

    - ``allow_all``: expose everything (including DESTRUCTIVE).
    - otherwise: expose SAFE tools + any named in ``allow`` (MODERATE allowed; DESTRUCTIVE
      still requires ``allow_all``).
    """
    allow_set = set(allow or [])
    out: dict[str, ToolDefinition] = {}
    for name, tdef in registry.list_tools().items():
        risk = tdef.risk_level
        if allow_all:
            out[name] = tdef
        elif risk == RiskLevel.SAFE:
            out[name] = tdef
        elif name in allow_set and risk != RiskLevel.DESTRUCTIVE:
            out[name] = tdef  # allowlist can escalate MODERATE, not DESTRUCTIVE
    return out


def _build_registry(config: Config) -> ToolRegistry:
    """Build the ToolRegistry from config + best-effort sandbox/tool_state deps."""
    from koboi.facade import _build_sandbox, _build_tools

    registry = _build_tools(config)
    # Best-effort: wire sandbox + tool_state so allowlisted fs/shell tools can run.
    # SAFE-only tools (calculator/web/memory/search) typically don't need these.
    try:
        sandbox = _build_sandbox(config, None)
        registry.set_dep("sandbox", sandbox)
    except Exception as e:  # noqa: BLE001
        print(f"[mcp-serve] sandbox not wired ({e}); SAFE-only tools still work", file=sys.stderr)
    try:
        from koboi.tools.state import ToolState

        registry.set_dep("tool_state", ToolState())
    except Exception:  # noqa: BLE001
        pass
    return registry


def _make_sync_handler(name: str, registry: ToolRegistry, loop: asyncio.AbstractEventLoop):
    """Bridge a sync MCP ``tools/call`` to the async ``registry.execute`` on a bg loop."""

    def handler(**arguments) -> str:
        fut = asyncio.run_coroutine_threadsafe(registry.execute(name, json.dumps(arguments)), loop)
        return fut.result(timeout=TOOL_CALL_TIMEOUT)

    return handler


def build_tool_server(
    config_path: str, allow: list[str] | None = None, allow_all: bool = False
) -> tuple[MCPServer, ToolRegistry, asyncio.AbstractEventLoop]:
    """Build (not run) an MCPServer exposing koboi tools. Used by the CLI + tests."""
    config = Config.from_yaml(config_path)
    registry = _build_registry(config)
    exposed = select_exposed_tools(registry, allow=allow, allow_all=allow_all)
    if not exposed:
        print(
            "[mcp-serve] no tools exposed. Configure tools.builtin in the config, "
            "or pass --allow <name> / --allow-all.",
            file=sys.stderr,
        )
        sys.exit(1)

    bg_loop = asyncio.new_event_loop()
    threading.Thread(target=bg_loop.run_forever, daemon=True).start()

    server = MCPServer(name=config.get("agent", "name", default="koboi-agent"))
    for name, tdef in exposed.items():
        schema = tdef.parameters or {"type": "object"}
        server.tool(name, tdef.description or "", schema)(_make_sync_handler(name, registry, bg_loop))
    return server, registry, bg_loop


def serve_koboi_tools(config_path: str, allow: list[str] | None = None, allow_all: bool = False) -> None:
    """``koboi mcp-serve`` entrypoint: build the tool server and run its stdio loop."""
    server, _registry, _loop = build_tool_server(config_path, allow=allow, allow_all=allow_all)
    server.run()
