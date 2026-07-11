"""koboi/server/mcp_registry.py -- Per-session MCP server registry (G6).

Tracks the MCP clients attached to a pooled session under stable ids so the
``/v1/sessions/{id}/mcp/servers`` endpoints can list / reconnect / remove them.
The clients themselves live on ``KoboiAgent._mcp_clients`` (and their tools on
``AgentCore.tools``); this registry only owns the id mapping + read/remove/reconnect
helpers. In-process, session-scoped, not persisted across restart/eviction.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from koboi.mcp.base import BaseMCPClient
    from koboi.tools.registry import ToolRegistry

_log = logging.getLogger(__name__)


class SessionMcpRegistry:
    """id -> MCP client for one session."""

    def __init__(self) -> None:
        self._clients: dict[str, BaseMCPClient] = {}

    @staticmethod
    def _slug(client: BaseMCPClient) -> str:
        base = client.name or client.endpoint or client.transport or "server"
        slug = re.sub(r"[^a-z0-9]+", "-", str(base).lower()).strip("-")
        return slug or "server"

    def ensure_populated(self, mcp_clients: list[BaseMCPClient]) -> None:
        """Index any session clients not yet tracked (idempotent). Uses deterministic slug ids;
        collisions get a short uuid suffix."""
        indexed = {id(c) for c in self._clients.values()}
        for client in mcp_clients:
            if id(client) in indexed:
                continue
            sid = self._slug(client)
            if sid in self._clients:
                sid = f"{sid}-{uuid.uuid4().hex[:6]}"
            self._clients[sid] = client

    def register(self, client: BaseMCPClient) -> str:
        """Assign a fresh (uuid-suffixed) id to a newly added client. Returns the id.

        29-C: identity-aware -- ``_mcp_registry_for`` calls ``ensure_populated`` which
        may already index this client object (by slug). Return that id instead of
        inserting a second entry, which would make GET list it twice and leave an
        orphan on DELETE."""
        for sid, existing in self._clients.items():
            if existing is client:
                return sid
        sid = f"{self._slug(client)}-{uuid.uuid4().hex[:6]}"
        self._clients[sid] = client
        return sid

    def get(self, sid: str) -> BaseMCPClient | None:
        return self._clients.get(sid)

    def remove(self, sid: str, registry: ToolRegistry | None, mcp_clients: list[BaseMCPClient]) -> bool:
        """Disable the client's tools, close it, drop from the agent's list. False if no such id.

        29-H: a failed ``close()`` (subprocess won't die, httpx teardown error) is logged
        so the operator can see the lingering resource, instead of returning silent success."""
        client = self._clients.pop(sid, None)
        if client is None:
            return False
        if registry is not None:
            try:
                registry.disable(list(client.tool_names))
            except Exception as e:  # noqa: BLE001  # nosec B110 - best-effort cleanup
                _log.warning("MCP tool disable failed for %r: %s", client.name, e)
        try:
            client.close()
        except Exception as e:  # noqa: BLE001  # nosec B110 - best-effort cleanup
            _log.warning("MCP client close failed for %r: %s", client.name, e)
        if client in mcp_clients:
            mcp_clients.remove(client)
        return True

    def reconnect(self, sid: str) -> bool:
        """close() + connect() the client (respawn stdio / re-handshake HTTP). False if no such id.

        29-H: a failed ``close()`` before respawn is logged (the old process may leak)."""
        client = self._clients.get(sid)
        if client is None:
            return False
        try:
            client.close()
        except Exception as e:  # noqa: BLE001  # nosec B110 - best-effort cleanup
            _log.warning("MCP client close-before-reconnect failed for %r: %s", client.name, e)
        client.connect()  # raises on failure -> caller maps to an error response
        return True

    def status(self) -> list[dict]:
        return [
            {
                "id": sid,
                "name": client.name or sid,
                "transport": client.transport,
                "connected": client.is_connected(),
                "tool_names": list(client.tool_names),
                "server_info": client.server_info,
            }
            for sid, client in self._clients.items()
        ]
