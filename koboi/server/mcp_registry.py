"""koboi/server/mcp_registry.py -- Per-session MCP server registry (G6) + attach gate.

Tracks the MCP clients attached to a pooled session under stable ids so the
``/v1/sessions/{id}/mcp/servers`` endpoints can list / reconnect / remove them.
The clients themselves live on ``KoboiAgent._mcp_clients`` (and their tools on
``AgentCore.tools``); this registry only owns the id mapping + read/remove/reconnect
helpers. In-process, session-scoped, not persisted across restart/eviction.

Also hosts :func:`check_stdio_attach` -- the issue #91 default-deny gate the POST
route applies to a caller-supplied stdio ``command``/``args`` before anything is
spawned.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from koboi.mcp.base import BaseMCPClient
    from koboi.tools.registry import ToolRegistry

_log = logging.getLogger(__name__)

# --- Issue #91: runtime stdio-attach gate (untrusted HTTP callers) ---------
#
# The YAML/CLI path is authored by a trusted operator, so ``facade._create_mcp_client``
# only basename-checks the runner against ``_MCP_DEFAULT_RUNNERS``. That threat model
# does NOT hold for ``POST /v1/sessions/{id}/mcp/servers``, which any authenticated
# tenant can call: every default runner (python/node/npx/uv/deno/bun/...) executes
# arbitrary code handed to it in ``args``. This gate is default-deny and never falls
# back to ``_MCP_DEFAULT_RUNNERS``.

#: Long flags that make an interpreter evaluate inline code. Mirrors the vocabulary
#: in ``koboi/harness/policy.py`` (C/E interpreters take ``-c``/``-e``) rather than
#: importing that module's private helpers across the harness/server boundary.
_INLINE_CODE_LONG_FLAGS = frozenset({"--eval", "--command"})
#: Short-flag characters meaning "inline code". Checked per-character so combined
#: forms (``-Sc``, ``-ic``, ``-We``) are caught the way ``policy._interpreter_deny_reason``
#: catches ``bash -ic``.
_INLINE_CODE_SHORT_CHARS = frozenset("ce")


def _inline_code_arg(args: Sequence[str]) -> str | None:
    """Return the first arg that would make an interpreter evaluate inline code.

    Catches ``-c`` / ``-e`` (including combined short flags like ``-Sc``),
    ``--eval`` / ``--command`` (with or without an ``=value`` suffix), and a bare
    ``-`` (read the program from stdin). Returns ``None`` when every arg is inert.
    """
    for raw in args:
        arg = str(raw)
        if arg == "-":
            return arg
        if arg.startswith("--"):
            if arg.split("=", 1)[0].lower() in _INLINE_CODE_LONG_FLAGS:
                return arg
        elif arg.startswith("-") and len(arg) > 1:
            flags = arg[1:].split("=", 1)[0].lower()
            if _INLINE_CODE_SHORT_CHARS & set(flags):
                return arg
    return None


def check_stdio_attach(
    command: str,
    args: Sequence[str],
    *,
    allow_stdio: bool,
    allowed_commands: Sequence[str],
) -> tuple[str, str] | None:
    """Vet an untrusted runtime stdio attach. Returns ``(error_code, message)`` or ``None``.

    ``None`` means the attach may proceed. Three fail-closed rungs, in order:

    1. ``allow_stdio`` is off (the default) -> nothing stdio is attachable at runtime.
    2. ``command`` is not an EXACT match for an entry in the operator's
       ``allowed_commands`` -- no basename matching (``/tmp/evil/python3`` must not
       ride in on a ``python3`` entry) and no ``_MCP_DEFAULT_RUNNERS`` fallback.
       An empty list therefore attaches nothing.
    3. ``args`` carry an interpreter inline-code / stdin-eval flag, which would turn
       even an allow-listed command into arbitrary execution.
    """
    if not allow_stdio:
        return (
            "stdio_attach_disabled",
            "runtime stdio MCP attach is disabled; set server.mcp_runtime_attach.allow_stdio",
        )
    if not command or command not in set(allowed_commands):
        return (
            "stdio_command_not_allowed",
            "stdio command is not in server.mcp_runtime_attach.allowed_commands",
        )
    bad = _inline_code_arg(args)
    if bad is not None:
        return (
            "stdio_args_rejected",
            f"stdio arg {bad!r} is an interpreter inline-code flag and is refused",
        )
    return None


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
