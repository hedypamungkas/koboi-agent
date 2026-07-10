"""koboi/mcp/base.py -- Abstract base class and shared utilities for MCP clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from koboi.types import MCPResource, MCPPrompt, MCPToolInfo, RiskLevel


if TYPE_CHECKING:
    from collections.abc import Callable

    from koboi.logger import AgentLogger
    from koboi.tools.registry import ToolRegistry


# Name-based risk inference for MCP tools (opt-in via mcp.servers[].risk_heuristic).
# MCP tools are SAFE by default (pre-#5); this heuristic flags side-effecting names
# so they reach the approval/policy gate when one is configured.
_DESTRUCTIVE_NAME_HINTS = (
    "delete",
    "remove",
    "destroy",
    "drop",
    "purge",
    "wipe",
    "exec",
    "shell",
    "format",
    "truncate",
    "kill",
    "terminate",
    "reset",
)
_MODERATE_NAME_HINTS = (
    "write",
    "create",
    "update",
    "insert",
    "send",
    "post",
    "put",
    "patch",
    "set",
    "add",
    "move",
    "rename",
    "merge",
    "submit",
    "deploy",
    "publish",
    "push",
)


def default_risk_heuristic(info: MCPToolInfo) -> RiskLevel:
    """Infer a RiskLevel from an MCP tool's name (opt-in).

    delete/remove/exec-style names -> DESTRUCTIVE; write/update/send-style -> MODERATE;
    otherwise SAFE. Uses token-boundary matching (not substring) to avoid false
    positives like ``get_deleted_items`` (SAFE, not DESTRUCTIVE).
    """
    import re

    name = (info.name or "").lower()
    # Tokenize on word boundaries (_, ., -, /) so "delete" matches "delete_record"
    # but NOT "get_deleted_items" (where the token is "deleted", not "delete").
    tokens = set(re.split(r"[_.\-/]", name))
    if tokens & set(_DESTRUCTIVE_NAME_HINTS):
        return RiskLevel.DESTRUCTIVE
    if tokens & set(_MODERATE_NAME_HINTS):
        return RiskLevel.MODERATE
    return RiskLevel.SAFE


class MCPError(Exception):
    """Error from MCP server response."""

    def __init__(self, code: int, message: str, data=None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"MCP Error {code}: {message}")


class BaseMCPClient(ABC):
    """Abstract base for MCP clients.

    Both stdio and HTTP transports inherit this. Defines the contract
    used by register_mcp_tools() and the facade.
    """

    def __init__(self, logger: AgentLogger | None = None):
        self.logger = logger
        self._request_id = 0
        self._server_info: dict = {}
        self._tools: list[MCPToolInfo] = []

    # --- Abstract transport methods ---

    @abstractmethod
    def connect(self) -> dict:
        """Establish connection, perform initialize handshake, return server info."""
        ...

    @abstractmethod
    def discover_tools(self) -> list[MCPToolInfo]:
        """Send tools/list, return discovered tools."""
        ...

    @abstractmethod
    async def call_tool(self, name: str, arguments: dict) -> str:
        """Call a tool by name. Must be async (sync transports use asyncio.to_thread)."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release all resources (subprocess, HTTP client, etc)."""
        ...

    @property
    def server_info(self) -> dict:
        return self._server_info

    # --- Status accessors (G6/G7) ---

    #: Transport label overridden by subclasses ("stdio" / "streamable-http").
    TRANSPORT: str = ""

    @property
    def transport(self) -> str:
        """Transport label for this client (e.g. ``"stdio"``)."""
        return self.TRANSPORT

    @property
    def tool_names(self) -> list[str]:
        """Names of the tools discovered from this server."""
        return [t.name for t in self._tools]

    @property
    def name(self) -> str:
        """Human-readable server name from the initialize handshake (best-effort)."""
        info = self._server_info.get("serverInfo") if isinstance(self._server_info, dict) else None
        if isinstance(info, dict) and info.get("name"):
            return str(info["name"])
        return ""

    @property
    def endpoint(self) -> str:
        """Connection target (command line for stdio, URL for HTTP). Override per transport."""
        return ""

    def is_connected(self) -> bool:
        """Whether the transport is currently live. Override per transport (default False)."""
        return False

    # --- Abstract transport I/O ---

    @abstractmethod
    def _send_request_impl(self, method: str, params: dict | None = None) -> dict:
        """Transport-specific: send JSON-RPC request and return parsed result."""
        ...

    @abstractmethod
    def _send_notification_impl(self, method: str, params: dict | None = None) -> None:
        """Transport-specific: send JSON-RPC notification (no response expected)."""
        ...

    # --- Shared JSON-RPC protocol helpers ---

    def _make_request(self, method: str, params: dict | None = None) -> dict:
        """Build JSON-RPC 2.0 request envelope."""
        self._request_id += 1
        msg: dict = {"jsonrpc": "2.0", "id": self._request_id, "method": method}
        if params is not None:
            msg["params"] = params
        return msg

    def _make_notification(self, method: str, params: dict | None = None) -> dict:
        """Build JSON-RPC 2.0 notification envelope (no id)."""
        msg: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        return msg

    def _check_response(self, response: dict) -> dict:
        """Raise MCPError if response contains an error, else return result."""
        if "error" in response:
            err = response["error"]
            raise MCPError(
                code=err.get("code", -1),
                message=err.get("message", "Unknown error"),
                data=err.get("data"),
            )
        return response.get("result", {})

    def _do_initialize_handshake(self, connect_desc: list | str | None = None) -> dict:
        """Perform the standard MCP initialize/initialized exchange.

        Subclasses call this from their connect() after setting up transport.

        Args:
            connect_desc: Command list or URL string for logging purposes.
        """
        result = self._send_request_impl(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "koboi-agent", "version": "0.1.0"},
            },
        )
        self._server_info = result
        self._send_notification_impl("initialized")

        if self.logger:
            desc = connect_desc if connect_desc is not None else []
            if isinstance(desc, str):
                desc = [desc]
            self.logger.log_mcp_connect(desc, result)

        return result

    # --- Shared content extraction ---

    @staticmethod
    def _extract_tool_result(result: dict) -> str:
        """Parse MCP tools/call response into a string.

        Extracts text content items, deduplicates identical texts,
        and joins with newlines.
        """
        content = result.get("content", [])
        parts: list[str] = []
        seen: set[str] = set()
        for item in content:
            if item.get("type") == "text":
                text = item.get("text", "")
                if text not in seen:
                    seen.add(text)
                    parts.append(text)
        return "\n".join(parts) if parts else str(result)

    # --- resources / prompts primitives (G2) ---
    # These are transport-agnostic: they reuse _send_request_impl, so both stdio and
    # HTTP transports inherit them for free (mirrors how _do_initialize_handshake works).

    def list_resources(self) -> list[MCPResource]:
        """Send resources/list, return the server's resource descriptors."""
        result = self._send_request_impl("resources/list")
        return [
            MCPResource(
                uri=r.get("uri", ""),
                name=r.get("name", ""),
                description=r.get("description", ""),
                mime_type=r.get("mimeType"),
            )
            for r in result.get("resources", [])
        ]

    def read_resource(self, uri: str) -> str:
        """Send resources/read, return the concatenated text contents."""
        result = self._send_request_impl("resources/read", {"uri": uri})
        parts = [c.get("text", "") for c in result.get("contents", []) if c.get("text")]
        return "\n".join(parts) if parts else str(result)

    def list_prompts(self) -> list[MCPPrompt]:
        """Send prompts/list, return the server's prompt descriptors."""
        result = self._send_request_impl("prompts/list")
        return [
            MCPPrompt(
                name=p.get("name", ""),
                description=p.get("description", ""),
                arguments=p.get("arguments", []),
            )
            for p in result.get("prompts", [])
        ]

    def get_prompt(self, name: str, arguments: dict | None = None) -> str:
        """Send prompts/get, return the rendered prompt text."""
        result = self._send_request_impl("prompts/get", {"name": name, "arguments": arguments or {}})
        parts: list[str] = []
        for msg in result.get("messages", []):
            content = msg.get("content", {})
            if isinstance(content, dict) and content.get("text"):
                parts.append(content["text"])
            elif isinstance(content, str):
                parts.append(content)
        return "\n".join(parts) if parts else str(result)


def register_mcp_tools(
    client: BaseMCPClient,
    registry: ToolRegistry,
    group: str | None = None,
    risk_level: RiskLevel = RiskLevel.SAFE,
    namespace_prefix: str | None = None,
    risk_resolver: Callable[[MCPToolInfo], RiskLevel] | None = None,
) -> list[str]:
    """Bridge: discover MCP tools and register them to ToolRegistry.

    Each MCP tool is wrapped into an async closure that calls
    client.call_tool(). Works with any BaseMCPClient subclass.

    Args:
        group: Optional group name to assign to all tools from this server.
        risk_level: RiskLevel for every tool from this server (default SAFE).
        namespace_prefix: Optional prefix (G8); registers each tool as
            ``{prefix}__{name}`` to avoid colliding with builtins / other servers.
        risk_resolver: Optional per-tool resolver; overrides risk_level when set
            (e.g. :func:`default_risk_heuristic`). A non-SAFE level only gates when
            ``guardrails.approval`` or ``policy.rules`` is configured.
    """
    tool_infos = client.discover_tools()
    registered: list[str] = []

    for info in tool_infos:

        def make_handler(tool_name: str, mcp_client: BaseMCPClient):
            async def handler(**kwargs) -> str:
                return await mcp_client.call_tool(tool_name, kwargs)

            return handler  # noqa: B023 - factory binds tool_name/mcp_client as params (not loop vars)

        handler = make_handler(info.name, client)
        reg_name = f"{namespace_prefix}__{info.name}" if namespace_prefix else info.name
        resolved_risk = risk_resolver(info) if risk_resolver else risk_level
        registry.register(
            name=reg_name,
            description=info.description,
            parameters=info.input_schema,
            fn=handler,
            risk_level=resolved_risk,
            group=group,
        )
        registered.append(reg_name)

    return registered
