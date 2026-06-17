"""koboi/mcp/base.py -- Abstract base class and shared utilities for MCP clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from koboi.types import MCPToolInfo, RiskLevel

if TYPE_CHECKING:
    from koboi.logger import AgentLogger
    from koboi.tools.registry import ToolRegistry


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
                "protocolVersion": "2024-11-05",
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


def register_mcp_tools(client: BaseMCPClient, registry: ToolRegistry) -> list[str]:
    """Bridge: discover MCP tools and register them to ToolRegistry.

    Each MCP tool is wrapped into an async closure that calls
    client.call_tool(). Works with any BaseMCPClient subclass.
    """
    tool_infos = client.discover_tools()
    registered: list[str] = []

    for info in tool_infos:

        def make_handler(tool_name: str, mcp_client: BaseMCPClient):
            async def handler(**kwargs) -> str:
                return await mcp_client.call_tool(tool_name, kwargs)

            return handler

        handler = make_handler(info.name, client)
        registry.register(
            name=info.name,
            description=info.description,
            parameters=info.input_schema,
            fn=handler,
            risk_level=RiskLevel.SAFE,
        )
        registered.append(info.name)

    return registered
