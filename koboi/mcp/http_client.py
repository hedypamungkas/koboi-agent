"""koboi/mcp/http_client.py -- Streamable HTTP transport for MCP (spec 2025-03-26).

Connects to remote MCP servers over HTTP using JSON-RPC 2.0.
Supports both JSON and SSE response formats per the MCP spec.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import TYPE_CHECKING

import httpx

from koboi.mcp.base import BaseMCPClient, MCPError
from koboi.types import MCPToolInfo

if TYPE_CHECKING:
    from koboi.logger import AgentLogger


class StreamableHTTPMCPClient(BaseMCPClient):
    """MCP client that communicates with a remote server via Streamable HTTP.

    Uses the MCP Streamable HTTP transport (spec 2025-03-26):
    - All messages sent as HTTP POST to a single endpoint
    - Responses may be JSON or SSE (Server-Sent Events)
    - Session managed via Mcp-Session-Id header
    - Auth via Bearer token (configurable)
    """

    TRANSPORT = "streamable-http"

    def is_connected(self) -> bool:
        """Live iff the httpx client has been created (not closed)."""
        return self._client is not None

    @property
    def endpoint(self) -> str:
        return self._url

    def __init__(
        self,
        url: str,
        logger: AgentLogger | None = None,
        auth_config: dict | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ):
        super().__init__(logger=logger)
        self._url = url.rstrip("/")
        self._auth_config = auth_config or {}
        self._extra_headers = headers or {}
        self._timeout = timeout
        self._session_id: str | None = None
        self._client: httpx.Client | None = None
        self._lock = threading.Lock()
        # G1: build an AuthStrategy (None | BearerAuth | OAuthClientCredentialsAuth).
        # Lazily imported to avoid importing the auth module for the common no-auth case.
        from koboi.mcp.auth import build_mcp_auth

        self._auth = build_mcp_auth(self._auth_config)

    # --- Public API ---

    def connect(self) -> dict:
        """Create HTTP client, send initialize handshake, return server info."""
        # M4: SSRF defense -- reject private/internal/loopback MCP URLs before any
        # egress. Reuses the web tool's checker (single source of PRIVATE_NETWORKS);
        # lazy import keeps the module cheap and avoids a facade/tools cycle.
        from koboi.tools.builtin.web import _check_url_ssrf

        try:
            _check_url_ssrf(self._url)
        except (ValueError, OSError) as e:
            raise MCPError(code=-1, message=f"SSRF-blocked MCP URL: {e}") from e
        self._client = httpx.Client(timeout=self._timeout)
        try:
            result = self._do_initialize_handshake(self._url)
        except Exception:
            self.close()
            raise
        return result

    def discover_tools(self) -> list[MCPToolInfo]:
        """Send tools/list, return list of available tools."""
        result = self._send_request("tools/list")
        self._tools = [
            MCPToolInfo(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
            )
            for t in result.get("tools", [])
        ]

        if self.logger:
            self.logger.log_mcp_discovery(self._tools)

        return self._tools

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Call a tool via HTTP POST. Async via thread offload."""
        return await asyncio.to_thread(self._call_tool_sync, name, arguments)

    def ensure_connected(self) -> None:
        """Re-establish the HTTP session if it was closed (G4). Single reconnect attempt."""
        if self._client is None:
            self.connect()

    def _call_tool_sync(self, name: str, arguments: dict) -> str:
        """Sync implementation of call_tool."""
        self.ensure_connected()
        result = self._send_request(
            "tools/call",
            {
                "name": name,
                "arguments": arguments,
            },
        )
        return self._extract_tool_result(result)

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()
            self._client = None

    # --- Transport-specific I/O (implements abstract methods) ---

    def _send_request_impl(self, method: str, params: dict | None = None) -> dict:
        """Send JSON-RPC request via HTTP POST, return parsed result."""
        message = self._make_request(method, params)
        response_data = self._post_json_rpc(message)
        return self._check_response(response_data)

    def _send_notification_impl(self, method: str, params: dict | None = None) -> None:
        """Send JSON-RPC notification via HTTP POST (no response expected)."""
        message = self._make_notification(method, params)
        self._post_json_rpc(message, is_notification=True)

    # --- Backward-compatible aliases ---

    def _send_request(self, method: str, params: dict | None = None) -> dict:
        """Alias for _send_request_impl for backward compatibility."""
        return self._send_request_impl(method, params)

    def _send_notification(self, method: str, params: dict | None = None) -> None:
        """Alias for _send_notification_impl for backward compatibility."""
        self._send_notification_impl(method, params)

    # --- HTTP transport details ---

    def _post_safe(self, message: dict, headers: dict[str, str]) -> httpx.Response:
        """POST with connect/timeout -> MCPError translation (shared by retry path)."""
        try:
            return self._client.post(self._url, json=message, headers=headers)  # type: ignore[union-attr]
        except httpx.ConnectError as e:
            raise MCPError(code=-1, message=f"Connection failed: {e}") from e
        except httpx.TimeoutException as e:
            raise MCPError(code=-1, message=f"Request timed out: {e}") from e

    def _post_json_rpc(self, message: dict, is_notification: bool = False) -> dict:
        """POST a JSON-RPC message to the MCP endpoint.

        Handles both JSON and SSE response formats.
        For notifications (no id), returns empty dict on 202.
        """
        if not self._client:
            raise RuntimeError("MCP HTTP client not connected")

        headers = self._build_headers()

        if self.logger:
            self.logger.log_mcp_comm("send", message)

        response = self._post_safe(message, headers)

        # Capture session ID from response
        new_session_id = response.headers.get("mcp-session-id")
        if new_session_id:
            self._session_id = new_session_id

        # Notifications get 202 Accepted with no body
        if is_notification:
            return {}

        # G1: on 401 with a refresh-capable auth (OAuth), refresh once and retry.
        if response.status_code == 401 and self._auth is not None and getattr(self._auth, "supports_refresh", False):
            self._auth.refresh(force=True)  # type: ignore[attr-defined]
            response = self._post_safe(message, self._build_headers())
            new_session_id = response.headers.get("mcp-session-id")
            if new_session_id:
                self._session_id = new_session_id

        # Handle HTTP errors
        if response.status_code >= 400:
            detail = self._extract_error_detail(response)
            raise MCPError(
                code=-32603,
                message=f"HTTP {response.status_code}: {detail}",
            )

        # Parse response based on content type
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            result = self._parse_sse_response(response)
        else:
            result = response.json()

        if self.logger:
            self.logger.log_mcp_comm("recv", result)

        return result

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for MCP requests."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        # G1: auth via AuthStrategy (None | BearerAuth | OAuthClientCredentialsAuth).
        if self._auth is not None:
            self._auth.apply(headers)

        headers.update(self._extra_headers)
        return headers

    @staticmethod
    def _parse_sse_response(response: httpx.Response) -> dict:
        """Parse SSE stream, return first JSON-RPC response found.

        SSE format:
            event: message
            data: {"jsonrpc":"2.0","id":1,"result":{...}}
        """
        for line in response.text.split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                payload = line[6:]
                if payload:
                    try:
                        data = json.loads(payload)
                        if "jsonrpc" in data and ("result" in data or "error" in data):
                            return data
                    except json.JSONDecodeError:
                        continue
        raise MCPError(code=-1, message="No JSON-RPC response found in SSE stream")

    @staticmethod
    def _extract_error_detail(response: httpx.Response) -> str:
        """Extract error detail from HTTP error response."""
        try:
            data = response.json()
            error = data.get("error", data)
            if isinstance(error, dict):
                return error.get("message", str(data))
            return str(error)
        except (json.JSONDecodeError, ValueError):
            return response.text[:500]
