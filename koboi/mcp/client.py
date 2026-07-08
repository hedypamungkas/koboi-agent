"""koboi/mcp/client.py -- stdio-based MCP client (subprocess JSON-RPC transport).

Also re-exports MCPError and register_mcp_tools for backward compatibility.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading

from koboi.mcp.base import BaseMCPClient, MCPError, register_mcp_tools  # noqa: F401
from koboi.types import MCPToolInfo

# Re-export for backward compatibility
__all__ = ["MCPClient", "MCPError", "register_mcp_tools"]


class MCPClient(BaseMCPClient):
    """Client that spawns MCP server subprocess and communicates via JSON-RPC 2.0.

    Connection/discovery is synchronous (called during from_config setup).
    Runtime call_tool() is async, offloading subprocess I/O to a thread
    so the event loop is never blocked.
    """

    CONNECT_TIMEOUT = 15.0

    def __init__(self, server_command: list[str], logger=None, connect_timeout: float = 15.0):
        super().__init__(logger=logger)
        self.server_command = server_command
        self._process: subprocess.Popen | None = None
        self._stderr_thread: threading.Thread | None = None
        self._connect_timeout = connect_timeout

    # --- Public API ---

    def connect(self) -> dict:
        """Spawn server, send initialize handshake. Return server info."""
        self._process = subprocess.Popen(
            self.server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        # Drain stderr in background thread (so buffer doesn't fill up)
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            daemon=True,
        )
        self._stderr_thread.start()

        # Initialize handshake
        try:
            result = self._do_initialize_handshake(self.server_command)
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
        """Send tools/call via thread to avoid blocking the event loop."""
        return await asyncio.to_thread(self._call_tool_sync, name, arguments)

    def _call_tool_sync(self, name: str, arguments: dict) -> str:
        """Sync implementation of call_tool, run in a thread."""
        result = self._send_request(
            "tools/call",
            {
                "name": name,
                "arguments": arguments,
            },
        )
        return self._extract_tool_result(result)

    def close(self) -> None:
        """Terminate server subprocess."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()

    # --- Transport-specific I/O (implements abstract methods) ---

    def _send_request_impl(self, method: str, params: dict | None = None) -> dict:
        """Send JSON-RPC request via stdio, read response."""
        message = self._make_request(method, params)
        self._write(message)
        response = self._read()
        return self._check_response(response)

    def _send_notification_impl(self, method: str, params: dict | None = None) -> None:
        """Send JSON-RPC notification via stdio (no response expected)."""
        message = self._make_notification(method, params)
        self._write(message)

    # --- Backward-compatible aliases (used by existing tests) ---

    def _send_request(self, method: str, params: dict | None = None) -> dict:
        """Alias for _send_request_impl for backward compatibility."""
        return self._send_request_impl(method, params)

    def _send_notification(self, method: str, params: dict | None = None) -> None:
        """Alias for _send_notification_impl for backward compatibility."""
        self._send_notification_impl(method, params)

    # --- stdio transport details ---

    def _write(self, message: dict) -> None:
        if not self._process or not self._process.stdin:
            raise RuntimeError("MCP client not connected")

        if self.logger:
            self.logger.log_mcp_comm("send", message)

        line = json.dumps(message, ensure_ascii=False) + "\n"
        self._process.stdin.write(line.encode())
        self._process.stdin.flush()

    def _read(self) -> dict:
        if not self._process or not self._process.stdout:
            raise RuntimeError("MCP client not connected")

        line = self._read_line_with_timeout(self._process.stdout)
        if not line:
            raise RuntimeError("MCP server closed connection")

        message = json.loads(line.decode())

        if self.logger:
            self.logger.log_mcp_comm("recv", message)

        return message

    def _read_line_with_timeout(self, stream, timeout: float | None = None) -> bytes:
        """Read a single line from stream with a timeout.

        Uses a background thread because readline() is blocking and
        select.select() does not work on pipes on all platforms.
        """
        timeout = timeout or self._connect_timeout
        result: list[bytes] = []
        error: list[Exception] = []

        def _reader():
            try:
                result.append(stream.readline())
            except Exception as e:
                error.append(e)

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            self.close()
            raise TimeoutError(f"MCP server did not respond within {timeout}s. The server process has been terminated.")

        if error:
            raise error[0]

        return result[0] if result else b""

    def _drain_stderr(self) -> None:
        """Read stderr in background so buffer doesn't fill up."""
        if not self._process or not self._process.stderr:
            return
        try:
            for _line in self._process.stderr:
                pass
        except (ValueError, OSError):
            pass
