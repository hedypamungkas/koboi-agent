"""
Base class for MCP Server.

Handles JSON-RPC 2.0 lifecycle via stdio:
- initialize / initialized handshake
- tools/list -- return list of registered tools
- tools/call -- execute requested tool

Subclass just needs to define tools with @server.tool(), then call server.run().
"""

from __future__ import annotations

import json
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class _MCPToolDef:
    name: str
    description: str
    inputSchema: dict
    handler: Callable


@dataclass
class _MCPResourceDef:
    uri: str
    name: str
    description: str
    mimeType: str
    handler: Callable  # () -> str (returns the resource text)


@dataclass
class _MCPPromptDef:
    name: str
    description: str
    arguments: list
    handler: Callable  # (arguments: dict) -> str (returns the rendered prompt text)


class MCPServer:
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self._tools: dict[str, _MCPToolDef] = {}
        self._resources: dict[str, _MCPResourceDef] = {}
        self._prompts: dict[str, _MCPPromptDef] = {}
        self._protocol_version = "2025-03-26"

    # --- Tool registration (similar to existing @tool decorator) ---

    def tool(self, name: str, description: str, input_schema: dict):
        """Decorator to register function as MCP tool."""

        def decorator(fn: Callable) -> Callable:
            self._tools[name] = _MCPToolDef(
                name=name,
                description=description,
                inputSchema=input_schema,
                handler=fn,
            )
            return fn

        return decorator

    # --- Resource / prompt registration (G2) ---

    def resource(self, uri: str, name: str = "", description: str = "", mime_type: str = "text/plain"):
        """Decorator to register a function returning a resource's text content."""

        def decorator(fn: Callable) -> Callable:
            self._resources[uri] = _MCPResourceDef(
                uri=uri,
                name=name or uri,
                description=description,
                mimeType=mime_type,
                handler=fn,
            )
            return fn

        return decorator

    def prompt(self, name: str, description: str = "", arguments: list | None = None):
        """Decorator to register a function rendering a prompt (args dict -> str)."""

        def decorator(fn: Callable) -> Callable:
            self._prompts[name] = _MCPPromptDef(
                name=name,
                description=description,
                arguments=arguments or [],
                handler=fn,
            )
            return fn

        return decorator

    # --- Protocol lifecycle ---

    def run(self) -> None:
        """Main loop -- read stdin, dispatch, write stdout."""
        self._log_stderr(f"MCP Server '{self.name}' v{self.version} waiting for messages...")

        while True:
            line = sys.stdin.readline()
            if not line:
                self._log_stderr("Client disconnected (EOF)")
                break

            line = line.strip()
            if not line:
                continue

            try:
                message = json.loads(line)
            except json.JSONDecodeError as e:
                self._log_stderr(f"Invalid JSON: {e}")
                self._write_response(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": f"Parse error: {e}"},
                    }
                )
                continue

            self._log_stderr(f"<-- {message.get('method', '?')} (id={message.get('id', '-')})")
            self._dispatch(message)

    # --- Method handlers ---

    def _dispatch(self, message: dict) -> None:
        method = message.get("method", "")
        id_ = message.get("id")
        params = message.get("params", {})

        # Notification (no id) -- no response needed
        if id_ is None and method == "initialized":
            self._log_stderr("    initialized notification received")
            return

        handlers = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resources_read,
            "prompts/list": self._handle_prompts_list,
            "prompts/get": self._handle_prompts_get,
        }

        handler = handlers.get(method)
        if handler is None:
            self._write_response(
                {
                    "jsonrpc": "2.0",
                    "id": id_,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }
            )
            return

        try:
            result = handler(id_, params)
            self._write_response(
                {
                    "jsonrpc": "2.0",
                    "id": id_,
                    "result": result,
                }
            )
        except Exception as e:
            self._log_stderr(f"    ERROR: {e}\n{traceback.format_exc()}")
            self._write_response(
                {
                    "jsonrpc": "2.0",
                    "id": id_,
                    "result": {
                        "content": [{"type": "text", "text": f"Error: {e}"}],
                        "isError": True,
                    },
                }
            )

    def _handle_initialize(self, id_, params: dict) -> dict:
        return {
            "protocolVersion": self._protocol_version,
            "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
            "serverInfo": {"name": self.name, "version": self.version},
        }

    def _handle_tools_list(self, id_, params: dict) -> dict:
        tools = [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.inputSchema,
            }
            for t in self._tools.values()
        ]
        return {"tools": tools}

    def _handle_tools_call(self, id_, params: dict) -> dict:
        name = params.get("name", "")
        arguments = params.get("arguments", {})

        if name not in self._tools:
            return {
                "content": [{"type": "text", "text": f"Error: unknown tool '{name}'"}],
                "isError": True,
            }

        tool_def = self._tools[name]
        result = tool_def.handler(**arguments)
        return {
            "content": [{"type": "text", "text": str(result)}],
        }

    # --- resources / prompts handlers (G2) ---

    def _handle_resources_list(self, id_, params: dict) -> dict:
        resources = [
            {
                "uri": r.uri,
                "name": r.name,
                "description": r.description,
                "mimeType": r.mimeType,
            }
            for r in self._resources.values()
        ]
        return {"resources": resources}

    def _handle_resources_read(self, id_, params: dict) -> dict:
        uri = params.get("uri", "")
        if uri not in self._resources:
            return {"contents": []}
        rdef = self._resources[uri]
        return {"contents": [{"uri": uri, "mimeType": rdef.mimeType, "text": str(rdef.handler())}]}

    def _handle_prompts_list(self, id_, params: dict) -> dict:
        prompts = [
            {
                "name": p.name,
                "description": p.description,
                "arguments": p.arguments,
            }
            for p in self._prompts.values()
        ]
        return {"prompts": prompts}

    def _handle_prompts_get(self, id_, params: dict) -> dict:
        name = params.get("name", "")
        arguments = params.get("arguments", {})
        if name not in self._prompts:
            return {"description": "", "messages": []}
        pdef = self._prompts[name]
        text = str(pdef.handler(arguments))
        return {
            "description": pdef.description,
            "messages": [{"role": "user", "content": {"type": "text", "text": text}}],
        }

    # --- Internal ---

    def _write_response(self, message: dict) -> None:
        line = json.dumps(message, ensure_ascii=False)
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
        self._log_stderr(f"--> response (id={message.get('id', '-')})")

    def _log_stderr(self, msg: str) -> None:
        sys.stderr.write(f"[{self.name}] {msg}\n")
        sys.stderr.flush()
