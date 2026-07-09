# koboi/mcp/ -- Model Context Protocol client and server

## What this is
MCP integration. Consumes external MCP servers (stdio subprocess or Streamable HTTP) and
exposes their tools through the koboi `ToolRegistry`, so agent configs get third-party
tools with zero glue. Also ships an `MCPServer` to TURN any Python functions into an MCP
server over stdio. JSON-RPC 2.0 wire protocol; MCP spec protocolVersion `2024-11-05`.

## Key files
```
base.py        BaseMCPClient ABC, MCPError exception, register_mcp_tools() bridge, shared JSON-RPC helpers
client.py      MCPClient -- stdio transport (spawns server subprocess); re-exports MCPError, register_mcp_tools
http_client.py StreamableHTTPMCPClient -- HTTP transport (POST, JSON or SSE), Bearer auth, SSRF defense
server.py      MCPServer -- stdio JSON-RPC server; @server.tool() decorator + run() loop
__init__.py    Re-exports BaseMCPClient, MCPClient, StreamableHTTPMCPClient, MCPServer, MCPError, register_mcp_tools
```

## Extension API
### Add a custom transport (client side)
Subclass `BaseMCPClient` and implement the six abstract methods:
```python
class BaseMCPClient(ABC):                 # koboi/mcp/base.py
    def __init__(self, logger: AgentLogger | None = None): ...
    @abstractmethod
    def connect(self) -> dict: ...                 # sync -- initialize handshake, return server info
    @abstractmethod
    def discover_tools(self) -> list[MCPToolInfo]: # sync -- tools/list -> MCPToolInfo(name, description, input_schema)
        ...
    @abstractmethod
    async def call_tool(self, name: str, arguments: dict) -> str: ...   # MUST be async
    @abstractmethod
    def close(self) -> None: ...
    @abstractmethod
    def _send_request_impl(self, method: str, params: dict | None = None) -> dict: ...
    @abstractmethod
    def _send_notification_impl(self, method: str, params: dict | None = None) -> None: ...
```
Reuse the inherited helpers: `_do_initialize_handshake()`, `_make_request()`,
`_check_response()`, `_extract_tool_result()`. Then wire it into the facade's
`_create_mcp_client()` (facade.py) under a new `transport` value -- there is NO
decorator/registry for clients (unlike sandbox's ComponentRegistry).

### Build an MCP server (server.py)
```python
from koboi.mcp.server import MCPServer
server = MCPServer("my-server", version="1.0.0")

@server.tool("add", description="Add numbers", input_schema={"type": "object", "properties": {...}})
def add(a: int, b: int) -> int:
    return a + b

server.run()   # stdio JSON-RPC loop; see mcp_servers/todo_server.py for a full example
```

## How it's wired
`mcp:` YAML -> `_build_mcp()` (facade.py) -> reads `mcp.servers` -> `_create_mcp_client()`
picks the subclass by `transport` (`"stdio"` default, `"streamable-http"`) -> `client.connect()`
-> `register_mcp_tools(client, registry, group=...)` discovers and registers each tool.

## Conventions
- `connect()`/`discover_tools()` are SYNC (run at setup). `call_tool()` is ASYNC; sync
  transports offload I/O via `asyncio.to_thread()` so the event loop is never blocked.
- `tools/list` reads the `inputSchema` key (camelCase) from each server tool entry.
- `_extract_tool_result()` joins text content items, dedupes identical lines.

## Gotchas
- **All MCP tools are registered at `RiskLevel.SAFE`** (hardcoded in `register_mcp_tools`).
  They are never risk-gated or approval-gated; destructive MCP tools bypass the trust/grade
  system. Trust/PolicyHook cannot rein them in by tool name either.
- **HTTP auth is Bearer-only** (`auth.type: "bearer"`). No OAuth flow, no token refresh,
  no 401-retry; enterprise token expiry (~1h) is unsupported.
- **stdio runner allow-list** (basename match): `_MCP_DEFAULT_RUNNERS` =
  `{npx, uvx, python, python3, node, uv, deno, bun}`. Other runners raise `ValueError`;
  permit more via `mcp.allowlist_commands`.
- **HTTP client SSRF-blocks** private/internal/loopback URLs at `connect()` (reuses the web
  tool's `_check_url_ssrf`); raises `MCPError`.
- **A failed MCP server connect is a WARNING, not fatal** -- `register_mcp_tools` for that
  server is skipped and the agent continues. Check logs if a server's tools go missing.
- `agent.mode: chat/plan` blocks ALL custom tools by name, including MCP-bridged ones; use
  `mode: act`. See koboi/hooks/CLAUDE.md and the approval-before-ModeHook ordering caveat.
