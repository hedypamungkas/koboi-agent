"""Todo MCP Server — example MCP server for todo management."""
from koboi.mcp.server import MCPServer

server = MCPServer(name="todo-server", version="1.0.0")

_todos: dict[int, dict] = {}
_next_id: int = 1


@server.tool(
    name="add_todo",
    description="Add a new todo item",
    input_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Todo title"},
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "Priority (default: medium)",
            },
        },
        "required": ["title"],
    },
)
def add_todo(title: str, priority: str = "medium") -> str:
    global _next_id
    _todos[_next_id] = {"id": _next_id, "title": title, "priority": priority, "completed": False}
    result = f"Todo added: '{title}' (id={_next_id}, priority={priority})"
    _next_id += 1
    return result


@server.tool(
    name="list_todos",
    description="List all todo items",
    input_schema={"type": "object", "properties": {}},
)
def list_todos() -> str:
    if not _todos:
        return "No todos."
    lines = ["Todos:"]
    for t in _todos.values():
        status = "x" if t["completed"] else "o"
        lines.append(f"  [{status}] #{t['id']} {t['title']} ({t['priority']})")
    return "\n".join(lines)


@server.tool(
    name="complete_todo",
    description="Mark a todo as completed",
    input_schema={
        "type": "object",
        "properties": {"id": {"type": "integer", "description": "Todo ID"}},
        "required": ["id"],
    },
)
def complete_todo(id: int) -> str:
    if id not in _todos:
        return f"Error: todo id={id} not found"
    _todos[id]["completed"] = True
    return f"Todo '{_todos[id]['title']}' completed."


@server.tool(
    name="delete_todo",
    description="Delete a todo",
    input_schema={
        "type": "object",
        "properties": {"id": {"type": "integer", "description": "Todo ID"}},
        "required": ["id"],
    },
)
def delete_todo(id: int) -> str:
    if id not in _todos:
        return f"Error: todo id={id} not found"
    title = _todos.pop(id)["title"]
    return f"Todo '{title}' deleted."


if __name__ == "__main__":
    server.run()
