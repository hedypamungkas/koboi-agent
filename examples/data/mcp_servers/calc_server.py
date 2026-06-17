"""Calc MCP Server -- example MCP server exposing calculator tools."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from koboi.mcp.server import MCPServer

server = MCPServer(name="calc-server", version="1.0.0")


@server.tool(
    name="add",
    description="Add two numbers",
    input_schema={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"},
        },
        "required": ["a", "b"],
    },
)
def add(a: float, b: float) -> str:
    return f"{a} + {b} = {a + b}"


@server.tool(
    name="multiply",
    description="Multiply two numbers",
    input_schema={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"},
        },
        "required": ["a", "b"],
    },
)
def multiply(a: float, b: float) -> str:
    return f"{a} x {b} = {a * b}"


@server.tool(
    name="factorial",
    description="Calculate factorial of a non-negative integer",
    input_schema={
        "type": "object",
        "properties": {
            "n": {"type": "integer", "description": "Non-negative integer"},
        },
        "required": ["n"],
    },
)
def factorial(n: int) -> str:
    if n < 0:
        return "Error: factorial is not defined for negative numbers"
    return f"{n}! = {math.factorial(n)}"


if __name__ == "__main__":
    server.run()
