---
name: creating-tools
description: Guide for creating new tools in the koboi tool system
---

# Creating Tools

## Pattern (decorator)
Use the `@tool()` decorator from `koboi.tools.registry`. Parameters is a JSON Schema dict.

## Template
```python
"""my_tools.py -- Custom tools."""
from koboi.tools.registry import tool
from koboi.types import RiskLevel


@tool(
    name="my_tool",
    description="What this tool does. Be specific.",
    parameters={
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "What this parameter does",
            },
        },
        "required": ["input"],
    },
    risk_level=RiskLevel.SAFE,  # SAFE | MODERATE | DESTRUCTIVE
)
async def my_tool(input: str) -> str:
    return f"Result: {input}"
```

## Registration (YAML)
```yaml
tools:
  custom:
    - module: "my_tools"
```

## Registration (programmatic)
```python
agent.add_tool("my_tool", my_tool_fn, "description", parameters_schema)
```

## Builtin tools for reference
- `calculator.py`: Simplest example (sync function, SAFE risk)
- `filesystem.py`: Multiple tools in one file (list_files, read_file, write_file, delete_file)
- `shell.py`: MODERATE risk example
- `web.py`: Async tool with httpx

## Important
- Tool functions can be sync or async (sync runs in thread via `asyncio.to_thread`)
- Return type must be `str` (the registry calls `str(result)`)
- Unknown parameters are stripped before execution (schema-driven)
- Set `timeout` on the `@tool` decorator for long-running tools
