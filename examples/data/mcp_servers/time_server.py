"""Time MCP Server -- example MCP server exposing time tools."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from koboi.mcp.server import MCPServer

server = MCPServer(name="time-server", version="1.0.0")


@server.tool(
    name="get_current_time",
    description="Get the current time in HH:MM:SS format",
    input_schema={"type": "object", "properties": {}},
)
def get_current_time() -> str:
    from datetime import datetime

    return f"Current time: {datetime.now().strftime('%H:%M:%S')}"


@server.tool(
    name="get_date",
    description="Get today's date in YYYY-MM-DD format",
    input_schema={"type": "object", "properties": {}},
)
def get_date() -> str:
    from datetime import datetime

    return f"Today's date: {datetime.now().strftime('%Y-%m-%d')}"


@server.tool(
    name="calculate_days_until",
    description="Calculate the number of days until a target date",
    input_schema={
        "type": "object",
        "properties": {
            "target_date": {
                "type": "string",
                "description": "Target date in YYYY-MM-DD format",
            },
        },
        "required": ["target_date"],
    },
)
def calculate_days_until(target_date: str) -> str:
    from datetime import datetime

    try:
        target = datetime.strptime(target_date, "%Y-%m-%d").date()
        today = datetime.now().date()
        diff = (target - today).days
        if diff > 0:
            return f"{diff} days until {target_date}"
        elif diff == 0:
            return f"{target_date} is today!"
        return f"{abs(diff)} days since {target_date}"
    except ValueError:
        return f"Error: Invalid date format '{target_date}'. Use YYYY-MM-DD."


if __name__ == "__main__":
    server.run()
