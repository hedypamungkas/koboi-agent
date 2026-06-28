"""examples/server_customize.py -- extend the server by code (Path B).

Demonstrates ``create_app(..., extra_tools=..., extra_hooks=...)``: the same
runtime as Path A, but each pooled agent also gets the custom tool/hook.

Requires the api extra::

    pip install koboi-agent[api]
    python examples/server_customize.py
"""

import uvicorn

from koboi.config import Config
from koboi.server import create_app


def weather(city: str) -> str:
    """A custom tool registered via the code path."""
    return f"Weather in {city}: sunny, 25C"


async def log_tool_calls(ctx):
    """A custom hook (POST_TOOL_USE would be declared via Hook ABC in real use)."""
    print(f"[audit] tool used: {getattr(ctx, 'tool_name', '?')}")
    return ctx


if __name__ == "__main__":
    config = Config.from_yaml("configs/server_simple.yaml")
    app = create_app(
        config,
        extra_tools=(
            (
                "weather",
                weather,
                "Get the weather for a city",
                {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            ),
        ),
        extra_hooks=((log_tool_calls, None),),
    )
    uvicorn.run(app, host="127.0.0.1", port=8000)
