"""Tier 2 extension: a custom tool loaded via `tools.custom` + KOBOI_EXTENSIONS_DIR.

Mounted into the container at /app/ext; `koboi/_extensions_path.py` puts that dir on
`sys.path` so the facade's `importlib.import_module("my_ext")` (tools.custom) finds it.
"""

from koboi.tools.registry import tool


@tool(
    name="ext_greeting",
    description="Tier-2 proof: a custom tool from the mounted extensions dir.",
    parameters={
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": [],
    },
)
def ext_greeting(name: str = "world") -> str:
    return f"hello {name} from the extensions dir"
