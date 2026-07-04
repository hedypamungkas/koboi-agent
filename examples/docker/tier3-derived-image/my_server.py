"""Tier 3 — customize-by-code (Path B).

Derive FROM the koboi image, then run this entrypoint which composes the app with
``create_app(extra_tools=..., extra_routes=...)``. Demonstrates that a pre-built image
is a base layer: full programmatic control (custom tools, custom routes, custom
approval handler) without touching koboi internals.
"""

import uvicorn
from fastapi.responses import JSONResponse

from koboi.config import Config
from koboi.server import create_app


def company_crm_lookup(customer_id: str = "demo") -> str:
    """An extra tool wired in by code (not config)."""
    return f"crm:{customer_id}"


def _tier3_route(app, pool):
    """ExtraRouteRegistrar: add a custom endpoint that proves Path B composition.
    Signature is `Callable[[FastAPI, AgentPool], None]` (see koboi/server/app.py)."""

    @app.get("/__tier3__")
    async def _tier3() -> JSONResponse:
        return JSONResponse({"tier": 3, "extra_tool": "company_crm_lookup", "via": "create_app"})


cfg = Config.from_yaml("/app/agent.yaml")
app = create_app(
    cfg,
    extra_tools=[
        # Pool's extra_tools format: (name, fn, description, parameters[, risk_level])
        (
            "company_crm_lookup",
            company_crm_lookup,
            "Tier-3 proof: an extra tool wired in by code.",
            {"type": "object", "properties": {"customer_id": {"type": "string"}}, "required": []},
        )
    ],
    extra_routes=[_tier3_route],
    enable_cors=False,
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
