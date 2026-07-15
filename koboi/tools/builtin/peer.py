"""koboi/tools/builtin/peer.py -- call_peer_agent tool (cross-instance A2A).

Calls one or more peer koboi agents (same-org, cross-instance) and returns their
answers. Fan-out is parallel (``asyncio.gather``); each peer call is isolated so
one failure/timeout never aborts the others. Targets the peer's purpose-built
``POST /v1/peer/invoke`` receiver (sync JSON, ``AutonomousApprovalHandler`` so a
peer agent that wants a destructive tool is denied immediately instead of
hanging on a HITL approval that no human will resolve).
"""

from __future__ import annotations

import asyncio
import logging

from koboi.tools.registry import tool
from koboi.types import RiskLevel

_logger = logging.getLogger(__name__)


@tool(
    name="call_peer_agent",
    description=(
        "Call one or more peer koboi agents (same-org, cross-instance) and return "
        "their answers. Pass a list of {peer, message} to fan out in parallel. Each "
        "peer runs to completion on its own instance and returns its final answer."
    ),
    parameters={
        "type": "object",
        "properties": {
            "calls": {
                "type": "array",
                "description": "Peer calls to make in parallel.",
                "items": {
                    "type": "object",
                    "properties": {
                        "peer": {
                            "type": "string",
                            "description": "Registered peer name (from the peers config).",
                        },
                        "message": {
                            "type": "string",
                            "description": "The question/task for the peer agent.",
                        },
                    },
                    "required": ["peer", "message"],
                },
                "minItems": 1,
                "maxItems": 10,
            },
        },
        "required": ["calls"],
    },
    risk_level=RiskLevel.MODERATE,  # live side-effecting network call -> act+ or mode.read_only_tools
    deps=["peer_registry"],
    idempotent=False,  # must NOT silently double-fire into a peer on crash-resume
)
async def call_peer_agent(calls: list[dict], _deps: dict | None = None) -> str:
    registry = _deps.get("peer_registry") if _deps else None
    if registry is None:
        return "Error: A2A peers not configured. Cannot call peer agents."

    from koboi.server.peers import invoke_peer  # shared A2A HTTP path (also used by RemoteAgentProxy)

    # Each slot resolves its own peer + isolates failures/timeouts, so gather can
    # never raise -- a bad peer becomes an error string in its own slot only.
    async def _slot(call: dict) -> str:
        name = str(call.get("peer", ""))
        message = str(call.get("message", ""))
        peer = registry.get(name)
        if peer is None:
            return f"[{name}] (FAILED: unknown peer)\nAnswer: <error>"
        try:
            res = await asyncio.wait_for(invoke_peer(peer, message), timeout=peer.timeout)
            answer = getattr(res, "content", res)  # PeerInvokeResult or a str (test fakes)
            return f"[{peer.name}] (OK)\nAnswer: {answer}"
        except Exception as exc:  # noqa: BLE001 -- isolate: timeout/http/parse errors stay in this slot
            _logger.warning("A2A call to peer '%s' failed: %s", peer.name, exc)
            return f"[{peer.name}] (FAILED: {exc})\nAnswer: <error>"

    parts = await asyncio.gather(*[_slot(c) for c in calls])
    return "\n\n---\n\n".join(parts)
