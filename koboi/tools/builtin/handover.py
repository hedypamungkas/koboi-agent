"""koboi/tools/builtin/handover.py -- transfer_to_human tool (Wave 2 B1).

The LLM-facing surface for human handover. The tool simply raises
``AgentHandoverError`` -- it does NOT pause on a Future (that would hold
``pool.session_lock`` and deadlock the human's next ``/chat/stream``). The
exception propagates out of the run; the server converts it to a typed
``HandoverEvent`` (interactive SSE) or an ``awaiting_human`` terminal status
(jobs), releasing the session lock so a human operator can take over.
"""

from __future__ import annotations

from koboi.exceptions import AgentHandoverError
from koboi.tools.registry import tool
from koboi.types import RiskLevel


@tool(
    name="transfer_to_human",
    group="handover",
    description=(
        "Yield control of this conversation to a human operator. Use when the "
        "request needs human judgment, a policy exception, sensitive handling, or "
        "empathy beyond your scope -- for example a complex complaint, a refund "
        "outside policy, or an upset customer. The conversation history and your "
        "summary are passed to the operator."
    ),
    parameters={
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Why handover is needed (what is beyond your scope / needs a human).",
            },
            "summary": {
                "type": "string",
                "description": "Concise case summary for the human operator (what the customer wants, what was tried).",
            },
        },
        "required": ["reason"],
    },
    risk_level=RiskLevel.SAFE,
)
async def transfer_to_human(reason: str, summary: str = "") -> str:
    raise AgentHandoverError(reason, summary)
