"""Example: cross-instance Agent-to-Agent (A2A) fan-out, fully in-process.

Instance X (agent A) calls peer agent C on instance Y via the ``call_peer_agent``
tool. This script wires both instances' FastAPI apps in-process and routes A's
tool HTTP call at Y via an ASGI transport -- so it runs offline (no API key, no
real sockets) yet exercises the real transport + inbound receiver + tool.

To run with REAL LLMs instead, serve the two configs on separate ports and drop
the stub clients::

    pip install -e ".[api]"
    koboi serve configs/a2a_instance_y.yaml &      # agent C, port 8002
    koboi serve configs/a2a_instance_x.yaml         # agent A, port 8000 (peers -> C)

then prompt A to "ask C ...".
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

# Run from a source checkout without installing: prefer the local koboi package
# over any editable install that may point at a different checkout.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from httpx import ASGITransport

from koboi.config import Config
from koboi.llm.base import LLMClient
from koboi.server.app import create_app
from koboi.types import AgentResponse, ToolCall, TokenUsage


class StubClient(LLMClient):
    """Tiny offline LLM stub: replays a scripted list of responses."""

    def __init__(self, responses: list[AgentResponse]):
        self._responses = list(responses)
        self._i = 0

    @property
    def model(self) -> str:
        return "stub"

    @model.setter
    def model(self, v: str) -> None:
        pass

    async def complete(self, messages, tools=None, response_format=None) -> AgentResponse:
        if self._i < len(self._responses):
            r = self._responses[self._i]
            self._i += 1
            return r
        return AgentResponse(content="(no further scripted response)", tool_calls=[], usage=TokenUsage(0, 0))

    async def complete_stream(self, messages, tools=None, response_format=None):
        r = await self.complete(messages, tools, response_format)
        from koboi.events import CompleteEvent

        yield CompleteEvent(response=r, content=r.content or "")

    async def get_embeddings(self, text: str):
        return None

    async def close(self):
        pass


def _resp(content=None, tool_calls=None) -> AgentResponse:
    return AgentResponse(content=content, tool_calls=tool_calls or [], usage=TokenUsage(1, 1))


def _tc(name: str, args: dict) -> ToolCall:
    return ToolCall(id=f"tc_{name}", name=name, arguments=json.dumps(args))


def _config(name: str, mode: str, peers: dict) -> Config:
    return Config.from_dict(
        {
            "agent": {"name": name, "mode": mode, "system_prompt": f"You are {name}.", "max_iterations": 5},
            "llm": {"provider": "openai", "model": "stub", "api_key": "stub"},
            "memory": {"backend": "memory"},
            "peers": peers,
        }
    )


async def main() -> None:
    # --- Instance Y: agent C answers peer calls (Bearer tok-y). ---
    app_y = create_app(
        _config("C", "chat", {"enabled": True, "inbound_tokens": ["tok-y"]}),
        client_factory=lambda: StubClient([_resp("C reports: all systems green")]),
    )

    # Route the call_peer_agent tool's httpx at Y in-process.
    real = httpx.AsyncClient

    class Routed(real):
        def __init__(self, *a, **k):
            k.setdefault("transport", ASGITransport(app=app_y))
            super().__init__(*a, **k)

    httpx.AsyncClient = Routed  # demo-only global swap so the tool's httpx hits Y

    # --- Instance X: agent A fans out to C, then summarizes. ---
    app_x = create_app(
        _config(
            "A",
            "act",
            {
                "enabled": True,
                "allow_private_network": True,
                "peers": [{"name": "C", "url": "http://peer-y:8000", "token": "tok-y", "agent_name": "C"}],
            },
        ),
        client_factory=lambda: StubClient(
            [
                _resp(tool_calls=[_tc("call_peer_agent", {"calls": [{"peer": "C", "message": "status?"}]})]),
                _resp("A summary: C said -> all systems green"),
            ]
        ),
    )

    agent = await app_x.state.pool.get_or_create("demo-a")
    result = await agent.run("Check with C and summarize.")
    httpx.AsyncClient = real  # restore

    print("=== A's final answer ===")
    print(result.content)


if __name__ == "__main__":
    asyncio.run(main())
