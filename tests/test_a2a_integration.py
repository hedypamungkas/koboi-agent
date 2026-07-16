"""End-to-end A2A integration: instance X{A} calls instance Y{C} via call_peer_agent.

A's loop emits a call_peer_agent tool call; the tool POSTs Y's /v1/peer/invoke
receiver (routed in-process via an ASGI-transport httpx client); Y runs agent C
and returns its answer; A incorporates it. Validates the full fan-out path.
"""

from __future__ import annotations

import httpx
import pytest
from httpx import ASGITransport

from koboi.config import Config
from koboi.server.app import create_app
from tests.conftest import MockClient, make_mock_response, make_mock_tool_call


def _cfg(name, mode, peers_cfg):
    return Config.from_dict(
        {
            "agent": {"name": name, "mode": mode, "system_prompt": f"You are {name}.", "max_iterations": 5},
            "llm": {"provider": "openai", "model": "gpt-4o-mini", "api_key": "x"},
            "memory": {"backend": "memory"},
            "peers": peers_cfg,
        }
    )


@pytest.fixture
def app_y():
    """Instance Y: agent C, accepts peer token 'tok-y'."""
    cfg = _cfg("C", "chat", {"enabled": True, "inbound_tokens": ["tok-y"]})
    return create_app(cfg, client_factory=lambda: MockClient([make_mock_response(content="C-says-hello")]))


class TestA2AIntegration:
    async def test_a_calls_y_end_to_end(self, app_y, monkeypatch):
        # Route the tool's real httpx calls at app_y in-process (no real socket).
        real_async_client = httpx.AsyncClient

        class _Routed(real_async_client):
            def __init__(self, *a, **k):
                k.setdefault("transport", ASGITransport(app=app_y))
                super().__init__(*a, **k)

        monkeypatch.setattr(httpx, "AsyncClient", _Routed)

        # Instance X: agent A in act mode, peer C pointed at Y.
        cfg_x = _cfg(
            "A",
            "act",
            {
                "enabled": True,
                "allow_private_network": True,
                "peers": [{"name": "C", "url": "http://peer-y:8000", "token": "tok-y"}],
            },
        )

        # A: first emits call_peer_agent(C), then a final summary that includes C's answer.
        def factory():
            return MockClient(
                [
                    make_mock_response(
                        tool_calls=[
                            make_mock_tool_call("call_peer_agent", {"calls": [{"peer": "C", "message": "hi"}]})
                        ]
                    ),
                    make_mock_response(content="A got: C-says-hello"),
                ]
            )

        app_x = create_app(cfg_x, client_factory=factory)
        agent = await app_x.state.pool.get_or_create("a-session")
        result = await agent.run("Ask peer C for help")

        # A's final answer incorporates the answer it got back from C.
        assert "C-says-hello" in result.content
