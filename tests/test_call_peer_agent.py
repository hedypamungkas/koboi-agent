"""Unit tests for the call_peer_agent tool (cross-instance A2A).

Fan-out + error isolation are tested by monkeypatching ``_call_one`` so we control
each peer's answer/failure without real HTTP. The real HTTP path is exercised
end-to-end in ``test_a2a_integration.py``.
"""

from __future__ import annotations

import asyncio
import json

import koboi.tools.builtin.peer as peer_mod
from koboi.server.peers import PeerRegistry
from koboi.tools.builtin import register_all
from koboi.tools.registry import ToolRegistry


def _registry_with_peers(peers: list[dict]) -> ToolRegistry:
    r = ToolRegistry()
    register_all(r)
    pr = PeerRegistry()
    pr.load_from_config({"enabled": True, "allow_private_network": True, "peers": peers})
    r.set_dep("peer_registry", pr)
    return r


def _exec(r: ToolRegistry, calls: list[dict]) -> str:
    return asyncio.get_event_loop().run_until_complete(_aexec(r, calls))


async def _aexec(r: ToolRegistry, calls: list[dict]) -> str:
    return await r.execute("call_peer_agent", json.dumps({"calls": calls}))


class TestCallPeerAgent:
    async def test_no_registry_returns_clear_error(self):
        r = ToolRegistry()
        register_all(r)  # no set_dep("peer_registry")
        out = await r.execute("call_peer_agent", json.dumps({"calls": [{"peer": "C", "message": "hi"}]}))
        assert "not configured" in out.lower()

    async def test_unknown_peer(self, monkeypatch):
        r = _registry_with_peers([{"name": "C", "url": "http://localhost:8002", "token": "t"}])

        async def fake_call(peer, msg):  # never reached for unknown peer
            return "should-not-reach"

        monkeypatch.setattr(peer_mod, "_call_one", fake_call)
        out = await r.execute("call_peer_agent", json.dumps({"calls": [{"peer": "Z", "message": "hi"}]}))
        assert "unknown peer" in out.lower()

    async def test_single_call(self, monkeypatch):
        r = _registry_with_peers([{"name": "C", "url": "http://localhost:8002", "token": "t"}])

        async def fake_call(peer, msg):
            return f"answer-from-{peer.name}:{msg}"

        monkeypatch.setattr(peer_mod, "_call_one", fake_call)
        out = await r.execute("call_peer_agent", json.dumps({"calls": [{"peer": "C", "message": "hi"}]}))
        assert "(OK)" in out
        assert "answer-from-C:hi" in out

    async def test_parallel_fanout(self, monkeypatch):
        r = _registry_with_peers(
            [
                {"name": "B", "url": "http://localhost:8001", "token": "t"},
                {"name": "C", "url": "http://localhost:8002", "token": "t"},
            ]
        )

        async def fake_call(peer, msg):
            return f"{peer.name}-ans"

        monkeypatch.setattr(peer_mod, "_call_one", fake_call)
        out = await r.execute(
            "call_peer_agent",
            json.dumps({"calls": [{"peer": "B", "message": "m"}, {"peer": "C", "message": "m"}]}),
        )
        assert "B-ans" in out
        assert "C-ans" in out

    async def test_one_peer_error_others_complete(self, monkeypatch):
        """Q4: a failing peer must not abort the fan-out."""
        r = _registry_with_peers(
            [
                {"name": "B", "url": "http://localhost:8001", "token": "t"},
                {"name": "C", "url": "http://localhost:8002", "token": "t"},
            ]
        )

        async def fake_call(peer, msg):
            if peer.name == "C":
                raise RuntimeError("boom")
            return "B-ok"

        monkeypatch.setattr(peer_mod, "_call_one", fake_call)
        out = await r.execute(
            "call_peer_agent",
            json.dumps({"calls": [{"peer": "B", "message": "m"}, {"peer": "C", "message": "m"}]}),
        )
        assert "B-ok" in out  # B completed
        assert "FAILED" in out and "boom" in out  # C isolated to its own slot

    async def test_timeout_isolated(self, monkeypatch):
        """Q4: a slow peer is bounded by its timeout; the other still completes."""
        r = _registry_with_peers(
            [
                {"name": "B", "url": "http://localhost:8001", "token": "t", "timeout": 0.02},
                {"name": "C", "url": "http://localhost:8002", "token": "t"},
            ]
        )

        async def fake_call(peer, msg):
            if peer.name == "B":
                await asyncio.sleep(1)
            return f"{peer.name}-ok"

        monkeypatch.setattr(peer_mod, "_call_one", fake_call)
        out = await r.execute(
            "call_peer_agent",
            json.dumps({"calls": [{"peer": "B", "message": "m"}, {"peer": "C", "message": "m"}]}),
        )
        assert "C-ok" in out  # C completed
        assert "FAILED" in out  # B timed out
