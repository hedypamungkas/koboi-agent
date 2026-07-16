"""Tests for RemoteAgentProxy + AgentDef.endpoint (P2: remote orchestration nodes).

A config-declared node with ``endpoint: <peer>`` becomes a RemoteAgentProxy that
POSTs the peer's /v1/peer/invoke. Verified at three levels: AgentDef round-trip,
the factory branch, and a real Orchestrator run using a remote node.
"""

from __future__ import annotations

import pytest

import koboi.server.peers as peers_mod
from koboi.loop import AgentCore
from koboi.orchestration.factory import AgentFactory
from koboi.orchestration.orchestrator import Orchestrator
from koboi.orchestration.remote_proxy import RemoteAgentProxy
from koboi.orchestration.router import KeywordRouter
from koboi.server.peers import PeerConfig, PeerInvokeResult, PeerRegistry
from koboi.types import AgentDef, RunResult
from tests.conftest import MockClient, make_mock_response


def _registry() -> PeerRegistry:
    reg = PeerRegistry()
    reg._peers["peerY"] = PeerConfig(name="peerY", url="http://localhost:8002", token="tok-y")
    return reg


class TestAgentDefEndpoint:
    def test_endpoint_round_trip(self):
        ad = AgentDef.from_dict({"name": "review", "endpoint": "peerY"})
        assert ad.endpoint == "peerY"
        assert ad.to_dict().get("endpoint") == "peerY"

    def test_no_endpoint_defaults_none_and_omitted(self):
        ad = AgentDef.from_dict({"name": "local"})
        assert ad.endpoint is None
        assert "endpoint" not in ad.to_dict()


class TestFactoryBranch:
    def test_endpoint_node_returns_proxy(self):
        ad = AgentDef(name="review", endpoint="peerY")
        proxy = AgentFactory.create_configured_agent(ad, client=None, peer_registry=_registry())
        assert isinstance(proxy, RemoteAgentProxy)
        assert proxy.name == "review"
        assert proxy.peer_name == "peerY"

    def test_endpoint_without_registry_raises(self):
        ad = AgentDef(name="review", endpoint="peerY")
        with pytest.raises(ValueError):
            AgentFactory.create_configured_agent(ad, client=None, peer_registry=None)

    def test_local_node_still_builds_core(self):
        # No endpoint -> the normal local-AgentCore path is unchanged.
        ad = AgentDef(name="local")
        core = AgentFactory.create_configured_agent(ad, client=MockClient([]), peer_registry=_registry())
        assert isinstance(core, AgentCore)


class TestRemoteAgentProxyRun:
    async def test_success(self, monkeypatch):
        async def fake(peer, msg):
            return PeerInvokeResult(content="REMOTE-ANSWER", receiver_trace_id="peer-T")

        monkeypatch.setattr(peers_mod, "invoke_peer", fake)
        proxy = RemoteAgentProxy("review", "peerY", _registry())
        res = await proxy.run("hi")
        assert isinstance(res, RunResult)
        assert res.content == "REMOTE-ANSWER"
        assert res.success is True
        assert res.metadata.get("peer_trace_id") == "peer-T"

    async def test_unknown_peer_returns_error_runresult(self):
        proxy = RemoteAgentProxy("review", "NOPE", _registry())
        res = await proxy.run("hi")
        assert res.content.startswith("Error:")
        assert res.success is False

    async def test_peer_failure_returns_error_runresult(self, monkeypatch):
        async def boom(peer, msg):
            raise RuntimeError("timeout")

        monkeypatch.setattr(peers_mod, "invoke_peer", boom)
        proxy = RemoteAgentProxy("review", "peerY", _registry())
        res = await proxy.run("hi")
        assert res.content.startswith("Error:")
        assert "timeout" in res.content
        assert res.success is False


class TestRemoteNodeInOrchestration:
    async def test_remote_node_participates_in_sequential(self, monkeypatch):
        """A RemoteAgentProxy is a usable orchestration node: routing + run() + answer flow."""

        async def fake(peer, msg):
            return PeerInvokeResult(content="REMOTE-ANSWER-42")

        monkeypatch.setattr(peers_mod, "invoke_peer", fake)
        proxy = RemoteAgentProxy("review", "peerY", _registry())
        router = KeywordRouter(agent_defs=[AgentDef(name="review", keywords=["review"])])
        client = MockClient([make_mock_response(content="synthesized")])
        orch = Orchestrator(client=client, router=router, agents_map={"review": proxy})
        result = await orch.run("please review this", mode="sequential")
        assert any("REMOTE-ANSWER-42" in r.answer for r in result.agent_results)
