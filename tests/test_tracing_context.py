"""Tests for P4 cross-instance W3C traceparent propagation."""

from __future__ import annotations

import asyncio
import sqlite3

import httpx
import pytest

from koboi import tracing_context as tc
from koboi.journal import StepJournal
from koboi.memory_sqlite import ensure_steps_table
from koboi.server.peers import PeerConfig, PeerInvokeResult, invoke_peer

_GOOD = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"


@pytest.fixture(autouse=True)
def _isolate_trace_context():
    """Reset the process-global trace ContextVar around every test (no leakage)."""
    tc.set_context(None)
    yield
    tc.set_context(None)


class TestTracingContext:
    def test_parse_good(self):
        p = tc.parse_traceparent(_GOOD)
        assert p is not None
        assert p.trace_id == "0af7651916cd43dd8448eb211c80319c" and p.flags == "01"

    def test_parse_rejects_bad(self):
        assert tc.parse_traceparent("junk") is None
        assert tc.parse_traceparent(None) is None
        assert tc.parse_traceparent("") is None
        # wrong-length fields
        assert tc.parse_traceparent("00-deadbeef-b7ad6b7169203331-01") is None

    def test_parse_rejects_all_zero(self):
        # W3C: all-zero trace-id / parent-id are invalid.
        assert tc.parse_traceparent("00-" + "0" * 32 + "-b7ad6b7169203331-01") is None
        assert tc.parse_traceparent("00-0af7651916cd43dd8448eb211c80319c-" + "0" * 16 + "-01") is None

    def test_mint_root_is_valid_w3c(self):
        a = tc.mint_root()
        b = tc.mint_root()
        assert a.trace_id != b.trace_id  # unique
        assert tc.parse_traceparent(a.as_traceparent()) is not None  # round-trips as valid W3C

    def test_child_keeps_trace_id_new_parent(self):
        root = tc.mint_root()
        c = tc.child(root)
        assert c.trace_id == root.trace_id  # same trace
        assert c.parent_id != root.parent_id  # fresh parent (proper W3C hop)

    def test_begin_request_honors_inbound_then_mints(self):
        tc.begin_request(_GOOD)
        assert tc.current_trace_id() == "0af7651916cd43dd8448eb211c80319c"
        tc.begin_request(None)
        assert tc.current_trace_id() != "0af7651916cd43dd8448eb211c80319c"  # minted a fresh root

    async def test_fanout_shares_trace_id(self):
        """asyncio.gather copies the ContextVar -> a parallel fan-out shares the trace-id."""
        tc.begin_request(_GOOD)
        root = tc.current_trace_id()

        async def read():
            await asyncio.sleep(0)
            return tc.current_trace_id()

        a, b = await asyncio.gather(read(), read())
        assert a == root and b == root


# --- invoke_peer outbound header ---


class _Resp:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"content": "ANS", "trace_id": "peerT"}


class _Client:
    def __init__(self):
        self.posted = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, headers=None):
        self.posted = {"url": url, "headers": headers}
        return _Resp()


class TestInvokePeerTrace:
    async def test_sends_child_traceparent_when_context_set(self, monkeypatch):
        client = _Client()
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client)
        tc.begin_request(_GOOD)
        root_parent = tc.current().parent_id

        res = await invoke_peer(PeerConfig(name="C", url="http://localhost:8002", token="t"), "hi")

        assert isinstance(res, PeerInvokeResult)
        assert res.content == "ANS" and res.trace_id == "peerT"
        sent = tc.parse_traceparent(client.posted["headers"]["traceparent"])
        assert sent is not None
        assert sent.trace_id == tc.current_trace_id()  # same trace
        assert sent.parent_id != root_parent  # fresh parent-id for the hop

    async def test_no_traceparent_header_when_no_context(self, monkeypatch):
        client = _Client()
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client)
        # No begin_request -> contextvar is None in this fresh test context.
        await invoke_peer(PeerConfig(name="C", url="http://localhost:8002", token="t"), "hi")
        assert "traceparent" not in client.posted["headers"]


class TestJournalTraceId:
    def test_record_step_stamps_trace_id(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "m.db"))
        ensure_steps_table(conn)
        journal = StepJournal(conn, "s1")
        tc.begin_request(_GOOD)
        journal.record_step(turn_index=1, step_index=0, status="running")

        row = conn.execute("SELECT trace_id FROM steps WHERE session_id = ?", ("s1",)).fetchone()
        assert row[0] == tc.current_trace_id()

    def test_record_step_trace_id_null_without_context(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "m.db"))
        ensure_steps_table(conn)
        journal = StepJournal(conn, "s1")
        # No begin_request -> current_trace_id() is None -> column is NULL.
        journal.record_step(turn_index=1, step_index=0, status="running")
        row = conn.execute("SELECT trace_id FROM steps WHERE session_id = ?", ("s1",)).fetchone()
        assert row[0] is None


class TestTracePropagationIntoRemoteNode:
    async def test_orchestrator_remote_node_carries_trace(self, monkeypatch):
        """C1 regression: a remote-node fan-out carries the W3C trace into invoke_peer."""
        import koboi.server.peers as peers_mod
        from koboi.orchestration.orchestrator import Orchestrator
        from koboi.orchestration.remote_proxy import RemoteAgentProxy
        from koboi.orchestration.router import KeywordRouter
        from koboi.server.peers import PeerConfig, PeerInvokeResult, PeerRegistry
        from koboi.types import AgentDef
        from tests.conftest import MockClient, make_mock_response

        registry = PeerRegistry()
        registry._peers["Y"] = PeerConfig(name="Y", url="http://localhost:8002", token="t")
        seen: dict = {}

        async def fake_invoke(peer, msg):
            seen["trace"] = tc.current_trace_id()  # capture the context at peer-call time
            return PeerInvokeResult(content="ans")

        monkeypatch.setattr(peers_mod, "invoke_peer", fake_invoke)
        proxy = RemoteAgentProxy("review", "Y", registry)
        router = KeywordRouter(agent_defs=[AgentDef(name="review", keywords=["review"])])
        client = MockClient([make_mock_response(content="synth")])
        orch = Orchestrator(client=client, router=router, agents_map={"review": proxy})

        tc.begin_request(_GOOD)
        root = tc.current_trace_id()
        await orch.run("please review", mode="sequential")
        assert seen.get("trace") == root  # trace propagated into the remote-node peer call
