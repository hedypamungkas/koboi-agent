"""Tests for P4 cross-instance W3C traceparent propagation."""

from __future__ import annotations

import asyncio
import sqlite3

import httpx
import pytest
from httpx import ASGITransport

from koboi import tracing_context as tc
from koboi.config import Config
from koboi.journal import StepJournal
from koboi.memory_sqlite import ensure_steps_table
from koboi.server.app import create_app
from koboi.server.peers import PeerConfig, PeerInvokeResult, invoke_peer
from tests.conftest import MockClient, make_mock_response, make_mock_tool_call

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
        assert res.content == "ANS" and res.receiver_trace_id == "peerT"
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

    async def test_retry_succeeds_after_transient_errors(self, monkeypatch):
        # Gap 1.1: invoke_peer retries on transient failures (ConnectError) then succeeds.
        async def _no_sleep(*a):
            pass

        monkeypatch.setattr(asyncio, "sleep", _no_sleep)  # eliminate backoff delays

        calls = {"n": 0}

        class _RetryClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None, headers=None):
                calls["n"] += 1
                if calls["n"] < 3:
                    raise httpx.ConnectError("transient")
                return _Resp()

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _RetryClient())
        res = await invoke_peer(PeerConfig(name="C", url="http://localhost:8002", token="t"), "hi")
        assert res.content == "ANS"
        assert calls["n"] == 3  # 2 retries + 1 success

    async def test_response_too_large_rejected(self, monkeypatch):
        # Gap 1.4: content > _MAX_PEER_CONTENT → ValueError (no retry — not transient).
        class _BigResp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"content": "x" * 65537}

        class _BigClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None, headers=None):
                return _BigResp()

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _BigClient())
        with pytest.raises(ValueError, match="too large"):
            await invoke_peer(PeerConfig(name="C", url="http://localhost:8002", token="t"), "hi")

    async def test_non_json_response_handled(self, monkeypatch):
        # Gap 1.2: B returns HTML (proxy error) → clear ValueError, not bare JSONDecodeError.
        class _HtmlResp:
            status_code = 200
            text = "<html>502 Bad Gateway</html>"

            def raise_for_status(self):
                pass

            def json(self):
                raise ValueError("Expecting value")

        class _HtmlClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None, headers=None):
                return _HtmlResp()

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _HtmlClient())
        with pytest.raises(ValueError, match="non-JSON"):
            await invoke_peer(PeerConfig(name="C", url="http://localhost:8002", token="t"), "hi")

    async def test_wrong_shape_json_handled(self, monkeypatch):
        # Gap 1.3: {"content": null} → ValueError ("no string content").
        class _NullResp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"content": None}

        class _NullClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None, headers=None):
                return _NullResp()

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _NullClient())
        with pytest.raises(ValueError, match="no string content"):
            await invoke_peer(PeerConfig(name="C", url="http://localhost:8002", token="t"), "hi")


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


def _trace_ids(db_path: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT DISTINCT trace_id FROM steps WHERE trace_id IS NOT NULL").fetchall()
    conn.close()
    return {r[0] for r in rows}


class TestCrossInstanceTrace:
    async def test_same_trace_id_in_both_journals(self, tmp_path, monkeypatch):
        """Capstone: one W3C trace-id lands in BOTH instances' step journals across a hop."""
        db_y = tmp_path / "y.db"
        db_x = tmp_path / "x.db"

        # Instance Y: agent C (sqlite journal), accepts peer token tok-y.
        cfg_y = Config.from_dict(
            {
                "agent": {"name": "C", "mode": "chat", "system_prompt": "C", "max_iterations": 3},
                "llm": {"provider": "openai", "model": "x", "api_key": "x"},
                "memory": {"backend": "sqlite", "db_path": str(db_y)},
                "peers": {"enabled": True, "inbound_tokens": ["tok-y"]},
            }
        )
        app_y = create_app(cfg_y, client_factory=lambda: MockClient([make_mock_response(content="C-answer")]))

        # Route X's call_peer_agent httpx at Y in-process (the hop carries the traceparent).
        real = httpx.AsyncClient

        class _Routed(real):
            def __init__(self, *a, **k):
                k.setdefault("transport", ASGITransport(app=app_y))
                super().__init__(*a, **k)

        monkeypatch.setattr(httpx, "AsyncClient", _Routed)

        # Instance X: agent A (act; sqlite journal), peer C -> Y.
        cfg_x = Config.from_dict(
            {
                "agent": {"name": "A", "mode": "act", "system_prompt": "A", "max_iterations": 5},
                "llm": {"provider": "openai", "model": "x", "api_key": "x"},
                "memory": {"backend": "sqlite", "db_path": str(db_x)},
                "peers": {
                    "enabled": True,
                    "allow_private_network": True,
                    "peers": [{"name": "C", "url": "http://peer-y:8000", "token": "tok-y"}],
                },
            }
        )
        app_x = create_app(
            cfg_x,
            client_factory=lambda: MockClient(
                [
                    make_mock_response(
                        tool_calls=[make_mock_tool_call("call_peer_agent", {"calls": [{"peer": "C", "message": "hi"}]})]
                    ),
                    make_mock_response(content="A got C-answer"),
                ]
            ),
        )

        tc.begin_request(_GOOD)
        root = tc.current_trace_id()
        agent = await app_x.state.pool.get_or_create("x-sess")
        await agent.run("ask C")

        # The same W3C trace-id is stamped in BOTH instances' step journals.
        assert root in _trace_ids(str(db_x))  # caller journaled the trace
        assert root in _trace_ids(str(db_y))  # peer continued the SAME trace-id via the header
