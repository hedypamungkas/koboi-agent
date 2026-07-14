"""Tests for the B1 handover primitive (Wave 2 Phase 2).

Covers: the ``transfer_to_human`` tool raises ``AgentHandoverError``; the loop
propagates it uncaught (the deadlock-free mechanism -- NO Future awaited);
``HandoverEvent`` serialization; mode-gating; and the server-level end-to-end
(handover on the SSE stream + the deadlock-regression: a second ``/chat/stream``
on the SAME session acquires the released ``pool.session_lock``) + the
``/transfer`` ownership route + the ``awaiting_human`` job terminal status.
"""

from __future__ import annotations

import json

import pytest

from koboi.events import HandoverEvent, event_to_dict
from koboi.exceptions import AgentHandoverError
from koboi.modes import AgentMode, ModeManager
from koboi.tools.builtin import register_all
from koboi.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Unit (no fastapi)
# ---------------------------------------------------------------------------


class TestHandoverTool:
    def _registry(self) -> ToolRegistry:
        r = ToolRegistry()
        register_all(r)
        return r

    async def test_tool_raises_agent_handover_error(self):
        r = self._registry()
        with pytest.raises(AgentHandoverError) as ei:
            await r.execute("transfer_to_human", json.dumps({"reason": "complex", "summary": "cust"}))
        assert ei.value.reason == "complex"
        assert ei.value.summary == "cust"

    def test_tool_registered_safe(self):
        td = self._registry().get_definition("transfer_to_human")
        assert td is not None
        # SAFE: handover is a deliberate yield, not a risky action -- it must NOT
        # be approval-gated (MODERATE would deny-by-default in autonomous jobs).
        from koboi.types import RiskLevel

        assert td.risk_level == RiskLevel.SAFE

    def test_handover_event_serializes(self):
        d = event_to_dict(HandoverEvent(handover_id="h1", reason="r", summary="s"))
        assert d["type"] == "handover"
        assert d["handover_id"] == "h1"
        assert d["reason"] == "r"
        assert d["summary"] == "s"

    def test_mode_gating_blocks_in_chat_allows_in_act(self):
        # transfer_to_human is NOT read-only -> blocked in CHAT/PLAN, allowed in ACT+.
        chat = ModeManager(initial_mode=AgentMode.CHAT)
        allowed_chat, _ = chat.is_tool_allowed("transfer_to_human")
        act = ModeManager(initial_mode=AgentMode.ACT)
        allowed_act, _ = act.is_tool_allowed("transfer_to_human")
        assert allowed_chat is False
        assert allowed_act is True


# ---------------------------------------------------------------------------
# AgentCore propagation (no fastapi) -- the deadlock-free mechanism
# ---------------------------------------------------------------------------


class TestHandoverPropagation:
    @staticmethod
    def _core():
        from koboi.loop import AgentCore
        from koboi.memory import ConversationMemory
        from tests.conftest import MockClient, make_mock_response, make_mock_tool_call

        tools = ToolRegistry()
        register_all(tools)
        client = MockClient(
            [
                make_mock_response(
                    content="I'll transfer you to a colleague.",
                    tool_calls=[make_mock_tool_call("transfer_to_human", {"reason": "r", "summary": "s"})],
                )
            ]
        )
        return AgentCore(
            client=client,
            memory=ConversationMemory(),
            tools=tools,
            mode_manager=ModeManager(initial_mode=AgentMode.ACT),
            max_iterations=3,
        )

    async def test_run_stream_propagates_handover(self):
        core = self._core()
        with pytest.raises(AgentHandoverError) as ei:
            async for _ in core.run_stream("please help"):
                pass
        assert ei.value.reason == "r"
        assert ei.value.summary == "s"

    async def test_run_propagates_handover(self):
        core = self._core()
        with pytest.raises(AgentHandoverError):
            await core.run("please help")


# ---------------------------------------------------------------------------
# Server-level (fastapi) -- e2e + deadlock-regression + /transfer + jobs
# ---------------------------------------------------------------------------


def _need_fastapi():
    pytest.importorskip("fastapi")
    import httpx  # noqa: F401
    from httpx import ASGITransport  # noqa: F401


def _server_config():
    from koboi.config import Config

    return Config.from_dict(
        {
            "agent": {"name": "srv", "system_prompt": "h", "max_iterations": 3},
            "llm": {"provider": "openai", "model": "gpt-4o-mini", "api_key": "test"},
            "memory": {"backend": "in_memory"},
            "sandbox": {"backend": "restricted"},
            "server": {"auth_required": False},
            # The facade registers builtins only when tools.builtin is set.
            "tools": {"builtin": ["transfer_to_human"]},
        },
        validate=True,
    )


def _parse_sse(text: str) -> list:
    out = []
    for line in text.split("\n"):
        if line.startswith("data: "):
            payload = line[6:]
            out.append("[DONE]" if payload == "[DONE]" else json.loads(payload))
    return out


class TestHandoverServerE2E:
    async def test_handover_event_on_stream_then_lock_released(self):
        _need_fastapi()
        import httpx
        from koboi.server import create_app
        from tests.conftest import MockClient, make_mock_response, make_mock_tool_call

        # 2 responses: turn 1 -> transfer_to_human (handover); turn 2 -> normal.
        handover_resp = make_mock_response(
            content="Transferring you.",
            tool_calls=[make_mock_tool_call("transfer_to_human", {"reason": "r", "summary": "s"})],
        )
        normal_resp = make_mock_response(content="Hello from the human side.")
        factory = lambda: MockClient([handover_resp, normal_resp])  # noqa: E731
        app = create_app(_server_config(), client_factory=factory, enable_cors=False)
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            # Turn 1: agent calls transfer_to_human -> HandoverEvent on the stream.
            async with client.stream(
                "POST",
                "/v1/chat/stream",
                json={"message": "I need a human", "mode": "act"},
                headers={"X-Session-Id": "sess-handover"},
            ) as r:
                body1 = (await r.aread()).decode()
            events1 = _parse_sse(body1)
            types1 = [e.get("type") if isinstance(e, dict) else e for e in events1]
            assert "handover" in types1, f"expected HandoverEvent, got {types1}"
            assert events1[-1] == "[DONE]"

            # Turn 2 (SAME session): must succeed, proving pool.session_lock was
            # released (no deadlock). If the lock were still held, this would hang.
            async with client.stream(
                "POST",
                "/v1/chat/stream",
                json={"message": "thanks", "mode": "act"},
                headers={"X-Session-Id": "sess-handover"},
                timeout=5.0,
            ) as r:
                body2 = (await r.aread()).decode()
            events2 = _parse_sse(body2)
            types2 = [e.get("type") if isinstance(e, dict) else e for e in events2]
            assert "complete" in types2, f"second turn must complete (lock released); got {types2}"


class TestTransferRoute:
    async def test_transfer_reassigns_owner(self):
        _need_fastapi()
        import httpx
        from koboi.server import create_app
        from tests.conftest import MockClient, make_mock_response

        app = create_app(
            _server_config(),
            client_factory=lambda: MockClient([make_mock_response(content="hi")]),
            enable_cors=False,
        )
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            # Create an owned session (owner = the creator's key id; dev-open here).
            create = await client.post("/v1/sessions", json={}, headers={"X-Session-Id": "sess-x"})
            sid = create.json().get("session_id") or "sess-x"

            # /transfer reassigns ownership to operator "op1".
            resp = await client.post(
                f"/v1/sessions/{sid}/transfer",
                json={"operator": "op1"},
                headers={"X-Session-Id": sid},
            )
            assert resp.status_code == 200
            assert resp.json()["transferred"] is True
            assert resp.json()["owner"] == "op1"


class TestHandoverJob:
    async def test_job_handover_is_awaiting_human(self):
        _need_fastapi()
        import httpx
        from koboi.server import create_app
        from tests.conftest import MockClient, make_mock_response, make_mock_tool_call

        handover_resp = make_mock_response(
            content="transferring",
            tool_calls=[make_mock_tool_call("transfer_to_human", {"reason": "r", "summary": "s"})],
        )
        app = create_app(
            _server_config(),
            client_factory=lambda: MockClient([handover_resp]),
            enable_cors=False,
        )
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            submit = await client.post(
                "/v1/jobs",
                json={"message": "help", "mode": "act"},
                headers={"X-Session-Id": "sess-job"},
            )
            assert submit.status_code == 202
            job_id = submit.json()["job_id"]

            # Poll until terminal.
            status = None
            for _ in range(40):
                got = await client.get(f"/v1/jobs/{job_id}")
                status = got.json().get("status")
                if status in {"awaiting_human", "completed", "failed", "timed_out", "cancelled"}:
                    break
            assert status == "awaiting_human", f"expected awaiting_human, got {status}"
