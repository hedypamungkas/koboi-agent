"""Unit tests for koboi/server/approvals.py (no FastAPI)."""

from __future__ import annotations

import asyncio

import pytest

from koboi.guardrails.approval_types import ApprovalRequest, ApprovalResponse
from koboi.server.approvals import ApprovalCoordinator, ApprovalRegistry
from koboi.types import RiskLevel


def _req(approval_id: str = "ap_test") -> ApprovalRequest:
    return ApprovalRequest(
        tool_name="run_shell",
        arguments="{}",
        risk_level=RiskLevel.DESTRUCTIVE,
        reason="test",
        approval_id=approval_id,
    )


class TestApprovalCoordinator:
    async def test_request_pushes_event_then_awaits(self):
        queue: asyncio.Queue = asyncio.Queue()
        coord = ApprovalCoordinator(queue)
        task = asyncio.create_task(coord.request(_req()))
        # PendingApprovalEvent should be on the queue immediately (put_nowait).
        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert event.approval_id == "ap_test"
        assert event.tool_name == "run_shell"
        assert event.risk_level == "destructive"
        # Resolve -> task completes with the response.
        assert coord.resolve("ap_test", ApprovalResponse(approved=True)) is True
        result = await asyncio.wait_for(task, timeout=1.0)
        assert result.approved is True

    async def test_resolve_unknown_returns_false(self):
        coord = ApprovalCoordinator(asyncio.Queue())
        assert coord.resolve("nope", ApprovalResponse(approved=True)) is False

    async def test_double_resolve_returns_false(self):
        queue: asyncio.Queue = asyncio.Queue()
        coord = ApprovalCoordinator(queue)
        task = asyncio.create_task(coord.request(_req()))
        await queue.get()
        coord.resolve("ap_test", ApprovalResponse(approved=True))
        await task
        assert coord.resolve("ap_test", ApprovalResponse(approved=False)) is False

    async def test_cancel_all_cancels_pending(self):
        queue: asyncio.Queue = asyncio.Queue()
        coord = ApprovalCoordinator(queue)
        task = asyncio.create_task(coord.request(_req()))
        await queue.get()
        coord.cancel_all()
        with pytest.raises(asyncio.CancelledError):
            await task


class TestApprovalRegistry:
    async def test_register_get_unregister(self):
        reg = ApprovalRegistry()
        coord = ApprovalCoordinator(asyncio.Queue())
        reg.register("s1", coord)
        assert reg.get("s1") is coord
        reg.unregister("s1")
        assert reg.get("s1") is None

    async def test_unregister_cancels_coordinator_futures(self):
        reg = ApprovalRegistry()
        coord = ApprovalCoordinator(asyncio.Queue())
        reg.register("s1", coord)
        reg.unregister("s1")
        # After unregister, the coordinator's futures are cancelled.
        assert coord.resolve("anything", ApprovalResponse(approved=True)) is False


class TestQueueBridgeIntegration:
    """Direct queue-bridge integration (no HTTP / ASGI transport).

    Tests the full M2 flow: agent → AsyncCallbackApprovalHandler →
    ApprovalCoordinator (queue + Future) → resolve → tool executes.
    """

    async def test_approve_then_complete(self):
        from koboi.events import ErrorEvent, PendingApprovalEvent
        from koboi.facade import KoboiAgent
        from koboi.config import Config
        from koboi.guardrails.approval import AsyncCallbackApprovalHandler
        from koboi.types import RiskLevel
        from tests.conftest import MockClient, make_mock_response, make_mock_tool_call

        config = Config.from_dict(
            {
                "agent": {"name": "t", "system_prompt": "h", "max_iterations": 3, "mode": "act"},
                "llm": {"provider": "openai", "model": "m", "api_key": "test", "base_url": "http://x"},
                "memory": {"backend": "in_memory"},
                "sandbox": {"backend": "passthrough"},
            },
            validate=True,
        )
        agent = KoboiAgent.from_dict(config.raw)
        agent._core.client = MockClient(
            [
                make_mock_response(tool_calls=[make_mock_tool_call("danger")]),
                make_mock_response(content="done"),
            ]
        )
        agent.add_tool(
            "danger",
            lambda **kw: "ok",
            "destructive test",
            {"type": "object", "properties": {}, "required": []},
            risk_level=RiskLevel.DESTRUCTIVE,
        )

        queue: asyncio.Queue = asyncio.Queue()
        coord = ApprovalCoordinator(queue, timeout=5.0)
        handler = AsyncCallbackApprovalHandler(callback=coord.request, trust_db=agent.trust_db, timeout=5.0)
        if hasattr(agent._core, "_tool_pipeline"):
            del agent._core._tool_pipeline
        agent._core.approval_handler = handler

        events: list = []

        async def run_agent():
            try:
                async for ev in agent.run_stream("go"):
                    await queue.put(ev)
            except Exception as exc:
                await queue.put(ErrorEvent(error=exc))
            finally:
                await queue.put(None)

        task = asyncio.create_task(run_agent())
        while True:
            ev = await asyncio.wait_for(queue.get(), timeout=10.0)
            if ev is None:
                break
            events.append(ev)
            if isinstance(ev, PendingApprovalEvent):
                coord.resolve(ev.approval_id, ApprovalResponse(approved=True))

        await task
        types = [type(e).__name__ for e in events]
        assert "PendingApprovalEvent" in types
        assert "CompleteEvent" in types
        # Tool was approved -> executed -> result "ok"
        tool_results = [e for e in events if type(e).__name__ == "ToolResultEvent"]
        assert any("ok" in getattr(e, "result", "").lower() for e in tool_results)

    async def test_deny_blocks_tool(self):
        from koboi.events import ErrorEvent, PendingApprovalEvent
        from koboi.facade import KoboiAgent
        from koboi.config import Config
        from koboi.guardrails.approval import AsyncCallbackApprovalHandler
        from koboi.types import RiskLevel
        from tests.conftest import MockClient, make_mock_response, make_mock_tool_call

        config = Config.from_dict(
            {
                "agent": {"name": "t", "system_prompt": "h", "max_iterations": 3, "mode": "act"},
                "llm": {"provider": "openai", "model": "m", "api_key": "test", "base_url": "http://x"},
                "memory": {"backend": "in_memory"},
                "sandbox": {"backend": "passthrough"},
            },
            validate=True,
        )
        agent = KoboiAgent.from_dict(config.raw)
        agent._core.client = MockClient(
            [
                make_mock_response(tool_calls=[make_mock_tool_call("danger")]),
                make_mock_response(content="done"),
            ]
        )
        agent.add_tool(
            "danger",
            lambda **kw: "ok",
            "destructive test",
            {"type": "object", "properties": {}, "required": []},
            risk_level=RiskLevel.DESTRUCTIVE,
        )

        queue: asyncio.Queue = asyncio.Queue()
        coord = ApprovalCoordinator(queue, timeout=5.0)
        handler = AsyncCallbackApprovalHandler(callback=coord.request, trust_db=agent.trust_db, timeout=5.0)
        if hasattr(agent._core, "_tool_pipeline"):
            del agent._core._tool_pipeline
        agent._core.approval_handler = handler

        events: list = []

        async def run_agent():
            try:
                async for ev in agent.run_stream("go"):
                    await queue.put(ev)
            except Exception as exc:
                await queue.put(ErrorEvent(error=exc))
            finally:
                await queue.put(None)

        task = asyncio.create_task(run_agent())
        while True:
            ev = await asyncio.wait_for(queue.get(), timeout=10.0)
            if ev is None:
                break
            events.append(ev)
            if isinstance(ev, PendingApprovalEvent):
                coord.resolve(ev.approval_id, ApprovalResponse(approved=False))

        await task
        types = [type(e).__name__ for e in events]
        assert "PendingApprovalEvent" in types
        assert "CompleteEvent" in types
        # Tool was denied -> result contains "denied"
        tool_results = [e for e in events if type(e).__name__ == "ToolResultEvent"]
        assert any("denied" in getattr(e, "result", "").lower() for e in tool_results)
