"""Wave 3: opt-in parallel execution of all-read-only tool batches."""

from __future__ import annotations

import asyncio
import json

from koboi.guardrails.rate_limiter import RateLimitConfig, RateLimiter
from koboi.hooks.chain import Hook, HookChain, HookContext, HookEvent
from koboi.loop import AgentCore
from koboi.memory import ConversationMemory
from koboi.tools.registry import ToolRegistry
from koboi.types import RiskLevel, ToolCall

from tests.conftest import MockClient, make_mock_response


class _Tracker:
    """Shared in-flight counter proving (non-)overlap."""

    def __init__(self):
        self.in_flight = 0
        self.max_in_flight = 0

    async def enter(self, delay: float):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(delay)
        self.in_flight -= 1


def _registry(tracker: _Tracker) -> ToolRegistry:
    registry = ToolRegistry()

    async def read_slow() -> str:
        await tracker.enter(0.05)
        return "SLOW-RESULT"

    async def read_fast() -> str:
        await tracker.enter(0.0)
        return "FAST-RESULT"

    def write_thing() -> str:
        return "WROTE"

    def moderate_thing() -> str:
        return "MODERATE-WROTE"

    empty = {"type": "object", "properties": {}, "required": []}
    registry.register(name="read_slow", description="slow read", parameters=empty, fn=read_slow)
    registry.register(name="read_fast", description="fast read", parameters=empty, fn=read_fast)
    registry.register(
        name="write_thing",
        description="a mutating tool",
        parameters=empty,
        fn=write_thing,
        risk_level=RiskLevel.SAFE,  # SAFE but non-idempotent -> ineligible
        idempotent=False,
    )
    registry.register(
        name="moderate_thing",
        description="a moderate-risk tool",
        parameters=empty,
        fn=moderate_thing,
        risk_level=RiskLevel.MODERATE,  # idempotent but MODERATE -> still ineligible
        idempotent=True,
    )
    return registry


def _tc(name: str, tc_id: str) -> ToolCall:
    return ToolCall(id=tc_id, name=name, arguments=json.dumps({}))


def _core(tracker: _Tracker, responses, *, enabled=True, max_concurrency=4, **kw) -> AgentCore:
    return AgentCore(
        client=MockClient(responses),
        memory=ConversationMemory(),
        tools=_registry(tracker),
        max_iterations=4,
        parallel_tools_config={"enabled": enabled, "max_concurrency": max_concurrency} if enabled else None,
        **kw,
    )


def _tool_rows(memory: ConversationMemory) -> list[tuple[str, str]]:
    return [(m["tool_call_id"], m["content"]) for m in memory.get_messages() if m.get("role") == "tool"]


def _batch_response(*names_ids):
    return make_mock_response(tool_calls=[_tc(n, i) for n, i in names_ids])


class TestParallelExecution:
    async def test_order_preserved_and_concurrent(self):
        tracker = _Tracker()
        core = _core(tracker, [_batch_response(("read_slow", "t1"), ("read_fast", "t2")), make_mock_response("done")])
        result = await core.run("go")
        assert result.success is True
        rows = _tool_rows(core.memory)
        # Memory order == original tool_calls order despite slow-first.
        assert [r[0] for r in rows] == ["t1", "t2"]
        assert rows[0][1] == "SLOW-RESULT"
        assert tracker.max_in_flight == 2  # genuinely overlapped

    async def test_mixed_batch_stays_sequential(self):
        tracker = _Tracker()
        core = _core(
            tracker,
            [_batch_response(("read_slow", "t1"), ("write_thing", "t2")), make_mock_response("done")],
        )
        await core.run("go")
        assert tracker.max_in_flight <= 1  # write_thing (idempotent=False) disables the batch

    async def test_default_off_is_sequential(self):
        tracker = _Tracker()
        core = _core(
            tracker, [_batch_response(("read_slow", "t1"), ("read_slow", "t2")), make_mock_response("d")], enabled=False
        )
        await core.run("go")
        assert tracker.max_in_flight <= 1

    async def test_max_concurrency_respected(self):
        tracker = _Tracker()
        calls = tuple(("read_slow", f"t{i}") for i in range(4))
        core = _core(tracker, [_batch_response(*calls), make_mock_response("d")], max_concurrency=2)
        await core.run("go")
        assert tracker.max_in_flight <= 2

    async def test_exception_isolation_and_pairing(self, monkeypatch):
        tracker = _Tracker()
        core = _core(tracker, [_batch_response(("read_slow", "t1"), ("read_fast", "t2")), make_mock_response("d")])
        real = core._pipeline.execute_tool_call

        async def boom(tc, iteration, on_event=None, defer_record=False):
            if tc.id == "t1":
                raise RuntimeError("kaboom")
            return await real(tc, iteration=iteration, on_event=on_event, defer_record=defer_record)

        monkeypatch.setattr(core._pipeline, "execute_tool_call", boom)
        result = await core.run("go")
        assert result.success is True
        rows = _tool_rows(core.memory)
        assert [r[0] for r in rows] == ["t1", "t2"]  # pairing invariant holds
        assert "Error executing 'read_slow'" in rows[0][1]
        assert rows[1][1] == "FAST-RESULT"  # sibling unaffected
        errored = [o for o in result.pipeline_outcomes if o["errored"]]
        assert len(errored) == 1 and errored[0]["error_kind"] == "execution_error"

    async def test_checkpointer_never_fires_for_parallel_batch(self):
        tracker = _Tracker()

        class SpyCheckpointer:
            def __init__(self):
                self.commits = 0
                self.workdir = "."

            def ensure(self):
                return True

            def commit(self, label):
                self.commits += 1
                return "sha"

        spy = SpyCheckpointer()
        core = _core(
            tracker,
            [_batch_response(("read_slow", "t1"), ("read_fast", "t2")), make_mock_response("d")],
            checkpointer=spy,
        )
        await core.run("go")
        assert spy.commits == 0  # eligible batches are idempotent-only

    async def test_hook_injects_land_after_results(self):
        tracker = _Tracker()

        class InjectingHook(Hook):
            def handles(self):
                return [HookEvent.POST_TOOL_USE]

            async def execute(self, ctx: HookContext) -> HookContext:
                ctx.inject_messages.append(f"note-for-{ctx.tool_name}")
                return ctx

        core = _core(tracker, [_batch_response(("read_slow", "t1"), ("read_fast", "t2")), make_mock_response("d")])
        core._pipeline.hooks = HookChain([InjectingHook()])
        await core.run("go")
        msgs = core.memory.get_messages()
        tool_idx = [i for i, m in enumerate(msgs) if m.get("role") == "tool"]
        inject_idx = [
            i for i, m in enumerate(msgs) if m.get("role") == "system" and "note-for-" in (m.get("content") or "")
        ]
        assert inject_idx and tool_idx
        assert min(inject_idx) > max(tool_idx)  # all injects AFTER the batch's results
        # grouped in original call order
        contents = [msgs[i]["content"] for i in inject_idx]
        assert contents == ["note-for-read_slow", "note-for-read_fast"]

    async def test_rate_limit_deny_inside_batch(self):
        tracker = _Tracker()
        limiter = RateLimiter(RateLimitConfig(max_tool_calls_per_session=1))
        core = _core(
            tracker,
            [_batch_response(("read_fast", "t1"), ("read_fast", "t2")), make_mock_response("d")],
            rate_limiter=limiter,
        )
        result = await core.run("go")
        rows = _tool_rows(core.memory)
        assert [r[0] for r in rows] == ["t1", "t2"]  # both answered, original order
        skipped = [o for o in result.pipeline_outcomes if o["skipped"]]
        assert len(skipped) == 1  # exactly one rate-limited

    async def test_sequential_behavior_unchanged_for_single_call(self):
        tracker = _Tracker()
        core = _core(tracker, [_batch_response(("read_fast", "t1")), make_mock_response("d")])
        result = await core.run("go")
        assert result.success is True
        assert _tool_rows(core.memory)[0][1] == "FAST-RESULT"

    async def test_slowest_completes_first_order_preserved(self):
        # Dispatch SLOW at call-index 0 and FAST at call-index 1: completion
        # order is FAST then SLOW, but memory rows must land in ORIGINAL call
        # order [t0, t1] (Anthropic tool_result pairing is positional).
        tracker = _Tracker()
        core = _core(
            tracker,
            [_batch_response(("read_slow", "t0"), ("read_fast", "t1")), make_mock_response("done")],
        )
        result = await core.run("go")
        assert result.success is True
        rows = _tool_rows(core.memory)
        assert [r[0] for r in rows] == ["t0", "t1"]  # original call order, NOT completion
        assert rows[0][1] == "SLOW-RESULT"
        assert rows[1][1] == "FAST-RESULT"
        assert tracker.max_in_flight >= 2  # genuinely overlapped

    async def test_exception_in_one_tool_preserves_pairing(self, monkeypatch):
        # 3-call read-only batch where the MIDDLE call raises: results land in
        # original order [t0_ok, t1_error, t2_ok]; t2 still ran (sibling isolation).
        tracker = _Tracker()
        core = _core(
            tracker,
            [
                _batch_response(("read_fast", "t0"), ("read_slow", "t1"), ("read_fast", "t2")),
                make_mock_response("done"),
            ],
        )
        real = core._pipeline.execute_tool_call

        async def boom(tc, iteration, on_event=None, defer_record=False):
            if tc.id == "t1":
                raise RuntimeError("kaboom")
            return await real(tc, iteration=iteration, on_event=on_event, defer_record=defer_record)

        monkeypatch.setattr(core._pipeline, "execute_tool_call", boom)
        result = await core.run("go")
        assert result.success is True
        rows = _tool_rows(core.memory)
        assert [r[0] for r in rows] == ["t0", "t1", "t2"]  # pairing invariant holds
        assert rows[0][1] == "FAST-RESULT"  # t0 ok
        assert "Error executing 'read_slow'" in rows[1][1]  # t1 errored
        assert rows[2][1] == "FAST-RESULT"  # t2 sibling unaffected
        errored = [o for o in result.pipeline_outcomes if o["errored"]]
        assert len(errored) == 1 and errored[0]["error_kind"] == "execution_error"

    async def test_mixed_risk_batch_stays_sequential(self):
        # A MODERATE tool in the batch disables concurrency for the WHOLE batch
        # (all-or-nothing gate in _parallel_batch_eligible); max_in_flight stays at 1.
        tracker = _Tracker()
        core = _core(
            tracker,
            [_batch_response(("read_slow", "t0"), ("moderate_thing", "t1")), make_mock_response("done")],
        )
        await core.run("go")
        assert tracker.max_in_flight <= 1  # MODERATE -> sequential fallback


class TestConfigWiring:
    def test_facade_threads_parallel_tools(self, tmp_path):
        from koboi.facade import KoboiAgent

        agent = KoboiAgent.from_dict(
            {
                "agent": {"name": "t", "parallel_tools": {"enabled": True, "max_concurrency": 2}},
                "llm": {"provider": "openai", "model": "m", "api_key": "k"},
                "memory": {"backend": "in_memory"},
            }
        )
        assert agent._core._parallel_enabled is True
        assert agent._core._parallel_max_concurrency == 2

    def test_default_disabled(self):
        from koboi.facade import KoboiAgent

        agent = KoboiAgent.from_dict(
            {
                "agent": {"name": "t"},
                "llm": {"provider": "openai", "model": "m", "api_key": "k"},
                "memory": {"backend": "in_memory"},
            }
        )
        assert agent._core._parallel_enabled is False
