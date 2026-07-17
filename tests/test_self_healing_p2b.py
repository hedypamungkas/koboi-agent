"""tests/test_self_healing_p2b.py -- orchestration per-node/subtree retry (self-healing P2b).

Verifies the dynamic-mode replan loop (max_replans>0) re-runs ONLY the failed
subtree, carrying forward succeeded nodes (+ nodes that fired a non-idempotent
side-effecting tool) so side effects never double-fire. Resolves the P0 caveat.
"""

from __future__ import annotations

from koboi.memory import ConversationMemory
from koboi.orchestration.orchestrator import Orchestrator
from koboi.orchestration.planner import PlanResult, PlanStep
from koboi.tools.registry import ToolRegistry
from koboi.types import AgentResult, RiskLevel, RoutingDecision, RunResult, ToolCall
from tests.conftest import MockClient, make_mock_response


class _AllRouter:
    def __init__(self, names):
        self._names = names

    async def route(self, q):
        return RoutingDecision(query=q, agents=list(self._names), confidence=1.0, method="keyword", reasoning="all")


def _step(sid: str, depends_on: list[str] | None = None) -> PlanStep:
    return PlanStep(id=sid, instruction=f"do {sid}", depends_on=depends_on or [])


def _ok(name: str) -> AgentResult:
    return AgentResult(agent_name=name, answer=f"{name}-ok", elapsed_seconds=0, tokens_used=0)


def _err(name: str, **extra) -> AgentResult:
    return AgentResult(agent_name=name, answer=f"Error: {name}", elapsed_seconds=0, tokens_used=0, failed=True, **extra)


class TestReplanPerNodeRetry:
    async def test_replan_skips_succeeded_nodes(self, mock_client, monkeypatch):
        from koboi.orchestration import planner

        plan_calls = {"n": 0}

        async def counting_plan(client, instruction, **kw):
            plan_calls["n"] += 1
            return PlanResult(
                needs_workflow=True,
                reason="multi",
                steps=[_step("a"), _step("b", ["a"]), _step("c", ["a"]), _step("d", ["b", "c"])],
            )

        monkeypatch.setattr(planner, "plan_or_skip", counting_plan)

        client = mock_client(responses=[make_mock_response("synthesis")])
        orch = Orchestrator(client=client, router=_AllRouter([]), agents_map={}, default_mode="dynamic", max_replans=1)

        run_log: list[str] = []

        async def controlled_single(name, query_input):
            run_log.append(name)
            # c fails the first time only; everything else succeeds.
            if name == "c" and run_log.count("c") == 1:
                return _err("c")
            return _ok(name)

        orch._run_single = controlled_single
        await orch.run("task", mode="dynamic")

        assert plan_calls["n"] == 2  # initial plan + 1 replan
        # a, b succeeded and are NOT downstream of failed c -> carried forward (once).
        # c failed -> re-run. d depends on c -> its round-1 success was stale -> re-run.
        assert run_log.count("a") == 1
        assert run_log.count("b") == 1
        assert run_log.count("c") == 2
        assert run_log.count("d") == 2

    async def test_replan_carries_non_idempotent_failed_node(self, mock_client, monkeypatch):
        from koboi.orchestration import planner

        plan_calls = {"n": 0}

        async def counting_plan(client, instruction, **kw):
            plan_calls["n"] += 1
            return PlanResult(
                needs_workflow=True, reason="multi", steps=[_step("a"), _step("b", ["a"]), _step("c", ["b"])]
            )

        monkeypatch.setattr(planner, "plan_or_skip", counting_plan)

        client = mock_client(responses=[make_mock_response("synthesis")])
        orch = Orchestrator(client=client, router=_AllRouter([]), agents_map={}, default_mode="dynamic", max_replans=2)

        run_log: list[str] = []

        async def controlled_single(name, query_input):
            run_log.append(name)
            if name == "b":
                # b failed AND fired a non-idempotent tool -> carried forward, never re-run.
                return _err("b", had_non_idempotent_tool=True)
            return _ok(name)

        orch._run_single = controlled_single
        await orch.run("task", mode="dynamic")

        # b is non-idempotent-failed -> nothing retryable -> no replan past the initial plan.
        assert plan_calls["n"] == 1
        assert run_log.count("b") == 1

    async def test_replan_reruns_only_failed_subtree(self, mock_client, monkeypatch):
        from koboi.orchestration import planner

        plan_calls = {"n": 0}

        async def counting_plan(client, instruction, **kw):
            plan_calls["n"] += 1
            return PlanResult(
                needs_workflow=True, reason="multi", steps=[_step("a"), _step("b", ["a"]), _step("c", ["b"])]
            )

        monkeypatch.setattr(planner, "plan_or_skip", counting_plan)

        client = mock_client(responses=[make_mock_response("synthesis")])
        orch = Orchestrator(client=client, router=_AllRouter([]), agents_map={}, default_mode="dynamic", max_replans=1)

        run_log: list[str] = []

        async def controlled_single(name, query_input):
            run_log.append(name)
            # a succeeds; b and c fail round 1, succeed on re-run.
            if name in ("b", "c") and run_log.count(name) == 1:
                return _err(name)
            return _ok(name)

        orch._run_single = controlled_single
        await orch.run("task", mode="dynamic")

        assert plan_calls["n"] == 2
        assert run_log.count("a") == 1  # succeeded -> carried forward
        assert run_log.count("b") == 2  # failed -> re-run
        assert run_log.count("c") == 2  # failed -> re-run

    async def test_cached_output_feeds_downstream_rerun(self, mock_client, monkeypatch):
        from koboi.orchestration import planner

        async def plan(client, instruction, **kw):
            return PlanResult(needs_workflow=True, reason="multi", steps=[_step("a"), _step("b", ["a"])])

        monkeypatch.setattr(planner, "plan_or_skip", plan)

        client = mock_client(responses=[make_mock_response("synthesis")])
        orch = Orchestrator(client=client, router=_AllRouter([]), agents_map={}, default_mode="dynamic", max_replans=1)

        run_log: list[str] = []
        b_inputs: list[str] = []

        async def controlled_single(name, query_input):
            run_log.append(name)
            if name == "b":
                b_inputs.append(query_input)
                if len(b_inputs) == 1:
                    return _err("b")
            return _ok(name)

        orch._run_single = controlled_single
        await orch.run("task", mode="dynamic")

        # a was NOT re-run (cached) yet b's re-run input carries a's answer forward.
        assert run_log.count("a") == 1
        assert len(b_inputs) == 2
        assert "a-ok" in b_inputs[1]

    async def test_replan_reruns_stale_downstream(self, mock_client, monkeypatch):
        from koboi.orchestration import planner

        async def plan(client, instruction, **kw):
            return PlanResult(
                needs_workflow=True, reason="multi", steps=[_step("a"), _step("b", ["a"]), _step("c", ["b"])]
            )

        monkeypatch.setattr(planner, "plan_or_skip", plan)

        client = mock_client(responses=[make_mock_response("synthesis")])
        orch = Orchestrator(client=client, router=_AllRouter([]), agents_map={}, default_mode="dynamic", max_replans=1)

        run_log: list[str] = []

        async def controlled_single(name, query_input):
            run_log.append(name)
            # b fails; c "succeeds" but on b's degraded (error) input -> stale.
            if name == "b" and run_log.count("b") == 1:
                return _err("b")
            return _ok(name)

        orch._run_single = controlled_single
        await orch.run("task", mode="dynamic")

        # c succeeded round 1 but is downstream of failed b -> re-run (its success was stale).
        assert run_log.count("a") == 1
        assert run_log.count("b") == 2
        assert run_log.count("c") == 2

    async def test_run_single_flags_side_effecting_tool(self):
        # Real _run_single detection: a DESTRUCTIVE tool (flagged idempotent=True, the
        # builtin gap) must set had_non_idempotent_tool=True; a SAFE idempotent tool not.

        class _Agent:
            def __init__(self, tools, tool_calls):
                self.tools = tools
                self.memory = ConversationMemory()
                self._tool_calls = tool_calls

            async def run(self, query):
                return RunResult(content="ok", tool_calls_made=self._tool_calls)

        orch = Orchestrator(client=MockClient([]), router=_AllRouter([]), agents_map={}, default_mode="sequential")

        reg_danger = ToolRegistry()
        reg_danger.register(
            "danger",
            "destructive op",
            {"type": "object", "properties": {}},
            lambda: "boom",
            risk_level=RiskLevel.DESTRUCTIVE,
        )
        orch._agents_map["n"] = _Agent(reg_danger, [ToolCall(id="t1", name="danger", arguments="{}")])
        res = await orch._run_single("n", "q")
        assert res.had_non_idempotent_tool is True  # DESTRUCTIVE -> side-effecting (idempotent=True gap)

        reg_safe = ToolRegistry()
        reg_safe.register("calc", "safe op", {"type": "object", "properties": {}}, lambda: "1")
        orch._agents_map["n2"] = _Agent(reg_safe, [ToolCall(id="t2", name="calc", arguments="{}")])
        res2 = await orch._run_single("n2", "q")
        assert res2.had_non_idempotent_tool is False
