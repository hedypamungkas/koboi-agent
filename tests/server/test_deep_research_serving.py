"""HIGH-1 + HIGH-2: Server interactive guard + jobs middle-path for orchestrated configs (core=None).

Tests that KoboiAgent(core=None, orchestrator=...) -- the deep_research serving path --
doesn't crash on _core access. The server's _run_agent closure wraps agent.run_stream(); if
that works with core=None, the server works (the guards just skip _core setup). Also tests
the config-level sandbox check for the jobs middle-path.
"""

from __future__ import annotations

import asyncio
import json


from koboi.events import OrchestrationCompleteEvent
from koboi.orchestration.dag_scheduler import DagScheduler
from koboi.orchestration.orchestrator import Orchestrator
from koboi.orchestration.router import KeywordRouter
from koboi.types import AgentResponse


class _FakeClient:
    """Minimal fake LLM client for the simple-request fallback path."""

    def __init__(self) -> None:
        self.model = "fake-model"
        self.provider = "fake"

    async def complete(self, messages, tools=None, response_format=None):
        text = " ".join(m.get("content", "") for m in messages)
        if "research planner" in text:
            return AgentResponse(
                content=json.dumps({"needs_workflow": False, "reason": "simple", "steps": []}),
                tool_calls=[],
            )
        return AgentResponse(content="Direct answer for the query.", tool_calls=[])


class TestHigh1ServerInteractiveGuard:
    """core=None agent (orchestrated) can run_stream + run + chat without _core crash."""

    async def test_run_stream_with_core_none(self, tmp_path):
        from koboi.facade import KoboiAgent

        orch = Orchestrator(
            client=_FakeClient(),
            router=KeywordRouter(),
            research={"max_depth": 1, "coverage_threshold": 0.7},
            dag_scheduler=DagScheduler(agents_map={}, deps={}, db_path=str(tmp_path / "r.db")),
            default_mode="deep_research",
        )
        agent = KoboiAgent(core=None, orchestrator=orch)
        events = [e async for e in agent.run_stream("hello")]
        assert any(isinstance(e, OrchestrationCompleteEvent) for e in events)
        assert agent._core is None  # confirms the guard path, not a fallback to _core

    async def test_run_with_core_none(self, tmp_path):
        from koboi.facade import KoboiAgent

        orch = Orchestrator(
            client=_FakeClient(),
            router=KeywordRouter(),
            research={"max_depth": 1, "coverage_threshold": 0.7},
            dag_scheduler=DagScheduler(agents_map={}, deps={}, db_path=str(tmp_path / "r.db")),
            default_mode="deep_research",
        )
        agent = KoboiAgent(core=None, orchestrator=orch)
        result = await agent.run("hello")
        assert result.content  # some answer
        assert result.metadata.get("execution_mode") == "deep_research"


class TestHigh2JobsMiddlePath:
    """Config-level sandbox check for orchestrated autonomous jobs (core=None)."""

    def test_config_level_sandbox_check_restricted(self):
        from koboi.config import Config

        config = Config.from_dict(
            {
                "agent": {"name": "dr"},
                "llm": {"model": "fake", "api_key": "k"},
                "sandbox": {"backend": "restricted"},
            }
        )
        backend = config.get("sandbox", "backend", default="passthrough")
        assert backend == "restricted"  # passes the jobs middle-path check

    def test_config_level_sandbox_check_passthrough(self):
        from koboi.config import Config

        config = Config.from_dict(
            {
                "agent": {"name": "dr"},
                "llm": {"model": "fake", "api_key": "k"},
                "sandbox": {"backend": "passthrough"},
            }
        )
        backend = config.get("sandbox", "backend", default="passthrough")
        assert backend == "passthrough"  # would trigger PermissionError in _execute_job

    def test_config_default_is_passthrough(self):
        from koboi.config import Config

        config = Config.from_dict(
            {
                "agent": {"name": "dr"},
                "llm": {"model": "fake", "api_key": "k"},
            }
        )
        backend = config.get("sandbox", "backend", default="passthrough")
        assert backend == "passthrough"  # no sandbox section -> default -> refused for jobs

    async def test_execute_job_captures_orchestration_complete(self, tmp_path):
        # Regression: deep_research jobs emitted OrchestrationCompleteEvent (NOT
        # CompleteEvent), so the middle path's isinstance check missed it and the job's
        # result_json stayed empty. _execute_job must capture final_answer now.
        from koboi.config import Config
        from koboi.facade import KoboiAgent
        from koboi.server.jobs import JobRegistry, JobStore, _execute_job
        from koboi.server.pool import AgentPool

        class _Fake:
            model = "fake-model"
            provider = "fake"

            async def complete(self, messages, tools=None, response_format=None):
                text = " ".join(m.get("content", "") for m in messages)
                if "research planner" in text:
                    return AgentResponse(
                        content=json.dumps({"needs_workflow": False, "reason": "simple", "steps": []}),
                        tool_calls=[],
                    )
                return AgentResponse(content="Direct answer.", tool_calls=[])

        orch = Orchestrator(
            client=_Fake(),
            router=KeywordRouter(),
            research={"max_depth": 1, "coverage_threshold": 0.7},
            dag_scheduler=DagScheduler(agents_map={}, deps={}, db_path=str(tmp_path / "r.db")),
            default_mode="deep_research",
            session_id="job-sess",
        )
        agent = KoboiAgent(core=None, orchestrator=orch)
        config = Config.from_dict(
            {
                "agent": {"name": "dr"},
                "llm": {"model": "fake", "api_key": "k"},
                "sandbox": {"backend": "restricted"},
            }
        )
        agent._config = config  # the jobs middle path reads sandbox backend from here
        pool = AgentPool(config)
        pool._agents["job-sess"] = agent  # pre-pool the orchestrated agent
        pool._locks["job-sess"] = asyncio.Lock()  # session_lock expects this to exist
        registry = JobRegistry()
        registry.register("job-1", "job-sess", owner="dev")
        store = JobStore(str(tmp_path / "jobs.db"))

        content, needs_clarification = await _execute_job(
            "job-1", pool, registry, store, "hello", mode=None, max_iterations=None, resume=False
        )
        assert content  # non-empty -- the report was captured from OrchestrationCompleteEvent
        assert "Direct answer" in content
        assert needs_clarification is False
        await pool.close_all()
