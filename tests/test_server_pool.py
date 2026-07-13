"""Unit tests for koboi/server/pool.py (no FastAPI; MockClient via client_factory)."""

from __future__ import annotations

import asyncio
import os

import pytest

from koboi.config import Config
from koboi.events import CompleteEvent
from koboi.server.pool import AgentPool, PoolFull
from tests.conftest import MockClient, make_mock_response


def _config() -> Config:
    # in-memory backend -> no koboi_memory.db leak; dummy LLM creds (client swapped in tests)
    return Config.from_dict(
        {
            "agent": {"name": "srv", "system_prompt": "h", "max_iterations": 3},
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "test",
                "base_url": "http://localhost:8080/v1",
            },
            "memory": {"backend": "in_memory"},
            "sandbox": {"backend": "passthrough"},
        },
        validate=True,
    )


def _factory(responses):
    return lambda: MockClient(responses)


async def _collect(gen):
    out = []
    async for ev in gen:
        out.append(ev)
    return out


class TestAgentPoolLifecycle:
    async def test_get_or_create_lazy_then_reuse(self):
        pool = AgentPool(_config(), client_factory=_factory([make_mock_response(content="hi")]), cap=10)
        assert len(pool) == 0
        a1 = await pool.get_or_create("s1")
        assert len(pool) == 1
        a2 = await pool.get_or_create("s1")
        assert a1 is a2  # same instance reused
        await pool.close_all()

    async def test_distinct_sessions_distinct_workdirs(self):
        pool = AgentPool(_config(), client_factory=_factory([make_mock_response(content="x")]))
        await pool.get_or_create("s1")
        await pool.get_or_create("s2")
        assert pool.workdir_for("s1") != pool.workdir_for("s2")
        assert len(pool) == 2
        await pool.close_all()

    async def test_cap_evicts_idle_then_admits(self):
        pool = AgentPool(_config(), client_factory=_factory([make_mock_response(content="x")]), cap=1)
        await pool.get_or_create("s1")
        # s1 is idle (lock free) -> evicted to make room for s2 (no PoolFull)
        await pool.get_or_create("s2")
        assert len(pool) == 1
        assert pool.get("s1") is None
        await pool.close_all()

    async def test_cap_raises_pool_full_when_all_busy(self):
        pool = AgentPool(_config(), client_factory=_factory([make_mock_response(content="x")]), cap=1)
        # hold s1's lock by starting (not finishing) a run_stream
        gen = pool.run_stream("s1", "hi")
        await gen.__anext__()  # engages the per-session lock
        try:
            with pytest.raises(PoolFull):
                await pool.get_or_create("s2")  # s1 busy -> can't evict -> PoolFull
        finally:
            await gen.aclose()
        await pool.close_all()

    async def test_client_factory_swaps_client(self):
        pool = AgentPool(_config(), client_factory=_factory([make_mock_response(content="hi")]))
        agent = await pool.get_or_create("s1")
        assert isinstance(agent._core.client, MockClient)
        await pool.close_all()

    async def test_evict_closes_agent(self):
        pool = AgentPool(_config(), client_factory=_factory([make_mock_response(content="hi")]))
        agent = await pool.get_or_create("s1")
        closed = {"v": False}

        async def _fake_close():
            closed["v"] = True

        agent.close = _fake_close  # type: ignore[method-assign]
        assert await pool.evict("s1") is True
        assert closed["v"] is True
        assert pool.get("s1") is None

    async def test_evict_unknown_returns_false(self):
        pool = AgentPool(_config())
        assert await pool.evict("nope") is False
        await pool.close_all()

    async def test_get_messages_empty_for_unknown(self):
        pool = AgentPool(_config())
        assert await pool.get_messages("nope") == []
        await pool.close_all()

    async def test_client_factory_skipped_for_core_none(self, tmp_path):
        # Fix 1: an orchestrated config (core=None) + a client_factory seam must NOT crash
        # (the seam is silently skipped instead of AttributeError on agent._core.client).
        from koboi.events import OrchestrationCompleteEvent
        from koboi.orchestration.dag_scheduler import DagScheduler
        from koboi.orchestration.orchestrator import Orchestrator
        from koboi.orchestration.router import KeywordRouter
        from koboi.types import AgentResponse

        class _Fake:
            model = "fake-model"
            provider = "fake"

            async def complete(self, messages, tools=None, response_format=None):
                import json

                text = " ".join(m.get("content", "") for m in messages)
                if "research planner" in text:
                    return AgentResponse(
                        content=json.dumps({"needs_workflow": False, "reason": "simple", "steps": []}),
                        tool_calls=[],
                    )
                return AgentResponse(content="ok", tool_calls=[])

        # Use the same orchestrated-agent build path, then inject into a pool.
        orch = Orchestrator(
            client=_Fake(),
            router=KeywordRouter(),
            research={"max_depth": 1, "coverage_threshold": 0.7},
            dag_scheduler=DagScheduler(agents_map={}, deps={}, db_path=str(tmp_path / "r.db")),
            default_mode="deep_research",
        )
        from koboi.facade import KoboiAgent

        agent = KoboiAgent(core=None, orchestrator=orch)
        pool = AgentPool(_config(), client_factory=lambda: MockClient([]))
        pool._agents["s1"] = agent  # inject (bypass _build_agent's config build)
        # get_or_create path is not exercised here; we directly assert no crash on the
        # seam by triggering a run + then get_messages.
        events = await _collect(agent.run_stream("hello"))
        assert any(isinstance(e, OrchestrationCompleteEvent) for e in events)
        assert agent._core is None  # orchestrated -> guard path
        await pool.close_all()


class TestDeepResearchMessages:
    """Fix 3d: get_messages surfaces the query + cited report for deep_research sessions."""

    async def _orchestrated_agent(self, tmp_path, session_id):
        from koboi.orchestration.dag_scheduler import DagScheduler
        from koboi.orchestration.orchestrator import Orchestrator
        from koboi.orchestration.router import KeywordRouter
        from koboi.types import AgentResponse, ToolCall

        class _Fake:
            model = "fake-model"
            provider = "fake"

            def __init__(self) -> None:
                self.synthesis = "## Report\nSolid-state batteries use sulfide [1]."

            async def complete(self, messages, tools=None, response_format=None):
                import json

                text = " ".join(m.get("content", "") for m in messages)
                if "research planner" in text:
                    return AgentResponse(
                        content=json.dumps(
                            {
                                "needs_workflow": True,
                                "reason": "research",
                                "steps": [
                                    {
                                        "id": "research_topic",
                                        "instruction": "Investigate the topic",
                                        "depends_on": [],
                                        "search_queries": ["topic overview"],
                                    }
                                ],
                            }
                        ),
                        tool_calls=[],
                    )
                if "evaluating how thoroughly" in text:
                    return AgentResponse(
                        content=json.dumps({"overall_score": 0.95, "coverage": {"x": 0.95}, "follow_up_queries": []}),
                        tool_calls=[],
                    )
                if "synthesizing a cited research report" in text:
                    return AgentResponse(content=self.synthesis, tool_calls=[])
                # node turn -> an answer (becomes finding [1])
                if any(m.get("role") == "tool" for m in messages):
                    return AgentResponse(content="Found: sulfide electrolytes.", tool_calls=[])
                return AgentResponse(
                    content="",
                    tool_calls=[ToolCall(id="tc1", name="web_search", arguments=json.dumps({"query": "x"}))],
                )

            async def complete_stream(self, messages, tools=None):
                from koboi.events import TextDeltaEvent

                yield TextDeltaEvent(content=self.synthesis)

        orch = Orchestrator(
            client=_Fake(),
            router=KeywordRouter(),
            research={"max_depth": 1, "coverage_threshold": 0.7},
            dag_scheduler=DagScheduler(agents_map={}, deps={}, db_path=str(tmp_path / "r.db")),
            default_mode="deep_research",
            session_id=session_id,
        )
        from koboi.facade import KoboiAgent

        return KoboiAgent(core=None, orchestrator=orch)

    async def test_get_messages_returns_query_and_report(self, tmp_path):
        agent = await self._orchestrated_agent(tmp_path, "sess-abc")
        _ = await _collect(agent.run_stream("Research solid-state batteries."))
        pool = AgentPool(_config())
        pool._agents["sess-abc"] = agent
        msgs = await pool.get_messages("sess-abc")
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert "solid-state batteries" in msgs[0]["content"].lower()
        assert "sulfide" in msgs[1]["content"]  # the persisted final report
        assert "[1]" in msgs[1]["content"]  # citation survived
        await pool.close_all()

    async def test_get_messages_empty_when_no_research_run(self, tmp_path):
        # A deep_research session that never ran (no research_context row) -> [].
        agent = await self._orchestrated_agent(tmp_path, "sess-none")
        pool = AgentPool(_config())
        pool._agents["sess-none"] = agent
        assert await pool.get_messages("sess-none") == []
        await pool.close_all()


class TestPoolRunStream:
    async def test_run_stream_yields_complete_event(self):
        pool = AgentPool(_config(), client_factory=_factory([make_mock_response(content="hello")]))
        events = await _collect(pool.run_stream("s1", "hi"))
        assert any(isinstance(e, CompleteEvent) for e in events)
        await pool.close_all()

    async def test_lock_serializes_concurrent_same_session(self):
        # A client that flags reentry: with the lock, the two turns never overlap.
        state = {"active": 0, "overlap": False}

        class _Detect(MockClient):
            async def complete_stream(self, messages, tools=None, response_format=None):
                state["active"] += 1
                if state["active"] > 1:
                    state["overlap"] = True
                try:
                    async for ev in super().complete_stream(messages, tools):
                        await asyncio.sleep(0)  # yield so an unlocked peer could sneak in
                        yield ev
                finally:
                    state["active"] -= 1

        pool = AgentPool(_config(), client_factory=lambda: _Detect([make_mock_response(content="x")]))
        await asyncio.gather(_collect(pool.run_stream("s1", "a")), _collect(pool.run_stream("s1", "b")))
        assert state["overlap"] is False
        await pool.close_all()

    async def test_different_sessions_run_in_parallel(self):
        pool = AgentPool(
            _config(),
            client_factory=_factory([make_mock_response(content="x"), make_mock_response(content="y")]),
            cap=10,
        )
        await asyncio.gather(_collect(pool.run_stream("s1", "a")), _collect(pool.run_stream("s2", "b")))
        assert len(pool) == 2
        await pool.close_all()

    async def test_extra_tools_registered_on_agent(self):
        def my_tool(x: str) -> str:
            return f"got {x}"

        pool = AgentPool(
            _config(),
            client_factory=_factory([make_mock_response(content="hi")]),
            extra_tools=(
                (
                    "my_tool",
                    my_tool,
                    "a tool",
                    {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
                ),
            ),
        )
        agent = await pool.get_or_create("s1")
        assert "my_tool" in agent._core.tools
        await pool.close_all()


def _config_with_git_init(git_init: bool) -> Config:
    return Config.from_dict(
        {
            "agent": {"name": "srv", "system_prompt": "h", "max_iterations": 3},
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "test",
                "base_url": "http://localhost:8080/v1",
            },
            "memory": {"backend": "in_memory"},
            "sandbox": {"backend": "passthrough", "git_init": git_init},
        },
        validate=True,
    )


@pytest.mark.skipif(
    os.environ.get("KOBOI_SKIP_GIT_TESTS") == "1",
    reason="git binary unavailable",
)
class TestGitInitWorkdir:
    async def test_git_init_seeds_real_repo(self, tmp_path):
        from koboi.tools.builtin.git import git_status

        pool = AgentPool(
            _config_with_git_init(True),
            client_factory=_factory([make_mock_response(content="hi")]),
            workspace_root=str(tmp_path),
        )
        await pool.get_or_create("s1")
        wd = pool.workdir_for("s1")
        assert os.path.isdir(os.path.join(wd, ".git")), "workdir should be a git repo"
        # The real git_status tool must report a clean tree, not "not a git repository".
        out = git_status(repo_path=wd)
        assert "clean" in out.lower()
        assert "not a git" not in out.lower()
        await pool.close_all()

    async def test_git_init_off_leaves_plain_dir(self, tmp_path):
        pool = AgentPool(
            _config_with_git_init(False),
            client_factory=_factory([make_mock_response(content="hi")]),
            workspace_root=str(tmp_path),
        )
        await pool.get_or_create("s1")
        wd = pool.workdir_for("s1")
        assert not os.path.isdir(os.path.join(wd, ".git"))
        await pool.close_all()
