"""tests/test_planner.py -- WS3: planner self-triage, extraction, fail-safes."""

from __future__ import annotations

import json

from koboi.orchestration.planner import plan_or_skip
from koboi.types import AgentResponse

PLAN_COMPLEX = json.dumps(
    {
        "needs_workflow": True,
        "reason": "multi-step",
        "steps": [
            {"id": "research", "instruction": "Research", "depends_on": []},
            {"id": "draft", "instruction": "Draft", "depends_on": ["research"]},
        ],
    }
)
PLAN_SIMPLE = json.dumps({"needs_workflow": False, "reason": "simple", "steps": []})


class _FakeClient:
    def __init__(self, content):
        self._content = content

    async def complete(self, messages, tools=None, response_format=None):
        return AgentResponse(content=self._content)

    async def get_embeddings(self, text):
        return None


class _ErrClient:
    async def complete(self, messages, tools=None, response_format=None):
        raise RuntimeError("boom")

    async def get_embeddings(self, text):
        return None


async def test_planner_extracts_complex():
    r = await plan_or_skip(_FakeClient(PLAN_COMPLEX), "do X then Y")
    assert r.needs_workflow is True
    assert [s.id for s in r.steps] == ["research", "draft"]
    assert r.deps == {"research": [], "draft": ["research"]}


async def test_planner_skips_simple():
    r = await plan_or_skip(_FakeClient(PLAN_SIMPLE), "what is 2+2")
    assert r.needs_workflow is False
    assert r.steps == []


async def test_planner_cycle_falls_back():
    cyclic = json.dumps(
        {
            "needs_workflow": True,
            "reason": "x",
            "steps": [
                {"id": "a", "instruction": "a", "depends_on": ["b"]},
                {"id": "b", "instruction": "b", "depends_on": ["a"]},
            ],
        }
    )
    r = await plan_or_skip(_FakeClient(cyclic), "x")
    assert r.needs_workflow is False
    assert "cyclic" in r.reason


async def test_planner_empty_falls_back():
    r = await plan_or_skip(_FakeClient("{}"), "x")
    assert r.needs_workflow is False


async def test_planner_malformed_falls_back():
    r = await plan_or_skip(_FakeClient("not json at all"), "x")
    assert r.needs_workflow is False


async def test_planner_client_error_falls_back():
    r = await plan_or_skip(_ErrClient(), "x")
    assert r.needs_workflow is False
    assert "error" in r.reason


async def test_planner_max_steps_cap():
    steps = [
        {"id": f"s{i}", "instruction": f"step {i}", "depends_on": ([f"s{i - 1}"] if i > 0 else [])} for i in range(20)
    ]
    big = json.dumps({"needs_workflow": True, "reason": "big", "steps": steps})
    r = await plan_or_skip(_FakeClient(big), "x", max_steps=5)
    assert r.needs_workflow is True
    assert len(r.steps) == 5
