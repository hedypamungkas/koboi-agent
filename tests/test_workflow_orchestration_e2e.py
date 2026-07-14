"""E2e: orchestration multi-node cache→capture→replay (Scenario A, mock).

Proves the wedge works for MULTI-NODE orchestration (koboi's killer feature):
every LLM call across the entire DAG (router + each node + synthesis) is cached
and replayed byte-identical with ZERO live calls.
"""

import asyncio

import yaml

from koboi.facade import KoboiAgent
from koboi.llm.cache import CachedClient
from koboi.workflows import capture_from_run
from koboi.workflows.cache_sidecar import DirectoryCacheSidecar
from tests.conftest import MockClient, make_mock_response


def _orch_config(conditionals: bool = False) -> dict:
    agents = [
        {
            "name": "classify",
            "system_prompt": "Classify sentiment as POSITIVE or NEGATIVE. One word only.",
            "keywords": ["classify", "review"],
        },
        {
            "name": "praise",
            "system_prompt": "Praise the review in one sentence.",
            "keywords": ["praise"],
            "depends_on": ["classify"],
        },
    ]
    if conditionals:
        agents[0]["conditionals"] = [
            {"to": "praise", "when": {"contains": "POSITIVE"}},
            {"to": "critique", "when": {"contains": "NEGATIVE"}},
        ]
        agents.append(
            {
                "name": "critique",
                "system_prompt": "Critique the review in one sentence.",
                "keywords": ["critique"],
                "depends_on": ["classify"],
            }
        )
    return {
        "agent": {"name": "sentiment-orch", "system_prompt": "You orchestrate sentiment routing."},
        "llm": {"provider": "openai", "model": "mock-model", "api_key": "test"},
        "memory": {"backend": "in_memory"},
        "orchestration": {
            "enabled": True,
            "execution": {"mode": "dag", "full_graph": True},
            "router": {"type": "keyword"},
            "agents": agents,
        },
    }


def _swap_inner(agent: KoboiAgent, mock: MockClient) -> None:
    """Swap the orchestrator's CachedClient inner to a MockClient (keeps the cache wrapper)."""
    client = agent._orchestrator.client
    if isinstance(client, CachedClient):
        client._inner = mock
    else:
        agent._orchestrator.client = mock


class TestOrchestrationCacheCaptureReplay:
    """A1: every LLM call in a multi-node DAG is cached + replayed."""

    def test_two_node_dag_all_cached_replayed_byte_identical(self, tmp_path):
        cfg = _orch_config(conditionals=False)
        cache_dir = str(tmp_path / "cache")

        # 1. CACHE RUN: build orchestration agent in cache mode, inject MockClient
        agent = KoboiAgent.from_dict(cfg, replay_mode="cache", cache_dir=cache_dir)
        mock = MockClient(
            [
                make_mock_response("POSITIVE"),  # classify
                make_mock_response("Great product!"),  # praise
                make_mock_response("Summary: positive review"),  # synthesis
            ]
        )
        _swap_inner(agent, mock)
        result1 = asyncio.run(agent.run("Review: amazing product!"))
        assert mock.call_count >= 2  # classify + praise (+ synthesis)

        # 2. CAPTURE: freeze ALL responses
        config_text = yaml.safe_dump(cfg, sort_keys=False)
        _, entries = capture_from_run(config_text=config_text, name="orch-cap", with_cache=True, cache_dir=cache_dir)
        assert len(entries) >= 2

        # 3. REPLAY: fresh agent, fresh MockClient (would answer "WRONG"), RAISE on miss
        sidecar = str(tmp_path / "sidecar")
        DirectoryCacheSidecar(sidecar).write(entries)
        agent2 = KoboiAgent.from_dict(cfg, replay_mode="replay", cache_dir=sidecar)
        mock2 = MockClient([make_mock_response("WRONG")])
        _swap_inner(agent2, mock2)
        result2 = asyncio.run(agent2.run("Review: amazing product!"))
        assert mock2.call_count == 0  # all cache hits
        assert result2.content == result1.content  # byte-identical synthesis


class TestOrchestrationConditionalBranch:
    """A2: conditional branches (skipped nodes) stay skipped on replay."""

    def test_conditional_branch_skipped_consistently(self, tmp_path):
        cfg = _orch_config(conditionals=True)
        cache_dir = str(tmp_path / "cache")

        # CACHE RUN: "amazing" → classify=POSITIVE → praise runs, critique SKIPPED
        agent = KoboiAgent.from_dict(cfg, replay_mode="cache", cache_dir=cache_dir)
        mock = MockClient(
            [
                make_mock_response("POSITIVE"),  # classify
                make_mock_response("Great product!"),  # praise (critique skipped)
                make_mock_response("Summary: positive"),  # synthesis
            ]
        )
        _swap_inner(agent, mock)
        result1 = asyncio.run(agent.run("Review: amazing product!"))
        # critique was NOT called (conditional skip)
        assert mock.call_count == 3  # classify + praise + synthesis (no critique)

        # CAPTURE
        config_text = yaml.safe_dump(cfg, sort_keys=False)
        _, entries = capture_from_run(config_text=config_text, name="cond-cap", with_cache=True, cache_dir=cache_dir)

        # REPLAY: same input → classify=POSITIVE (from cache) → praise (from cache),
        # critique still skipped (same conditional evaluation)
        sidecar = str(tmp_path / "sidecar")
        DirectoryCacheSidecar(sidecar).write(entries)
        agent2 = KoboiAgent.from_dict(cfg, replay_mode="replay", cache_dir=sidecar)
        mock2 = MockClient([make_mock_response("WRONG")])
        _swap_inner(agent2, mock2)
        result2 = asyncio.run(agent2.run("Review: amazing product!"))
        assert mock2.call_count == 0  # all hits
        assert result2.content == result1.content  # same path, same output
