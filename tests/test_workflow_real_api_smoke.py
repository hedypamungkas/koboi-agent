"""Real-API smoke tests: cache→capture→replay with a LIVE LLM (Scenario C).

Env-gated: skipped unless OPENAI_API_KEY is set. Costs ~1-4 API calls (cache run
only; replay = 0 calls). Proves the wedge against a REAL provider (not just a mock):
real serialization, real cache key stability, real byte-identical replay.

Run: OPENAI_API_KEY=sk-... pytest tests/test_workflow_real_api_smoke.py -v -s
"""

import asyncio
import os

import pytest

from koboi.config import Config
from koboi.facade import KoboiAgent
from koboi.workflows import capture_from_run
from koboi.workflows.cache_sidecar import DirectoryCacheSidecar

_LIVE = pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY (real-API smoke test)")

_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
_QUESTION = "What is 2+2? Answer with just the number."
_ORCH_QUESTION = "Review: 'This product is amazing!' Reply with one word."


def _llm():
    llm = {"provider": "openai", "model": _MODEL, "api_key": os.environ["OPENAI_API_KEY"]}
    if os.environ.get("OPENAI_BASE_URL"):
        llm["base_url"] = os.environ["OPENAI_BASE_URL"]
    return llm


def _single_agent_cfg():
    return Config.from_dict(
        {
            "agent": {"name": "smoke", "system_prompt": "You answer concisely."},
            "llm": _llm(),
            "memory": {"backend": "in_memory"},
        }
    )


def _orch_cfg():
    return Config.from_dict(
        {
            "agent": {"name": "orch-smoke", "system_prompt": "You orchestrate."},
            "llm": _llm(),
            "memory": {"backend": "in_memory"},
            "orchestration": {
                "enabled": True,
                "execution": {"mode": "dag", "full_graph": True},
                "router": {"type": "keyword"},
                "agents": [
                    {
                        "name": "classify",
                        "system_prompt": "Classify as POSITIVE or NEGATIVE. One word.",
                        "keywords": ["classify", "review"],
                    },
                    {
                        "name": "praise",
                        "system_prompt": "Reply with one positive word.",
                        "keywords": ["praise"],
                        "depends_on": ["classify"],
                    },
                ],
            },
        }
    )


@_LIVE
class TestRealApiSingleAgentSmoke:
    """C1: single-agent cache→capture→replay with a real LLM."""

    def test_byte_identical_replay(self, tmp_path):
        cfg = _single_agent_cfg()
        cache_dir = str(tmp_path / "cache")

        # 1. CACHE RUN: real LLM call, response memoized
        agent = KoboiAgent._from_config(cfg, replay_mode="cache", cache_dir=cache_dir)
        result1 = asyncio.run(agent.run(_QUESTION))
        assert result1.content  # got a real response

        # 2. CAPTURE: freeze the cache
        _, entries = capture_from_run(config_text=cfg.to_yaml(), name="smoke", with_cache=True, cache_dir=cache_dir)
        assert len(entries) >= 1

        # 3. REPLAY: fresh agent, RAISE on miss → 0 live calls, byte-identical
        sidecar = str(tmp_path / "sidecar")
        DirectoryCacheSidecar(sidecar).write(entries)
        agent2 = KoboiAgent._from_config(cfg, replay_mode="replay", cache_dir=sidecar)
        result2 = asyncio.run(agent2.run(_QUESTION))
        assert result2.content == result1.content  # byte-identical


@_LIVE
class TestRealApiOrchestrationSmoke:
    """C2: orchestration multi-node cache→capture→replay with a real LLM."""

    def test_multi_node_byte_identical_replay(self, tmp_path):
        cfg = _orch_cfg()
        cache_dir = str(tmp_path / "cache")

        # 1. CACHE RUN: multiple real LLM calls (classify + praise + synthesis)
        agent = KoboiAgent._from_config(cfg, replay_mode="cache", cache_dir=cache_dir)
        result1 = asyncio.run(agent.run(_ORCH_QUESTION))
        assert result1.content  # got a real synthesis

        # 2. CAPTURE
        _, entries = capture_from_run(
            config_text=cfg.to_yaml(), name="orch-smoke", with_cache=True, cache_dir=cache_dir
        )
        assert len(entries) >= 2  # at least classify + praise (+ synthesis)

        # 3. REPLAY: 0 live calls, byte-identical synthesis
        sidecar = str(tmp_path / "sidecar")
        DirectoryCacheSidecar(sidecar).write(entries)
        agent2 = KoboiAgent._from_config(cfg, replay_mode="replay", cache_dir=sidecar)
        result2 = asyncio.run(agent2.run(_ORCH_QUESTION))
        assert result2.content == result1.content  # byte-identical
