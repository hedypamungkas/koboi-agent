"""End-to-end demonstration of the Deterministic Workflow feature (v1 + v2 + v3).

Walks the full real-world arc with a MockClient (no API key):

    config --cache run--> captured bundle (+cache sidecar) --replay--> byte-identical, 0 live calls

Covers:
  +  happy path  -- the wedge: cache run -> capture -> offline replay (0 calls, identical)
  -  negative    -- a divergent input on replay raises CacheMissError (honest, no silent live call)
  *  edge cases  -- empty cache + replay; export round-trip preserves determinism;
                    redact-cache marks provenance; sentinel downgrade; no-sidecar run
"""

import asyncio

import pytest

from koboi.config import Config
from koboi.llm.cache import CachedClient, CacheMissError, CacheMissPolicy, ResponseCache
from koboi.loop import AgentCore
from koboi.types import RunResult
from koboi.workflows import (
    DeterminismProfile,
    build_from_config_path,
    capture_from_run,
    prepare_captured_bundle,
    validate_workflow,
)
from koboi.workflows.cache_sidecar import DirectoryCacheSidecar
from tests.conftest import MockClient, make_mock_response

QUESTION = "What is the capital of France?"
ANSWER = "Paris"
CONFIG_TEXT = (
    "agent:\n  name: geo\n  system_prompt: 'You are a concise geography assistant.'\n"
    "llm:\n  provider: openai\n  model: mock-model\n  api_key: ${OPENAI_API_KEY:}\n"
)


def _run(core: AgentCore, message: str) -> RunResult:
    return asyncio.run(core.run(message))


def _cache_run_agent(cache_dir, answer=ANSWER) -> tuple[AgentCore, MockClient]:
    """An AgentCore whose client is a MockClient wrapped in a STORE CachedClient."""
    mock = MockClient([make_mock_response(content=answer)])
    core = AgentCore(client=CachedClient(mock, ResponseCache(cache_dir)))
    return core, mock


def _replay_agent(cache_dir) -> tuple[AgentCore, MockClient]:
    """An AgentCore whose client is a FRESH MockClient wrapped in a RAISE CachedClient
    (offline -- a miss raises instead of calling the model)."""
    mock = MockClient([make_mock_response(content="SHOULD-NOT-BE-CALLED")])
    core = AgentCore(client=CachedClient(mock, ResponseCache(cache_dir), on_miss=CacheMissPolicy.RAISE))
    return core, mock


# --------------------------------------------------------------------------- #
# +  HAPPY PATH -- the wedge
# --------------------------------------------------------------------------- #
class TestEndToEndHappyPath:
    def test_cache_run_capture_offline_replay_is_byte_identical(self, tmp_path):
        # 1. cache-mode run: the MockClient is called once, response is cached
        cache_dir = tmp_path / "run_cache"
        core1, mock1 = _cache_run_agent(cache_dir)
        result1 = _run(core1, QUESTION)
        assert mock1.call_count == 1
        assert result1.content == ANSWER

        # 2. capture: freeze the run's cache into a sidecar
        wd, entries = capture_from_run(
            config_text=CONFIG_TEXT, name="geo-cap", with_cache=True, cache_dir=str(cache_dir)
        )
        assert entries is not None and len(entries) >= 1
        assert wd.provenance.with_cache is True

        # 3. hydrate the sidecar into a fresh cache dir (simulates importing the bundle elsewhere)
        sidecar_dir = tmp_path / "sidecar"
        DirectoryCacheSidecar(sidecar_dir).write(entries)

        # 4. offline replay: a FRESH MockClient (would return the wrong answer if called),
        #    wrapped in a RAISE CachedClient over the sidecar -> every response is a hit
        core2, mock2 = _replay_agent(sidecar_dir)
        result2 = _run(core2, QUESTION)
        assert mock2.call_count == 0  # fully offline -- the model was never called
        assert result2.content == ANSWER  # byte-identical to the original run

    def test_prepare_captured_bundle_points_at_sidecar_for_rerun(self, tmp_path):
        # The unifying helper injects replay.mode=cache + cache_dir into the bundle
        # so a re-run (koboi run --workflow) loads the sidecar.
        wd, _ = capture_from_run(config_text=CONFIG_TEXT, name="x")
        bundle = wd.to_bundle_yaml()
        prepared = prepare_captured_bundle(bundle, cache_dir="/tmp/sidecar")
        import yaml as _y

        data = _y.safe_load(prepared)
        assert data["replay"]["mode"] == "cache"
        assert data["replay"]["cache_dir"] == "/tmp/sidecar"


# --------------------------------------------------------------------------- #
# -  NEGATIVE -- divergent input on a replay fails honestly
# --------------------------------------------------------------------------- #
class TestEndToEndNegative:
    def test_divergent_input_on_replay_raises(self, tmp_path):
        # capture a run for QUESTION, then replay a DIFFERENT question -> cache miss -> RAISE
        cache_dir = tmp_path / "run_cache"
        core1, _ = _cache_run_agent(cache_dir)
        _run(core1, QUESTION)
        wd, entries = capture_from_run(
            config_text=CONFIG_TEXT, name="geo-cap", with_cache=True, cache_dir=str(cache_dir)
        )
        sidecar_dir = tmp_path / "sidecar"
        DirectoryCacheSidecar(sidecar_dir).write(entries)

        core2, mock2 = _replay_agent(sidecar_dir)
        with pytest.raises(CacheMissError):
            _run(core2, "What is the capital of Germany?")  # not cached -> honest failure
        assert mock2.call_count == 0  # never fell back to a live call


# --------------------------------------------------------------------------- #
# *  EDGE CASES
# --------------------------------------------------------------------------- #
class TestEdgeCases:
    def test_empty_cache_replay_raises_immediately(self, tmp_path):
        # replay with NO populated cache -> immediate miss (no silent live call)
        core, mock = _replay_agent(tmp_path / "empty")
        with pytest.raises(CacheMissError):
            _run(core, QUESTION)
        assert mock.call_count == 0

    def test_redact_cache_marks_provenance_and_masks_secret(self, tmp_path):
        cache_dir = tmp_path / "c"
        core, _ = _cache_run_agent(cache_dir, answer="sk-live-supersecretkey1234567890abcd")
        _run(core, QUESTION)
        wd, entries = capture_from_run(
            config_text=CONFIG_TEXT, name="x", with_cache=True, cache_dir=str(cache_dir), redact_cache=True
        )
        assert wd.provenance.cache_redacted is True
        assert "sk-live" not in entries[0][1]["response"]["content"]

    def test_export_round_trip_preserves_determinism(self, tmp_path):
        cfg_path = tmp_path / "geo.yaml"
        cfg_path.write_text(
            CONFIG_TEXT + "orchestration:\n  enabled: true\n  determinism:\n    temperature: 0.0\n    seed: 7\n",
            encoding="utf-8",
        )
        wd = build_from_config_path(cfg_path, name="geo")
        bundle = wd.to_bundle_yaml()
        # round-trip through YAML
        wd2 = capture_from_run(config_text=bundle, name="geo2")[0]
        det = wd2.determinism
        assert det is not None and det.temperature == 0.0 and det.seed == 7

    def test_sentinel_node_downgrades_workflow_cache_to_live(self):
        wf = DeterminismProfile(replay_mode="cache")
        node = DeterminismProfile(replay_mode="live")
        assert wf.merge(node).replay_mode == "live"  # explicit downgrade works
        assert wf.merge(DeterminismProfile(replay_mode=None)).replay_mode == "cache"  # None = inherit

    def test_validate_workflow_warns_on_unpinned_model(self, tmp_path):
        cfg_with_det = CONFIG_TEXT + "orchestration:\n  enabled: true\n  determinism:\n    temperature: 0.0\n"
        wd, _ = capture_from_run(config_text=cfg_with_det, name="x")
        warnings = validate_workflow(wd)
        assert any("model_pin" in w for w in warnings)  # determinism set, no model_pin -> drift warning

    def test_bundle_is_rerunnable_as_config(self, tmp_path):
        # the captured bundle (envelope popped) loads as a valid Config
        wd, _ = capture_from_run(config_text=CONFIG_TEXT, name="x")
        cfg = Config.from_string(wd.to_bundle_yaml())
        assert cfg.raw["agent"]["name"] == "geo"

    def test_cache_mode_live_run_does_not_wrap(self, tmp_path):
        # a live-mode run (default) does NOT cache -- the inner client is called every time
        mock = MockClient([make_mock_response(content="a"), make_mock_response(content="b")])
        core = AgentCore(client=mock)  # no CachedClient wrapper
        r1 = _run(core, "q")
        r2 = _run(core, "q")
        assert mock.call_count == 2  # both live, no memoization
        assert r1.content == "a" and r2.content == "b"
