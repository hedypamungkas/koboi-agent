"""Tests for koboi.workflows.capture (v2 step 7): capture pipeline + offline replay."""

import asyncio

from koboi.config import Config
from koboi.llm.cache import CachedClient, CacheMissPolicy, ResponseCache
from koboi.types import AgentResponse, TokenUsage
from koboi.workflows import capture_from_run, prepare_captured_bundle, validate_capture
from koboi.workflows.cache_sidecar import DirectoryCacheSidecar
from tests.conftest import MockClient

CONFIG_TEXT = "agent:\n  name: x\nllm:\n  provider: openai\n  model: m\n  api_key: ${OPENAI_API_KEY:}\n"


def _populate_cache(cache_dir, content="answer"):
    inner = MockClient([AgentResponse(content=content, usage=TokenUsage())])
    client = CachedClient(inner, ResponseCache(cache_dir))
    asyncio.run(client.complete([{"role": "user", "content": "hi"}]))
    return inner


class TestCaptureFromRun:
    def test_sets_provenance(self, tmp_path):
        wd, entries = capture_from_run(
            config_text=CONFIG_TEXT, name="cap", source_run_id="job_123", source_session_id="sess_456"
        )
        assert wd.name == "cap"
        assert wd.provenance.source_run_id == "job_123"
        assert wd.provenance.source_session_id == "sess_456"
        assert wd.provenance.with_cache is False
        assert entries is None

    def test_with_cache_freezes_entries(self, tmp_path):
        _populate_cache(tmp_path / "run_cache")
        wd, entries = capture_from_run(
            config_text=CONFIG_TEXT, name="cap", with_cache=True, cache_dir=str(tmp_path / "run_cache")
        )
        assert entries is not None and len(entries) == 1
        assert wd.provenance.with_cache is True
        assert wd.provenance.cache_entries == 1

    def test_redact_cache_masks_secret_content(self, tmp_path):
        cache_dir = tmp_path / "run_cache"
        inner = MockClient([AgentResponse(content="sk-live-supersecretkey1234567890abcd", usage=TokenUsage())])
        asyncio.run(CachedClient(inner, ResponseCache(cache_dir)).complete([{"role": "user", "content": "hi"}]))
        wd, entries = capture_from_run(
            config_text=CONFIG_TEXT, name="cap", with_cache=True, cache_dir=str(cache_dir), redact_cache=True
        )
        assert entries is not None
        payload = entries[0][1]
        assert "sk-live" not in payload["response"]["content"]
        assert wd.provenance.cache_redacted is True

    def test_bundle_tolerates_envelope_in_config_from_string(self, tmp_path):
        wd, _ = capture_from_run(config_text=CONFIG_TEXT, name="cap")
        bundle = wd.to_bundle_yaml()
        # The envelope-bearing bundle must still load as a Config (extra='ignore').
        cfg = Config.from_string(bundle)
        assert cfg.raw["agent"]["name"] == "x"


class TestPrepareCapturedBundle:
    def test_injects_replay_cache(self):
        bundle = "workflow:\n  name: cap\nagent:\n  name: x\nllm:\n  model: m\n"
        out = prepare_captured_bundle(bundle, cache_dir="/tmp/sidecar")
        import yaml as _y

        data = _y.safe_load(out)
        assert data["replay"]["mode"] == "cache"
        assert data["replay"]["cache_dir"] == "/tmp/sidecar"

    def test_noop_without_cache_dir(self):
        bundle = "workflow:\n  name: cap\n"
        assert prepare_captured_bundle(bundle, cache_dir=None) == bundle


class TestOfflineReplayProof:
    def test_captured_bundle_is_rerunnable_offline(self, tmp_path):
        # 1. simulate a cache-mode run -> populate a cache_dir
        run_cache = tmp_path / "run_cache"
        _populate_cache(run_cache, content="answer")
        # 2. capture: freeze the cache into entries
        wd, entries = capture_from_run(config_text=CONFIG_TEXT, name="cap", with_cache=True, cache_dir=str(run_cache))
        assert entries is not None and len(entries) == 1
        # 3. hydrate a sidecar dir + re-run with RAISE-on-miss (offline, no live call)
        sidecar = tmp_path / "sidecar"
        DirectoryCacheSidecar(sidecar).write(entries)
        replay_inner = MockClient([])  # would return "No more responses" if called
        replay_client = CachedClient(replay_inner, ResponseCache(sidecar), on_miss=CacheMissPolicy.RAISE)
        r2 = asyncio.run(replay_client.complete([{"role": "user", "content": "hi"}]))
        assert replay_inner.call_count == 0  # fully offline
        assert r2.content == "answer"  # byte-identical


class TestValidateCapture:
    def test_warns_empty_cache_when_with_cache(self, tmp_path):
        wd, _ = capture_from_run(
            config_text=CONFIG_TEXT, name="cap", with_cache=True, cache_dir=str(tmp_path / "empty")
        )
        warnings = validate_capture(wd, None)
        assert any("no cache entries" in w for w in warnings)

    def test_no_warnings_for_clean_capture(self, tmp_path):
        _populate_cache(tmp_path / "c")
        wd, entries = capture_from_run(
            config_text=CONFIG_TEXT, name="cap", with_cache=True, cache_dir=str(tmp_path / "c")
        )
        # may have the model_pin warning (no model_pin) but no capture-specific ones
        warnings = validate_capture(wd, entries)
        assert not any("no cache entries" in w for w in warnings)


class TestCaptureEnvelopeWins:
    def test_capture_from_bundle_keeps_new_envelope(self, tmp_path):
        # Capturing from a bundle that already has a workflow: envelope must NOT
        # keep the old envelope -- the new provenance (source_run_id) wins.
        bundle_text = (
            "workflow:\n  name: old\n  schema_version: '1.0'\n  provenance: {source_run_id: oldrun}\n"
            "agent:\n  name: x\nllm:\n  provider: openai\n  model: m\n"
        )
        wd, _ = capture_from_run(config_text=bundle_text, name="cap", source_run_id="newrun")
        out = wd.to_bundle_dict()
        assert out["workflow"]["provenance"]["source_run_id"] == "newrun"
        assert out["workflow"]["name"] == "cap"
