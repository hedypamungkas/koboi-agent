"""W1 tests: async job loop + video/music providers, budget, backend, tools.

The async providers' poll loop sleeps on ``asyncio.sleep``; a module-level ``no_sleep`` fixture
neutralizes it so the surplus video/music tests run instantly with a mocked transport.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from koboi.media.async_job import MediaJob, run_async_job
from koboi.media.backend import MediaBackend, build_media
from koboi.media.budget import CountingVideoProvider
from koboi.media.providers.mock import MockMusicProvider, MockVideoProvider
from koboi.media.providers.surplus import SurplusMusicProvider, SurplusVideoProvider
from koboi.media.store import MediaStore
from koboi.media.types import MediaBudget, MediaRequest, MediaResult, MediaUnit
from koboi.tools.builtin import register_all
from koboi.tools.registry import ToolRegistry
from koboi.types import RiskLevel


@pytest.fixture
def no_sleep(monkeypatch):
    """Neutralize run_async_job's poll-interval sleep so async-provider tests are instant."""

    async def _noop(_delay):
        return None

    monkeypatch.setattr("koboi.media.async_job.asyncio.sleep", _noop)


def _req(modality: str = "video") -> MediaRequest:
    return MediaRequest(modality=modality, prompt="x")


# ── run_async_job poll loop ──


class TestRunAsyncJob:
    async def test_immediate_succeeded(self, no_sleep):
        async def submit(req):
            return MediaJob(
                job_id="j1",
                kind="video",
                status="succeeded",
                result=MediaResult(request_id="r", modality="video", status="ok"),
            )

        async def poll(job):
            return job

        async def fetch(job):
            return b"vidbytes"

        res = await run_async_job(submit, poll, fetch, _req(), "video")
        assert res.status == "ok"
        assert res.data == b"vidbytes"
        assert res.modality == "video"

    async def test_one_poll_success(self, no_sleep):
        calls = {"n": 0}

        async def submit(req):
            return MediaJob(job_id="j1", kind="video", status="running")

        async def poll(job):
            calls["n"] += 1
            job.status = "succeeded"
            job.result = MediaResult(request_id="", modality="video", status="ok", duration_seconds=5.0)
            return job

        async def fetch(job):
            return b"vidbytes"

        res = await run_async_job(submit, poll, fetch, _req(), "video")
        assert res.status == "ok"
        assert res.data == b"vidbytes"
        assert calls["n"] == 1

    async def test_failed_terminal(self, no_sleep):
        async def submit(req):
            return MediaJob(job_id="j1", kind="video", status="running")

        async def poll(job):
            job.status = "failed"
            return job

        res = await run_async_job(submit, poll, AsyncMock(), _req(), "video")
        assert res.status == "failed"

    async def test_expired_terminal(self, no_sleep):
        async def submit(req):
            return MediaJob(job_id="j1", kind="music", status="running")

        async def poll(job):
            job.status = "expired"
            return job

        res = await run_async_job(submit, poll, AsyncMock(), _req(), "music")
        assert res.status == "failed"

    async def test_max_wait_timeout_carries_job_id(self, no_sleep):
        async def submit(req):
            return MediaJob(job_id="j9", kind="video", status="running")

        async def poll(job):
            return job  # never terminal

        res = await run_async_job(submit, poll, AsyncMock(), _req(), "video", poll_interval=0.0, max_wait=0.0)
        assert res.status == "failed"
        assert "still running" in (res.rejection_reason or "")
        assert res.raw.get("job_id") == "j9"

    async def test_submit_failure(self, no_sleep):
        async def submit(req):
            raise RuntimeError("net down")

        res = await run_async_job(submit, AsyncMock(), AsyncMock(), _req(), "video")
        assert res.status == "failed"
        assert "submit failed" in (res.rejection_reason or "")


# ── mock video/music providers ──


class TestMockVideoMusic:
    async def test_mock_video(self):
        res = await MockVideoProvider().generate_video(MediaRequest(modality="video", prompt="x", duration_seconds=3))
        assert res.status == "ok"
        assert res.data == b"mock-video-artifact"
        assert res.content_type == "video/mp4"
        assert res.duration_seconds == 3.0
        assert res.billing_unit == MediaUnit.JOB

    async def test_mock_music(self):
        res = await MockMusicProvider().generate_music(MediaRequest(modality="music", prompt="x", duration_seconds=5))
        assert res.status == "ok"
        assert res.data == b"mock-music-artifact"
        assert res.content_type == "audio/mpeg"
        assert res.duration_seconds == 5.0


# ── surplus video/music providers (mocked transport) ──


class TestSurplusVideoProvider:
    async def test_generate_video_per_job(self, no_sleep):
        provider = SurplusVideoProvider(api_key="k", model="kling-v3")
        provider._transport.post = AsyncMock(return_value={"id": "job_1", "status": "queued", "job_token": "tok"})
        provider._transport.get = AsyncMock(
            return_value={
                "id": "job_1",
                "status": "succeeded",
                "duration_seconds": 5,
                "width": 1280,
                "height": 720,
                "download_url": "https://cdn/artifact.mp4",
                "usage": {"buyer_cost_usd": 5000000},
            }
        )
        provider._transport.get_bytes = AsyncMock(return_value=b"mp4bytes")
        res = await provider.generate_video(MediaRequest(modality="video", prompt="waves", duration_seconds=5))
        assert res.status == "ok"
        assert res.data == b"mp4bytes"
        assert res.duration_seconds == 5.0
        assert res.billing_unit == MediaUnit.JOB
        assert res.width == 1280
        provider._transport.get_bytes.assert_awaited()

    async def test_generate_video_per_second_happyhorse(self, no_sleep):
        provider = SurplusVideoProvider(api_key="k", model="happyhorse-1-0")
        provider._transport.post = AsyncMock(return_value={"id": "j", "status": "queued"})
        provider._transport.get = AsyncMock(
            return_value={"id": "j", "status": "succeeded", "duration_seconds": 6, "download_url": "https://x/v.mp4"}
        )
        provider._transport.get_bytes = AsyncMock(return_value=b"v")
        res = await provider.generate_video(MediaRequest(modality="video", prompt="x", duration_seconds=6))
        assert res.billing_unit == MediaUnit.SECOND
        assert res.billing_quantity == 6.0

    async def test_cancel_calls_delete(self):
        provider = SurplusVideoProvider(api_key="k")
        provider._transport.delete = AsyncMock(return_value={})
        await provider.cancel_video(MediaJob(job_id="j1", kind="video"))
        provider._transport.delete.assert_awaited()

    async def test_close_calls_transport(self):
        provider = SurplusVideoProvider(api_key="k")
        provider._transport.close = AsyncMock()
        await provider.close()
        provider._transport.close.assert_awaited()

    async def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("SURPLUS_API_KEY", raising=False)
        provider = SurplusVideoProvider(api_key="")
        with pytest.raises(ValueError):
            await provider.submit_video(MediaRequest(modality="video", prompt="x"))


class TestSurplusMusicProvider:
    async def test_generate_music(self, no_sleep):
        provider = SurplusMusicProvider(api_key="k")
        provider._transport.post = AsyncMock(return_value={"id": "m1", "status": "queued"})
        provider._transport.get = AsyncMock(
            return_value={"id": "m1", "status": "succeeded", "duration_seconds": 10, "download_url": "https://x/a.mp3"}
        )
        provider._transport.get_bytes = AsyncMock(return_value=b"mp3bytes")
        res = await provider.generate_music(MediaRequest(modality="music", prompt="lo-fi beat", duration_seconds=10))
        assert res.status == "ok"
        assert res.data == b"mp3bytes"
        assert res.content_type == "audio/mpeg"
        assert res.billing_unit == MediaUnit.JOB


# ── budget (video/music) ──


class TestMediaBudgetVideoMusic:
    def test_remaining_video_seconds_cap(self):
        assert MediaBudget(max_video_seconds=10, used_video_seconds=10).remaining("video") is False

    def test_remaining_music_seconds_cap(self):
        assert MediaBudget(max_music_seconds=30, used_music_seconds=30).remaining("music") is False

    def test_record_video_accrues_duration(self):
        b = MediaBudget()
        b.record(MediaResult(request_id="r", modality="video", status="ok", duration_seconds=5.0))
        assert b.used_video_seconds == 5.0

    def test_record_music_accrues_duration(self):
        b = MediaBudget()
        b.record(MediaResult(request_id="r", modality="music", status="ok", duration_seconds=8.0))
        assert b.used_music_seconds == 8.0


class TestCountingVideoProvider:
    async def test_rejects_when_exhausted(self):
        budget = MediaBudget(max_video_seconds=0)  # 0 -> exhausted immediately
        provider = CountingVideoProvider(MockVideoProvider(), budget)
        res = await provider.generate_video(MediaRequest(modality="video", prompt="x"))
        assert res.status == "rejected"

    async def test_delegates_and_records(self):
        budget = MediaBudget(max_video_seconds=60)
        provider = CountingVideoProvider(MockVideoProvider(duration_seconds=4.0), budget)
        res = await provider.generate_video(MediaRequest(modality="video", prompt="x", duration_seconds=4))
        assert res.status == "ok"
        assert budget.used_video_seconds == 4.0


# ── backend build_media + slots ──


class TestBuildMediaVideoMusic:
    def test_builds_video_music_when_configured(self, tmp_path):
        backend = build_media(
            {
                "enabled": True,
                "image": {"provider": "mock"},
                "video": {"provider": "mock"},
                "music": {"provider": "mock"},
                "storage": {"dir": str(tmp_path)},
            }
        )
        assert backend is not None
        assert backend.video is not None
        assert backend.music is not None

    def test_no_video_music_when_absent(self, tmp_path):
        backend = build_media({"enabled": True, "image": {"provider": "mock"}, "storage": {"dir": str(tmp_path)}})
        assert backend.video is None
        assert backend.music is None

    def test_budget_wraps_video(self, tmp_path):
        backend = build_media(
            {
                "enabled": True,
                "image": {"provider": "mock"},
                "video": {"provider": "mock"},
                "budget": {"max_cost_usd": 1.0},
                "storage": {"dir": str(tmp_path)},
            }
        )
        assert isinstance(backend.video, CountingVideoProvider)


class TestMediaBackendVideoMusic:
    async def test_generate_video_materializes(self, tmp_path):
        backend = MediaBackend(video=MockVideoProvider(), store=MediaStore(backend="local", dir=str(tmp_path)))
        res = await backend.generate_video(MediaRequest(modality="video", prompt="x"))
        assert res.status == "ok"
        assert res.local_path is not None
        assert Path(res.local_path).exists()

    async def test_generate_video_without_slot_fails(self):
        backend = MediaBackend(video=None)
        res = await backend.generate_video(MediaRequest(modality="video", prompt="x"))
        assert res.status == "failed"
        assert "not configured" in (res.rejection_reason or "")


# ── tools ──


class TestGenerateVideoMusicTools:
    @staticmethod
    def _registry(tmp_path) -> ToolRegistry:
        registry = ToolRegistry()
        register_all(registry)
        backend = build_media(
            {
                "enabled": True,
                "video": {"provider": "mock"},
                "music": {"provider": "mock"},
                "storage": {"dir": str(tmp_path)},
            }
        )
        registry.set_dep("media_provider", backend)
        return registry

    async def test_generate_video(self, tmp_path):
        out = await self._registry(tmp_path).execute("generate_video", __import__("json").dumps({"prompt": "waves"}))
        assert out.startswith("Video saved:")
        assert str(tmp_path) in out

    async def test_generate_music(self, tmp_path):
        out = await self._registry(tmp_path).execute("generate_music", __import__("json").dumps({"prompt": "lo-fi"}))
        assert out.startswith("Music saved:")

    async def test_video_not_configured(self):
        registry = ToolRegistry()
        register_all(registry)
        out = await registry.execute("generate_video", __import__("json").dumps({"prompt": "x"}))
        assert "media not configured" in out

    def test_video_flags(self):
        registry = ToolRegistry()
        register_all(registry)
        td = registry.get_definition("generate_video")
        assert td is not None
        assert td.risk_level == RiskLevel.DESTRUCTIVE
        assert td.idempotent is False
        assert td.timeout == 1800.0

    def test_music_flags(self):
        registry = ToolRegistry()
        register_all(registry)
        td = registry.get_definition("generate_music")
        assert td is not None
        assert td.risk_level == RiskLevel.MODERATE
        assert td.idempotent is False
        assert td.timeout == 600.0
