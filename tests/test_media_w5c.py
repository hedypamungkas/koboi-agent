"""W5c tests: non-blocking tools (#4) + async-jobs REST (#3) + TUI gallery (#2)."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from httpx import ASGITransport

from koboi.config import Config
from koboi.media.async_job import MediaJob
from koboi.media.backend import MediaBackend
from koboi.media.base import BaseVideoProvider
from koboi.media.providers.mock import MockImageProvider
from koboi.media.store import MediaStore
from koboi.media.types import MediaRequest, MediaResult
from koboi.server.app import create_app
from koboi.tools.builtin import register_all
from koboi.tools.registry import ToolRegistry


# ── TUI parse helper (#2) ──


def _import_parse():
    """parse_media_artifact lives in the [tui] extra module; skip if textual absent."""
    pytest.importorskip("textual")
    from koboi.tui.screens.media_gallery import parse_media_artifact

    return parse_media_artifact


class TestParseMediaArtifact:
    def test_parse_valid_image(self):
        parse = _import_parse()
        art = parse("generate_image", "Image saved: /tmp/x.png (image/png, 1024x1024, $0.04/image, model=m)")
        assert art is not None
        assert art["modality"] == "image"
        assert art["path"] == "/tmp/x.png"
        assert "model=m" in art["meta"]

    def test_parse_non_media_tool(self):
        parse = _import_parse()
        assert parse("web_search", "Search results...") is None

    def test_parse_error_result(self):
        parse = _import_parse()
        assert parse("generate_image", "Error: media not configured") is None

    def test_parse_empty(self):
        parse = _import_parse()
        assert parse("generate_image", "") is None


# ── MediaBackend submit/check (#4) ──


class _PendingVideoProvider(BaseVideoProvider):
    """submit -> queued; poll -> succeeded (one round); fetch -> bytes."""

    def __init__(self) -> None:
        self.polls = 0

    async def submit_video(self, req: MediaRequest) -> MediaJob:
        return MediaJob(job_id="vid-1", kind="video", status="queued")

    async def poll_video(self, job: MediaJob) -> MediaJob:
        self.polls += 1
        job.status = "succeeded"
        job.result = MediaResult(request_id="vid-1", modality="video", status="ok", content_type="video/mp4")
        return job

    async def fetch_video_artifact(self, job: MediaJob) -> bytes:
        return b"vidbytes"


class TestMediaBackendSubmitCheck:
    async def test_image_submit_immediate(self, tmp_path):
        backend = MediaBackend(image=MockImageProvider(), store=MediaStore(backend="local", dir=str(tmp_path)))
        job = await backend.submit_media_job(MediaRequest(modality="image", prompt="x"))
        assert job.status == "succeeded"
        assert job.result is not None and job.result.local_path

    async def test_video_submit_then_check(self, tmp_path):
        video = _PendingVideoProvider()
        backend = MediaBackend(video=video, store=MediaStore(backend="local", dir=str(tmp_path)))
        job = await backend.submit_media_job(MediaRequest(modality="video", prompt="x"))
        assert job.status == "queued"
        assert job.job_id == "vid-1"
        # check -> poll -> succeeded + materialized
        job2 = await backend.check_media_job("vid-1")
        assert job2.status == "succeeded"
        assert job2.result is not None
        assert job2.result.data == b"vidbytes"
        assert job2.result.local_path  # materialized
        assert video.polls == 1
        # re-check is idempotent (terminal)
        job3 = await backend.check_media_job("vid-1")
        assert job3.status == "succeeded"
        assert video.polls == 1  # not re-polled

    async def test_check_unknown_returns_none(self):
        backend = MediaBackend()
        assert await backend.check_media_job("nope") is None


# ── submit/check tools (#4) ──


def _registry_with_video(tmp_path) -> ToolRegistry:
    registry = ToolRegistry()
    register_all(registry)
    backend = MediaBackend(video=_PendingVideoProvider(), store=MediaStore(backend="local", dir=str(tmp_path)))
    registry.set_dep("media_provider", backend)
    return registry


class TestSubmitCheckTools:
    async def test_submit_then_check(self, tmp_path):
        import json

        registry = _registry_with_video(tmp_path)
        out = await registry.execute("submit_media_job", json.dumps({"modality": "video", "prompt": "x"}))
        assert "submitted" in out and "vid-1" in out
        out2 = await registry.execute("check_media_job", json.dumps({"job_id": "vid-1"}))
        assert "succeeded" in out2

    async def test_check_unknown(self, tmp_path):
        import json

        registry = _registry_with_video(tmp_path)
        out = await registry.execute("check_media_job", json.dumps({"job_id": "ghost"}))
        assert "No media job" in out


# ── async-jobs REST (#3) ──


def _media_cfg(db_path: str, storage_dir: str) -> Config:
    return Config.from_dict(
        {
            "agent": {"name": "srv", "system_prompt": "h", "max_iterations": 3},
            "llm": {"provider": "openai", "model": "m", "api_key": "test", "base_url": "http://x"},
            "memory": {"backend": "sqlite", "db_path": db_path},
            "sandbox": {"backend": "passthrough"},
            "server": {"auth_required": False},
            "media": {"enabled": True, "image": {"provider": "mock"}, "storage": {"dir": storage_dir}},
        },
        validate=True,
    )


def _app(cfg: Config):
    from tests.conftest import MockClient, make_mock_response

    factory = lambda: MockClient([make_mock_response(content="ok")])  # noqa: E731
    return create_app(cfg, client_factory=factory, enable_cors=False)


class TestMediaJobsEndpoint:
    async def test_submit_then_poll(self, tmp_path):
        cfg = _media_cfg(str(tmp_path / "d.db"), str(tmp_path / "art"))
        async with httpx.AsyncClient(transport=ASGITransport(app=_app(cfg)), base_url="http://t") as c:
            r = await c.post("/v1/media/jobs", json={"modality": "image", "prompt": "x"})
            assert r.status_code == 202
            job_id = r.json()["job_id"]
            assert r.json()["status"] == "pending"
            # poll until terminal (mock image completes fast)
            status = None
            for _ in range(20):
                await asyncio.sleep(0.05)
                g = await c.get(f"/v1/media/jobs/{job_id}")
                assert g.status_code == 200
                status = g.json()["status"]
                if status in ("succeeded", "failed"):
                    break
            assert status == "succeeded"
            assert g.json()["result"]["modality"] == "image"

    async def test_get_unknown_404(self, tmp_path):
        cfg = _media_cfg(str(tmp_path / "d.db"), str(tmp_path / "art"))
        async with httpx.AsyncClient(transport=ASGITransport(app=_app(cfg)), base_url="http://t") as c:
            r = await c.get("/v1/media/jobs/ghost")
            assert r.status_code == 404
