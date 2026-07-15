"""W5b tests: REST /v1/media/generate endpoint + R2/S3 MediaStore backend."""

from __future__ import annotations

import importlib.util
import sys
import types

import httpx
import pytest
from httpx import ASGITransport

from koboi.config import Config
from koboi.media.backend import build_media
from koboi.media.store import MediaStore
from koboi.media.types import MediaResult
from koboi.server.app import create_app
from koboi.server.schema import MediaGenerateRequest, MediaGenerateResponse
from tests.conftest import MockClient, make_mock_response


def _inject_mock_boto3(monkeypatch, put_object=None) -> list[dict]:
    """Inject a fake boto3 module; returns a list recording put_object kwargs."""
    calls: list[dict] = []
    mock_client = types.SimpleNamespace(put_object=lambda **kw: calls.append(kw) or (put_object and put_object(**kw)))
    mock_boto3 = types.ModuleType("boto3")
    mock_boto3.client = lambda service_name="s3", **_kw: mock_client
    monkeypatch.setitem(sys.modules, "boto3", mock_boto3)
    return calls


# ── schema models ──


class TestSchemaModels:
    def test_request_extra_ignored(self):
        req = MediaGenerateRequest(modality="image", prompt="x", bogus="y")
        assert req.modality == "image"
        assert not hasattr(req, "bogus")

    def test_response_cost_coerce(self):
        resp = MediaGenerateResponse(request_id="r", modality="image", status="ok", cost_usd=0.02)
        assert resp.cost_usd == 0.02


# ── R2/S3 MediaStore ──


class TestR2S3Store:
    async def test_r2_put_returns_s3_uri(self, monkeypatch):
        calls = _inject_mock_boto3(monkeypatch)
        store = MediaStore(backend="r2", bucket="mybucket", endpoint_url="https://x.r2.com", key_prefix="media")
        uri = await store.save(
            MediaResult(request_id="x", modality="image", status="ok", data=b"\x89PNGdata", content_type="image/png")
        )
        assert uri.startswith("s3://mybucket/media/")
        assert uri.endswith(".png")
        assert len(calls) == 1
        assert calls[0]["Bucket"] == "mybucket"
        assert calls[0]["ContentType"] == "image/png"

    def test_r2_missing_bucket_raises(self, monkeypatch):
        _inject_mock_boto3(monkeypatch)
        with pytest.raises(ValueError):
            MediaStore(backend="r2")

    def test_r2_missing_boto3_raises_notimplemented(self):
        if importlib.util.find_spec("boto3") is not None:
            pytest.skip("boto3 installed; NotImplementedError path not exercisable here")
        with pytest.raises(NotImplementedError):
            MediaStore(backend="r2", bucket="b")

    def test_unknown_backend_raises(self):
        with pytest.raises(NotImplementedError):
            MediaStore(backend="ftp")


class TestBuildMediaR2Wiring:
    def test_r2_storage_wired(self, monkeypatch, tmp_path):
        _inject_mock_boto3(monkeypatch)
        backend = build_media(
            {
                "enabled": True,
                "image": {"provider": "mock"},
                "storage": {"backend": "r2", "bucket": "b", "dir": str(tmp_path)},
            }
        )
        assert backend is not None
        assert backend.store._backend == "r2"
        assert backend.store._writer is not None


# ── REST endpoint ──


def _media_cfg(db_path: str, storage_dir: str) -> Config:
    return Config.from_dict(
        {
            "agent": {"name": "srv", "system_prompt": "h", "max_iterations": 3},
            "llm": {"provider": "openai", "model": "m", "api_key": "test", "base_url": "http://x"},
            "memory": {"backend": "sqlite", "db_path": db_path},
            "sandbox": {"backend": "passthrough"},
            "server": {"auth_required": False},
            "media": {
                "enabled": True,
                "image": {"provider": "mock"},
                "storage": {"backend": "local", "dir": storage_dir},
            },
        },
        validate=True,
    )


def _app(cfg: Config):
    factory = lambda: MockClient([make_mock_response(content="ok")])  # noqa: E731
    return create_app(cfg, client_factory=factory, enable_cors=False)


class TestMediaGenerateEndpoint:
    async def test_generate_image(self, tmp_path):
        cfg = _media_cfg(str(tmp_path / "d.db"), str(tmp_path / "art"))
        async with httpx.AsyncClient(transport=ASGITransport(app=_app(cfg)), base_url="http://t") as c:
            r = await c.post("/v1/media/generate", json={"modality": "image", "prompt": "a cat"})
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "ok"
            assert body["modality"] == "image"
            assert body["local_path"]

    async def test_idempotency_409_on_duplicate(self, tmp_path):
        cfg = _media_cfg(str(tmp_path / "d.db"), str(tmp_path / "art"))
        headers = {"X-Session-Id": "sess-1", "Idempotency-Key": "k1"}
        async with httpx.AsyncClient(transport=ASGITransport(app=_app(cfg)), base_url="http://t") as c:
            r1 = await c.post("/v1/media/generate", json={"prompt": "x"}, headers=headers)
            assert r1.status_code == 200
            r2 = await c.post("/v1/media/generate", json={"prompt": "x"}, headers=headers)
            assert r2.status_code == 409

    async def test_media_not_configured_returns_failed(self, tmp_path):
        # No media block -> the agent has no media_provider; endpoint returns 200 + status failed.
        cfg = Config.from_dict(
            {
                "agent": {"name": "srv", "system_prompt": "h", "max_iterations": 3},
                "llm": {"provider": "openai", "model": "m", "api_key": "test", "base_url": "http://x"},
                "memory": {"backend": "sqlite", "db_path": str(tmp_path / "d.db")},
                "sandbox": {"backend": "passthrough"},
                "server": {"auth_required": False},
            },
            validate=True,
        )
        async with httpx.AsyncClient(transport=ASGITransport(app=_app(cfg)), base_url="http://t") as c:
            r = await c.post("/v1/media/generate", json={"prompt": "x"})
            assert r.status_code == 200
            assert r.json()["status"] == "failed"
