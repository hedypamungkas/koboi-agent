"""Tests for koboi.media.store (MediaStore) + koboi.media.backend (MediaBackend, build_media)."""

from __future__ import annotations

from pathlib import Path

import pytest

from koboi.media.backend import MediaBackend, build_media
from koboi.media.budget import CountingImageProvider
from koboi.media.providers.mock import MockImageProvider
from koboi.media.store import MediaStore
from koboi.media.types import MediaRequest, MediaResult


class TestMediaStore:
    async def test_save_writes_data_and_returns_path(self, tmp_path):
        store = MediaStore(backend="local", dir=str(tmp_path))
        result = MediaResult(request_id="x", modality="image", status="ok", data=b"rawbytes", content_type="image/png")
        path = await store.save(result)
        assert Path(path).exists()
        assert Path(path).read_bytes() == b"rawbytes"
        assert path.endswith(".png")

    async def test_save_dedups_identical_content(self, tmp_path):
        store = MediaStore(backend="local", dir=str(tmp_path))
        result = MediaResult(request_id="x", modality="image", status="ok", data=b"same", content_type="image/png")
        p1 = await store.save(result)
        p2 = await store.save(result)
        assert p1 == p2  # content-hash dedup -> same path, not rewritten

    async def test_save_fetches_url_when_no_data(self, tmp_path, monkeypatch):
        store = MediaStore(backend="local", dir=str(tmp_path))

        class _Resp:
            content = b"url-bytes"

            def raise_for_status(self) -> None:
                return None

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url):
                return _Resp()

        monkeypatch.setattr("koboi.media.store.httpx.AsyncClient", lambda **kwargs: _Client())

        result = MediaResult(
            request_id="x", modality="image", status="ok", url="https://example.com/i.png", content_type="image/png"
        )
        path = await store.save(result)
        assert Path(path).read_bytes() == b"url-bytes"

    def test_non_local_backend_not_implemented(self):
        with pytest.raises(NotImplementedError):
            MediaStore(backend="r2", dir=str("./x"))


class TestBuildMedia:
    def test_returns_none_when_disabled(self):
        assert build_media(None) is None
        assert build_media({}) is None
        assert build_media({"enabled": False}) is None

    def test_enabled_builds_backend_with_mock(self, tmp_path):
        backend = build_media({"enabled": True, "image": {"provider": "mock"}, "storage": {"dir": str(tmp_path)}})
        assert backend is not None
        assert isinstance(backend.image, MockImageProvider)

    def test_budget_wraps_in_counting_provider(self, tmp_path):
        backend = build_media(
            {
                "enabled": True,
                "image": {"provider": "mock"},
                "budget": {"max_cost_usd": 1.0, "max_images": 5},
                "storage": {"dir": str(tmp_path)},
            }
        )
        assert isinstance(backend.image, CountingImageProvider)


class TestMediaBackend:
    async def test_generate_image_materializes_to_disk(self, tmp_path):
        backend = MediaBackend(image=MockImageProvider(), store=MediaStore(backend="local", dir=str(tmp_path)))
        result = await backend.generate_image(MediaRequest(prompt="cat"))
        assert result.status == "ok"
        assert result.local_path is not None
        assert Path(result.local_path).exists()

    async def test_generate_image_without_image_slot_fails(self):
        backend = MediaBackend(image=None, store=None)
        result = await backend.generate_image(MediaRequest(prompt="cat"))
        assert result.status == "failed"
        assert "not configured" in (result.rejection_reason or "")

    async def test_close_no_exception(self, tmp_path):
        backend = MediaBackend(image=MockImageProvider(), store=MediaStore(backend="local", dir=str(tmp_path)))
        await backend.close()
