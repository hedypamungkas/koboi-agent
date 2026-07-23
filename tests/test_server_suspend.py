"""Opt-in POST /v1/sessions/{id}/suspend -- consistent DB snapshot for an external
suspend/snapshot step (e.g. koboi-range)."""

from __future__ import annotations

import os
import sqlite3

import httpx
from httpx import ASGITransport

from koboi.config import Config
from koboi.server.app import create_app
from tests.conftest import MockClient, make_mock_response


def _cfg(db_path: str, backend: str = "sqlite", suspend: bool = False) -> Config:
    return Config.from_dict(
        {
            "agent": {"name": "srv", "system_prompt": "h", "max_iterations": 3},
            "llm": {"provider": "openai", "model": "m", "api_key": "test", "base_url": "http://x"},
            "memory": {"backend": backend, "db_path": db_path},
            "sandbox": {"backend": "passthrough"},
            "server": {"auth_required": False, "suspend_enabled": suspend},
        },
        validate=True,
    )


def _app(cfg: Config):
    factory = lambda: MockClient([make_mock_response(content="ok")])  # noqa: E731
    return create_app(cfg, client_factory=factory, enable_cors=False)


class TestSuspendRoute:
    async def test_disabled_by_default_returns_404(self, tmp_path):
        db = str(tmp_path / "d.db")
        async with httpx.AsyncClient(transport=ASGITransport(app=_app(_cfg(db))), base_url="http://t") as c:
            r = await c.post("/v1/sessions/sess-1/suspend")
            assert r.status_code == 404  # opt-in: route self-disables when unset

    async def test_enabled_writes_consistent_snapshot(self, tmp_path):
        db = str(tmp_path / "d.db")
        async with httpx.AsyncClient(
            transport=ASGITransport(app=_app(_cfg(db, suspend=True))), base_url="http://t"
        ) as c:
            r = await c.post("/v1/sessions")
            assert r.status_code == 201
            sid = r.json()["session_id"]

            r = await c.post(f"/v1/sessions/{sid}/suspend")
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["snapshot_path"].endswith(".suspend.db")
            assert body["snapshot_bytes"] > 0
            assert "ok" in body["checkpoint"]
            assert os.path.exists(body["snapshot_path"])
            # the snapshot file is standalone + integrity-clean
            v = sqlite3.connect(body["snapshot_path"])
            assert v.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            v.close()

    async def test_requires_sqlite_backend(self, tmp_path):
        db = str(tmp_path / "d.db")
        async with httpx.AsyncClient(
            transport=ASGITransport(app=_app(_cfg(db, backend="in_memory", suspend=True))),
            base_url="http://t",
        ) as c:
            r = await c.post("/v1/sessions/sess-1/suspend")
            assert r.status_code == 409
            assert "not_persisted" in r.text
