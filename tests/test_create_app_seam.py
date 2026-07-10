"""Issue #1: create_app state-injection seam + JobRegistry.get_events surface."""

from __future__ import annotations

import pytest

from koboi.config import Config
from koboi.server.jobs import JobRegistry, JobStore
from koboi.server.ownership import OwnershipStore


def _config(tmp_path) -> Config:
    cfg = Config.from_dict(
        {
            "agent": {"name": "t", "system_prompt": "h", "max_iterations": 3, "mode": "chat"},
            "llm": {"provider": "openai", "model": "m", "api_key": "test", "base_url": "http://x"},
            "memory": {"backend": "sqlite", "db_path": str(tmp_path / "d.db")},
        },
        validate=True,
    )
    return cfg


class TestJobRegistryGetEvents:
    def test_get_events_returns_capped_list(self):
        jr = JobRegistry()
        jr.register("J1", "sess", "owner")
        jr.append_event("J1", {"x": 1})
        jr.append_event("J1", {"x": 2})
        assert jr.get_events("J1") == [{"x": 1}, {"x": 2}]

    def test_get_events_unknown_job(self):
        assert JobRegistry().get_events("nope") == []


@pytest.mark.skipif(
    pytest.importorskip("fastapi", reason="fastapi not installed") is None,
    reason="fastapi not installed",
)
class TestCreateAppSeam:
    def test_defaults_preserved(self, tmp_path):
        from koboi.server.app import create_app

        app = create_app(_config(tmp_path))
        assert isinstance(app.state.job_store, JobStore)
        assert isinstance(app.state.job_registry, JobRegistry)
        assert isinstance(app.state.ownership, OwnershipStore)

    def test_injected_stores_are_used(self, tmp_path):
        from koboi.server.app import create_app

        js = JobStore(db_path=str(tmp_path / "j.db"))
        jr = JobRegistry()
        ow = OwnershipStore(db_path=str(tmp_path / "o.db"))
        app = create_app(_config(tmp_path), job_store=js, event_buffer=jr, ownership_store=ow)
        assert app.state.job_store is js
        assert app.state.job_registry is jr
        assert app.state.ownership is ow
