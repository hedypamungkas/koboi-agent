"""Tests for the control-plane sidecar DB path resolution (G2).

Gated on the ``api`` extra: ``pytest.importorskip("fastapi")`` skips cleanly when
fastapi isn't installed (app.py imports fastapi at module top).
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from koboi.config import Config  # noqa: E402
from koboi.server import create_app  # noqa: E402
from koboi.server.app import _sidecar_db_path  # noqa: E402
from koboi.server.jobs import JobStore  # noqa: E402
from tests.conftest import MockClient, make_mock_response  # noqa: E402


class TestSidecarDbPath:
    def test_sqlite_uses_explicit_path(self):
        assert _sidecar_db_path("sqlite", "./foo.db") == "./foo.db"

    def test_sqlite_defaults_when_omitted(self):
        assert _sidecar_db_path("sqlite", None) == "koboi_memory.db"

    def test_non_sqlite_uses_explicit_path(self):
        assert _sidecar_db_path("in_memory", "./sidecar.db") == "./sidecar.db"

    def test_non_sqlite_ephemeral_when_omitted(self):
        assert _sidecar_db_path("in_memory", None) == ":memory:"


class TestCreateAppSidecarDurability:
    """create_app must wire the helper so a non-sqlite + db_path sidecar persists."""

    def test_non_sqlite_with_db_path_is_durable(self, tmp_path):
        db = str(tmp_path / "sidecar.db")
        config = Config.from_dict(
            {
                "agent": {"name": "t"},
                "llm": {"provider": "openai", "model": "m", "api_key": "x", "base_url": "http://x"},
                "memory": {"backend": "in_memory", "db_path": db},
                "sandbox": {"backend": "passthrough"},
            },
            validate=True,
        )
        app = create_app(
            config,
            client_factory=lambda: MockClient([make_mock_response(content="ok")]),
            enable_cors=False,
        )
        app.state.job_store.insert("job_x", "s1", "alice", "hi")
        app.state.job_store.close()
        app.state.ownership.close()
        # A fresh store on the same file sees the job → resume_on_startup could find it.
        again = JobStore(db)
        assert again.get("job_x") is not None
        again.close()
