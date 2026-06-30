"""tests/test_pool_workdir -- Bucket A1: per-session workdir is created on build.

Regression for the e2e failure where ``list_files``/``read_file`` returned
``path '/app/workspace/<id>' not found``: ``AgentPool._build_agent`` stamped
``sandbox.workdir`` per session but never created the directory, so any session
that didn't call ``write_file`` (which ``makedirs``-es lazily) had no workdir.
"""

from __future__ import annotations

import os
from pathlib import Path

from koboi.config import Config
from koboi.server.pool import AgentPool
from tests.conftest import MockClient, make_mock_response


def _config(**overrides) -> Config:
    cfg = {
        "agent": {"name": "srv", "system_prompt": "h", "max_iterations": 3},
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test",
            "base_url": "http://localhost:8080/v1",
        },
        "memory": {"backend": "in_memory"},
        "sandbox": {"backend": "restricted"},
    }
    cfg.update(overrides)
    return Config.from_dict(cfg, validate=True)


def _pool(tmp_path, **overrides):
    return AgentPool(
        _config(**overrides),
        client_factory=lambda: MockClient([make_mock_response(content="hi")]),
        workspace_root=str(tmp_path),
    )


class TestPoolWorkdir:
    async def test_workdir_created_on_get_or_create(self, tmp_path):
        pool = _pool(tmp_path)
        sid = "abc123"
        workdir = pool.workdir_for(sid)
        assert not Path(workdir).exists()  # not created until the agent is built

        await pool.get_or_create(sid)

        assert Path(workdir).is_dir()  # now exists (would fail without the fix)
        await pool.close_all()

    async def test_read_only_path_resolves_to_existing_workdir(self, tmp_path):
        """A read-only session (no write_file) must still have a usable workdir.

        Before the fix the workdir only appeared as a side effect of write_file,
        so list_files('.') resolved to a non-existent dir -> 'path not found'.
        """
        pool = _pool(tmp_path)
        sid = "sess_ro"
        agent = await pool.get_or_create(sid)

        sandbox = agent._core.tools.get_dep("sandbox")
        assert sandbox is not None
        resolved = sandbox.validate_path(".")
        assert os.path.isdir(resolved)  # listdir would succeed, not FileNotFoundError

        await pool.close_all()

    async def test_passthrough_backend_also_creates_workdir(self, tmp_path):
        """mkdir is backend-agnostic; passthrough sessions get a workdir too."""
        pool = _pool(tmp_path, sandbox={"backend": "passthrough"})
        sid = "pass_sess"
        workdir = pool.workdir_for(sid)

        await pool.get_or_create(sid)

        assert Path(workdir).is_dir()
        await pool.close_all()
