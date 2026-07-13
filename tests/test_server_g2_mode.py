"""G2: per-request ``mode`` + ``max_iterations`` knobs over HTTP.

Covers the safe-scoping rules locked for G2:
- default HTTP allowlist = {chat, plan, act, auto}; yolo is opt-in only
- jobs always reject yolo (no-HITL run must keep the approval gate + rate limiter)
- max_iterations is clamped to ``server.limits.max_iterations_cap``
- invalid / out-of-allowlist mode -> 400 ``invalid_mode``
- config-only path unchanged when the fields are absent

Gated on the ``api`` extra like the other server tests.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
import httpx  # noqa: E402
from httpx import ASGITransport  # noqa: E402

from koboi.config import Config  # noqa: E402
from koboi.modes import AgentMode  # noqa: E402
from koboi.server import create_app  # noqa: E402
from koboi.server.app import _resolve_allowed_modes, _resolve_mode  # noqa: E402
from koboi.server.jobs import JobStore  # noqa: E402
from tests.conftest import MockClient, make_mock_response, make_mock_tool_call  # noqa: E402


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
        "sandbox": {"backend": "restricted"},  # C3: jobs require containment
        "server": {"auth_required": False},  # dev-open for non-auth tests
    }
    cfg.update(overrides)
    return Config.from_dict(cfg, validate=True)


def _app(responses=None, **server_overrides) -> object:
    """Build an app; ``server_overrides`` merge into the ``server:`` block."""
    factory = lambda: MockClient(responses or [make_mock_response(content="hello")])  # noqa: E731
    server = {"auth_required": False, **server_overrides}
    return create_app(_config(server=server), client_factory=factory, enable_cors=False)


def _client(app):
    return httpx.AsyncClient(base_url="http://testserver", transport=ASGITransport(app=app))


# ---------------------------------------------------------------------------
# 1. Validation helpers (the security logic lives here) -- pure unit tests.
# ---------------------------------------------------------------------------


class TestResolveHelpers:
    def test_default_allowlist_excludes_yolo(self):
        am = _resolve_allowed_modes(None)
        assert am == frozenset({"chat", "plan", "act", "auto"})
        assert "yolo" not in am

    def test_empty_allowlist_falls_back_to_default(self):
        assert _resolve_allowed_modes([]) == frozenset({"chat", "plan", "act", "auto"})

    def test_explicit_allowlist_can_include_yolo(self):
        am = _resolve_allowed_modes(["chat", "yolo"])
        assert am == frozenset({"chat", "yolo"})

    def test_invalid_allowlist_entry_raises_at_startup(self):
        # A YAML typo must fail loud, not silently widen/narrow the boundary.
        with pytest.raises(ValueError, match="server.allowed_modes"):
            _resolve_allowed_modes(["chat", "bogus"])

    def test_non_list_allowlist_raises(self):
        with pytest.raises(ValueError, match="must be a list"):
            _resolve_allowed_modes("chat")

    def test_resolve_mode_none_returns_none(self):
        # None = config default applies (config-only path unchanged).
        assert _resolve_mode(None, _resolve_allowed_modes(None), allow_yolo=True) is None

    def test_resolve_mode_bogus_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            _resolve_mode("bogus", _resolve_allowed_modes(None), allow_yolo=True)

    def test_resolve_mode_outside_allowlist_raises(self):
        am = _resolve_allowed_modes(["chat", "plan"])  # act not permitted
        with pytest.raises(ValueError, match="not allowed"):
            _resolve_mode("act", am, allow_yolo=True)

    def test_resolve_mode_yolo_rejected_when_not_allowlisted(self):
        with pytest.raises(ValueError, match="not allowed"):
            _resolve_mode("yolo", _resolve_allowed_modes(None), allow_yolo=True)

    def test_resolve_mode_yolo_allowed_when_opted_in(self):
        am = _resolve_allowed_modes(["chat", "yolo"])
        assert _resolve_mode("yolo", am, allow_yolo=True) is AgentMode.YOLO

    def test_resolve_mode_yolo_rejected_for_jobs_even_if_allowlisted(self):
        # The job rule: allow_yolo=False beats the allowlist.
        am = _resolve_allowed_modes(["chat", "yolo"])
        with pytest.raises(ValueError, match="not allowed for autonomous jobs"):
            _resolve_mode("yolo", am, allow_yolo=False)

    def test_resolve_mode_act_ok_for_jobs(self):
        am = _resolve_allowed_modes(None)
        assert _resolve_mode("act", am, allow_yolo=False) is AgentMode.ACT


# ---------------------------------------------------------------------------
# 2. Interactive (/v1/chat/stream) -- 400 / 200 boundary.
# ---------------------------------------------------------------------------


class TestChatModeValidation:
    async def test_yolo_rejected_under_default_allowlist(self):
        async with _client(_app()) as c:
            r = await c.post("/v1/chat/stream", json={"message": "hi", "mode": "yolo"})
            assert r.status_code == 400
            assert r.json()["error"]["code"] == "invalid_mode"

    async def test_bogus_mode_rejected(self):
        async with _client(_app()) as c:
            r = await c.post("/v1/chat/stream", json={"message": "hi", "mode": "bogus"})
            assert r.status_code == 400
            assert r.json()["error"]["code"] == "invalid_mode"

    async def test_act_rejected_when_not_in_operator_allowlist(self):
        # Operator locked down to read-only; caller requests act -> 400.
        async with _client(_app(allowed_modes=["chat", "plan"])) as c:
            r = await c.post("/v1/chat/stream", json={"message": "hi", "mode": "act"})
            assert r.status_code == 400
            assert r.json()["error"]["code"] == "invalid_mode"

    async def test_yolo_allowed_when_operator_opts_in(self):
        async with _client(_app(allowed_modes=["chat", "yolo"])) as c:
            r = await c.post("/v1/chat/stream", json={"message": "hi", "mode": "yolo"})
            assert r.status_code == 200

    async def test_valid_mode_accepted(self):
        async with _client(_app()) as c:
            r = await c.post("/v1/chat/stream", json={"message": "hi", "mode": "act"})
            assert r.status_code == 200

    async def test_no_mode_backward_compat(self):
        # Absent mode -> config default applies, request proceeds normally.
        async with _client(_app()) as c:
            r = await c.post("/v1/chat/stream", json={"message": "hi"})
            assert r.status_code == 200


# ---------------------------------------------------------------------------
# 3. Jobs (/v1/jobs) -- yolo always rejected; mode/max_iterations accepted.
# ---------------------------------------------------------------------------


class TestJobModeValidation:
    async def test_job_yolo_always_rejected(self):
        # Even with yolo in the operator allowlist, jobs refuse it.
        async with _client(_app(allowed_modes=["chat", "yolo"])) as c:
            r = await c.post("/v1/jobs", json={"message": "do thing", "mode": "yolo"})
            assert r.status_code == 400
            assert r.json()["error"]["code"] == "invalid_mode"

    async def test_job_bogus_rejected(self):
        async with _client(_app()) as c:
            r = await c.post("/v1/jobs", json={"message": "do thing", "mode": "bogus"})
            assert r.status_code == 400

    async def test_job_valid_mode_accepted(self):
        async with _client(_app()) as c:
            r = await c.post("/v1/jobs", json={"message": "do thing", "mode": "act"})
            assert r.status_code == 202

    async def test_job_no_mode_backward_compat(self):
        async with _client(_app()) as c:
            r = await c.post("/v1/jobs", json={"message": "do thing"})
            assert r.status_code == 202

    async def test_job_large_max_iterations_accepted_not_rejected(self):
        # max_iterations is clamped (ceiling), never rejected for being large.
        async with _client(_app(limits={"max_iterations_cap": 5})) as c:
            r = await c.post("/v1/jobs", json={"message": "do thing", "max_iterations": 100000})
            assert r.status_code == 202


# ---------------------------------------------------------------------------
# 4. JobStore round-trip -- the new columns persist (so resume re-applies them).
# ---------------------------------------------------------------------------


class TestJobStoreG2Columns:
    def test_insert_persists_mode_and_max_iterations(self, tmp_path):
        store = JobStore(db_path=str(tmp_path / "jobs.db"))
        store.insert("j1", "s1", "dev", "msg", mode="act", max_iterations=7)
        row = store.get("j1")
        assert row is not None
        assert row["mode"] == "act"
        assert row["max_iterations"] == 7

    def test_insert_defaults_none_for_omitted(self, tmp_path):
        store = JobStore(db_path=str(tmp_path / "jobs.db"))
        store.insert("j2", "s2", "dev", "msg")
        row = store.get("j2")
        assert row is not None
        assert row["mode"] is None
        assert row["max_iterations"] is None

    def test_legacy_row_gains_columns_via_migration(self, tmp_path):
        # A pre-G2 DB must gain the columns (idempotent ALTER) without data loss.
        import sqlite3

        db = str(tmp_path / "legacy.db")
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE jobs (job_id TEXT PRIMARY KEY, session_id TEXT, owner TEXT, "
            "status TEXT, message TEXT, idempotency_key TEXT, created_at REAL, updated_at REAL)"
        )
        conn.execute("INSERT INTO jobs VALUES ('old','s','dev','pending','m',NULL,0,0)")
        conn.commit()
        conn.close()
        # Opening with JobStore runs the migration.
        store = JobStore(db_path=db)
        row = store.get("old")
        assert row is not None
        assert row["message"] == "m"  # data preserved
        assert row["mode"] is None  # new column, NULL for legacy rows
        assert row["max_iterations"] is None


# ---------------------------------------------------------------------------
# 5. Behavioral -- the stamped mode actually changes tool-execution behavior.
# ---------------------------------------------------------------------------


class TestModeStampingTakesEffect:
    async def test_chat_mode_blocks_write_tool(self):
        """mode=chat must reach ModeHook: a write_file tool call is blocked.

        Proves the per-request stamp (not just validation) takes effect: the
        pipeline denies the call with the CHAT-mode reason instead of executing.
        write_file is MODERATE, so the interactive approval handler auto-allows
        it (no HITL prompt / no 120s timeout) -- the mode block is what stops it.
        """
        import json

        factory = lambda: MockClient(  # noqa: E731
            [
                make_mock_response(tool_calls=[make_mock_tool_call("write_file", {"path": "a.txt", "content": "x"})]),
                make_mock_response(content="ok"),
            ]
        )
        app = create_app(_config(), client_factory=factory, enable_cors=False)
        async with _client(app) as c:
            async with c.stream(
                "POST",
                "/v1/chat/stream",
                json={"message": "write a file", "mode": "chat"},
            ) as r:
                assert r.status_code == 200
                text = (await r.aread()).decode()

        events = []
        for line in text.split("\n"):
            if line.startswith("data: "):
                payload = line[6:]
                if payload != "[DONE]":
                    events.append(json.loads(payload))
        # The write_file call was denied with the CHAT-mode block reason.
        blocked = [
            e
            for e in events
            if e.get("type") == "tool_result"
            and ("CHAT mode" in e.get("result", "") or "not allowed" in e.get("result", ""))
        ]
        assert blocked, f"expected a CHAT-mode tool block; events={events}"
