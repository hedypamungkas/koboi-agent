"""tests/test_server_media_admission.py -- issue #74: /v1/media/jobs admission caps
+ bounded _MediaJobTracker.

Same class as #50 (job-admission bypass) but on the media route: before this fix,
``submit_media_job_route`` admitted an unbounded number of billed media generations
(no per-tenant / global cap) and the in-memory ``_MediaJobTracker`` dict grew forever
(no eviction/TTL). These tests pin the race-free admission gate and the tracker
retention cap.

The gate considers the COMBINED load — media-local in-flight counters (owned and
released by this route) PLUS the shared ``job_registry`` running counts (read-only,
same public API ``/v1/jobs`` uses) — so a tenant cannot bypass ``per_tenant_max`` /
``max_concurrent`` by routing through ``/v1/media/jobs``. Media owns its own counters
and never mutates ``job_registry``, so this fix stays conflict-free with the parallel
PR #73 (``fix/50``) that reworks ``JobRegistry`` in ``koboi/server/jobs.py`` (no edit
to that file).
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
import httpx  # noqa: E402
from httpx import ASGITransport  # noqa: E402

from koboi.config import Config  # noqa: E402
from koboi.server import create_app  # noqa: E402
from tests.conftest import MockClient, make_mock_response  # noqa: E402

TOKEN = "secret-media"
OWNER = "env:" + hashlib.sha256(TOKEN.encode()).hexdigest()[:12]
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
SESSION_ID = "media-sess-1"
MEDIA_BODY = {"modality": "image", "prompt": "x", "session_id": SESSION_ID}


def _media_config(tmp_path: Path, *, per_tenant: int, max_concurrent: int, queue_depth: int = 32) -> Config:
    return Config.from_dict(
        {
            "agent": {"name": "srv-media", "system_prompt": "h", "max_iterations": 1},
            "llm": {"provider": "openai", "model": "m", "api_key": "x", "base_url": "http://x"},
            "memory": {"backend": "in_memory"},
            "sandbox": {"backend": "passthrough"},
            "server": {"auth_required": True},
            "jobs": {
                "per_tenant_max": per_tenant,
                "max_concurrent": max_concurrent,
                "queue_depth": queue_depth,
            },
            "media": {
                "enabled": True,
                "image": {"provider": "mock"},
                "storage": {"dir": str(tmp_path / "art")},
            },
        },
        validate=True,
    )


def _media_app(tmp_path: Path, **cfg) -> object:
    return create_app(
        _media_config(tmp_path, **cfg),
        client_factory=lambda: MockClient([make_mock_response(content="ok")]),
        enable_cors=False,
        api_keys=[TOKEN],
    )


def _hold_slots(monkeypatch, delay: float = 0.5) -> None:
    """Slow media_generate so admitted jobs hold their in-flight slot for the whole
    burst. Without this the fast mock provider can release a slot mid-burst and make
    the exact admission count non-deterministic. The admission gate itself is
    race-free (synchronous check+increment, no await between them)."""
    import koboi.facade as facade

    orig = facade.KoboiAgent.media_generate

    async def slow(self, req):  # noqa: ANN001, ANN202
        await asyncio.sleep(delay)
        return await orig(self, req)

    monkeypatch.setattr(facade.KoboiAgent, "media_generate", slow)


async def _await_terminal(client: httpx.AsyncClient, job_id: str, *, timeout_s: float = 2.0) -> dict:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        r = await client.get(f"/v1/media/jobs/{job_id}", headers=HEADERS)
        if r.status_code == 200 and r.json().get("status") in ("succeeded", "failed"):
            return r.json()
        await asyncio.sleep(0.02)
    raise AssertionError(f"media job {job_id} did not reach terminal within {timeout_s}s")


def _errcode(r: httpx.Response) -> str:
    try:
        return str(r.json().get("error", {}).get("code", r.text))
    except Exception:
        return r.text[:60]


class TestMediaAdmissionCaps:
    """Issue #74: /v1/media/jobs must enforce per-tenant and global admission."""

    async def test_media_jobs_respect_per_tenant_cap(self, tmp_path, monkeypatch):
        # Hold slots so the burst admission count is exact (no mid-burst release).
        _hold_slots(monkeypatch, delay=0.5)
        cap = 5
        app = _media_app(tmp_path, per_tenant=cap, max_concurrent=64)
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            responses = await asyncio.gather(
                *(c.post("/v1/media/jobs", json=MEDIA_BODY, headers=HEADERS) for _ in range(cap + 1))
            )
        admitted = [r for r in responses if r.status_code == 202]
        rejected = [r for r in responses if r.status_code == 429]
        assert len(admitted) == cap, (
            f"expected {cap} admitted, got {len(admitted)}: {[r.status_code for r in responses]}"
        )
        assert len(rejected) >= 1, "expected at least one per-tenant 429"
        assert all(_errcode(r) == "too_many_jobs_per_tenant" for r in rejected)

    async def test_media_jobs_respect_global_cap(self, tmp_path, monkeypatch):
        _hold_slots(monkeypatch, delay=0.5)
        app = _media_app(tmp_path, per_tenant=64, max_concurrent=2)
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            responses = await asyncio.gather(
                *(c.post("/v1/media/jobs", json=MEDIA_BODY, headers=HEADERS) for _ in range(3))
            )
        admitted = [r for r in responses if r.status_code == 202]
        rejected = [r for r in responses if r.status_code == 429]
        assert len(admitted) == 2, f"expected 2 admitted, got {len(admitted)}: {[r.status_code for r in responses]}"
        assert len(rejected) >= 1, "expected at least one global 429"
        assert all(_errcode(r) == "queue_full" for r in rejected)

    async def test_media_admission_slot_released_on_completion(self, tmp_path):
        """A completed media job must release its slot (finally), so a later burst
        is fully admitted again — no leak, no double-decrement."""
        cap = 3
        app = _media_app(tmp_path, per_tenant=cap, max_concurrent=64)
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            # Phase 1: submit `cap` jobs sequentially, awaiting terminal between each.
            for _ in range(cap):
                r = await c.post("/v1/media/jobs", json=MEDIA_BODY, headers=HEADERS)
                assert r.status_code == 202, r.text
                await _await_terminal(c, r.json()["job_id"])
            # Both counters must have returned to zero (released in finally).
            per_owner = getattr(app.state, "media_inflight_per_owner", None)
            assert per_owner is not None, "media admission counters missing"
            assert per_owner.get(OWNER, 0) == 0, f"per-owner slot leaked: {per_owner.get(OWNER)}"
            assert getattr(app.state, "media_inflight_global", 0) == 0, "global slot leaked"
            # Phase 2: a fresh full burst must be fully admitted (no carried-over denial).
            rs2 = await asyncio.gather(
                *(c.post("/v1/media/jobs", json=MEDIA_BODY, headers=HEADERS) for _ in range(cap))
            )
            admitted2 = sum(1 for r in rs2 if r.status_code == 202)
            assert admitted2 == cap, f"post-completion burst should be fully admitted, got {admitted2}/{cap}"

    async def test_media_admission_releases_slot_on_preflight_exception(self, tmp_path, monkeypatch):
        """Regression (code-review finding): if pool.get_or_create raises a NON-PoolFull
        exception (e.g. InvalidSessionId on an untrusted body.session_id, or an
        _build_agent failure), the reserved slot MUST still be released — otherwise
        repeating it max_concurrent times permanently DoS-locks the media endpoint
        at 429 queue_full (counter leaked, never decremented)."""
        cap = 2
        app = _media_app(tmp_path, per_tenant=64, max_concurrent=cap)
        pool_obj = app.state.pool
        orig_get_or_create = pool_obj.get_or_create

        async def boom(_session_id):
            raise RuntimeError("simulated pre-flight failure (e.g. _build_agent)")

        monkeypatch.setattr(pool_obj, "get_or_create", boom)
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            # Trigger `cap` pre-flight failures. Each reserves a slot then raises
            # before any media task is spawned, so the task's finally cannot rescue
            # the slot — the route's own except must release it. (The RuntimeError
            # propagates through ASGITransport; we catch it at the client.)
            for _ in range(cap):
                with pytest.raises(RuntimeError):
                    await c.post("/v1/media/jobs", json=MEDIA_BODY, headers=HEADERS)
            # Counter MUST be back at 0. RED (pre-fix): stuck at `cap` (leaked).
            assert app.state.media_inflight_global == 0, (
                f"slot leaked after pre-flight exceptions: {app.state.media_inflight_global}"
            )
            # Prove no permanent DoS: after repairing get_or_create, a valid submit
            # is still admitted (RED: 429 queue_full because the cap is exhausted).
            monkeypatch.setattr(pool_obj, "get_or_create", orig_get_or_create)
            r_ok = await c.post("/v1/media/jobs", json=MEDIA_BODY, headers=HEADERS)
            assert r_ok.status_code == 202, f"permanent DoS: valid submit rejected {r_ok.status_code}"


class TestMediaJobTrackerBound:
    """Issue #74: _MediaJobTracker must be bounded — evict the oldest TERMINAL entries
    FIFO when create() would exceed the retention cap. Never evict pending/in-flight."""

    async def test_media_tracker_bounded(self, tmp_path, monkeypatch):
        import koboi.server.app as appmod

        n = 3
        # raising=False keeps the test robust to the constant being defined on the
        # module (it is, post-fix) — the assertion below is what proves eviction ran.
        monkeypatch.setattr(appmod, "_MEDIA_TRACKER_MAX", n, raising=False)
        app = _media_app(tmp_path, per_tenant=64, max_concurrent=64)
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            # Submit n + 3 jobs SEQUENTIALLY, awaiting each to terminal so every prior
            # entry is terminal (evictable) when the next create() runs.
            for _ in range(n + 3):
                r = await c.post("/v1/media/jobs", json=MEDIA_BODY, headers=HEADERS)
                assert r.status_code == 202, r.text
                await _await_terminal(c, r.json()["job_id"])
            jobs = app.state.media_jobs._jobs
            assert len(jobs) <= n, f"tracker unbounded: {len(jobs)} > {n}"
            # No pending entry may ever be retained as a side effect of bad eviction
            # (all retained entries must be terminal at this point).
            for rec in jobs.values():
                assert rec["status"] in ("succeeded", "failed"), f"non-terminal entry retained after eviction: {rec}"
