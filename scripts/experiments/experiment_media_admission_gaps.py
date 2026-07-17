#!/usr/bin/env python3
"""experiment_media_admission_gaps.py -- empirical proof that /v1/media/jobs has
NO admission control (per-tenant or global), unlike /v1/jobs.

This is the 4th bug found in the same audit pass that produced #50/#51/#69.
It is the same CLASS as #50 (job-admission bypass) but on the media route:
submit_media_job_route (koboi/server/app.py) calls pool.get_or_create() and
spawns a background generation task WITHOUT ever consulting job_registry
admission (peek_admit / active_count_for_owner). So a single authenticated
tenant can admit an unbounded number of billed media generations and grow the
in-memory _MediaJobTracker dict forever (no eviction/TTL).

METHOD (no network, in-process): build a REAL create_app() with auth on,
media.enabled + mock image provider (offline), and drive it via httpx
ASGITransport. Each CHECK prints OPEN (bug present) vs FIXED (capped) with
concrete status-code evidence. Run:

    python experiment_media_admission_gaps.py

Exits 1 if any CHECK is OPEN (the current state of `main`), 0 only after a fix.
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
from pathlib import Path

# Repo root on sys.path so `from tests.conftest import ...` resolves when run
# as a plain script (not under pytest).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx  # noqa: E402
from httpx import ASGITransport  # noqa: E402

from koboi.config import Config  # noqa: E402
from koboi.server import create_app  # noqa: E402

try:  # the canonical test double used by the server test-suite
    from tests.conftest import MockClient, make_mock_response
except Exception:  # pragma: no cover - minimal inline fallback
    from koboi.types import AgentResponse, TokenUsage  # noqa: E402
    from koboi.llm.base import LLMClient  # noqa: E402

    class MockClient(LLMClient):  # type: ignore[no-redef]
        def __init__(self, responses=None):
            self.responses = responses or []
            self._i = 0
            self._model = "mock-model"

        @property
        def model(self):
            return self._model

        @model.setter
        def model(self, v):
            self._model = v

        async def complete(self, messages, tools=None, response_format=None):
            if self._i < len(self.responses):
                r = self.responses[self._i]
                self._i += 1
                return r
            return AgentResponse(content="ok", tool_calls=[])

        async def complete_stream(self, messages, tools=None, response_format=None):
            r = await self.complete(messages, tools, response_format)
            yield type("E", (), {"response": r, "content": r.content or ""})

        async def get_embeddings(self, text):
            return None

        async def close(self):
            pass

    def make_mock_response(content="ok"):
        return AgentResponse(content=content, tool_calls=[], usage=TokenUsage(10, 20))


TOKEN = "secret"
OWNER = "env:" + hashlib.sha256(TOKEN.encode()).hexdigest()[:12]
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
MEDIA_BODY = {"modality": "image", "prompt": "x", "session_id": "media-sess-1"}


def _config(tmp: Path, *, per_tenant=5, max_concurrent=64, queue_depth=32) -> Config:
    return Config.from_dict(
        {
            "agent": {"name": "exp", "system_prompt": "h", "max_iterations": 1},
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
                "storage": {"dir": str(tmp / "art")},
            },
        },
        validate=True,
    )


def _errcode(resp: httpx.Response) -> str:
    try:
        return str(resp.json().get("error", {}).get("code", resp.text))
    except Exception:
        return resp.text[:60]


def _build_app(tmp: Path, **cfg) -> tuple[object, httpx.AsyncClient]:
    app = create_app(
        _config(tmp, **cfg),
        client_factory=lambda: MockClient([make_mock_response(content="ok")]),
        enable_cors=False,
        api_keys=[TOKEN],
    )
    return app, httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


async def check_1_per_tenant_cap(tmp: Path) -> tuple[str, str]:
    """Pre-seed 5 running jobs for the tenant (== per_tenant_max). /v1/jobs must
    429 too_many_jobs_per_tenant; /v1/media/jobs admits (202) = the bug."""
    app, _ = _build_app(tmp)
    for i in range(5):
        rec = app.state.job_registry.register(f"seed-{i}", "sess-seed", OWNER)
        rec.status = "running"
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r_jobs = await c.post("/v1/jobs", json={"message": "hi"}, headers=HEADERS)
        r_media = await c.post("/v1/media/jobs", json=MEDIA_BODY, headers=HEADERS)
    jobs_capped = r_jobs.status_code == 429 and _errcode(r_jobs) == "too_many_jobs_per_tenant"
    media_bypassed = r_media.status_code == 202
    verdict = "OPEN" if media_bypassed else "FIXED"
    evidence = (
        f"/v1/jobs -> {r_jobs.status_code} {_errcode(r_jobs)} (cap {'enforced' if jobs_capped else 'NOT enforced?!'}); "
        f"/v1/media/jobs -> {r_media.status_code} (per-tenant cap {'BYPASSED' if media_bypassed else 'enforced'})"
    )
    return verdict, evidence


async def check_2_global_cap(tmp: Path) -> tuple[str, str]:
    """max_concurrent=2, queue_depth=1; pre-seed 2 running + 1 queued so the
    global cap + queue are saturated. /v1/jobs must 429 queue_full;
    /v1/media/jobs admits (202) = the bug."""
    app, _ = _build_app(tmp, per_tenant=64, max_concurrent=2, queue_depth=1)
    for i in range(2):
        rec = app.state.job_registry.register(f"g-{i}", "sess-g", "anyowner")
        rec.status = "running"
    app.state.job_registry.enqueue_pending("g-queued")  # fill the queue
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r_jobs = await c.post("/v1/jobs", json={"message": "hi"}, headers=HEADERS)
        r_media = await c.post("/v1/media/jobs", json=MEDIA_BODY, headers=HEADERS)
    jobs_capped = r_jobs.status_code == 429 and _errcode(r_jobs) == "queue_full"
    media_bypassed = r_media.status_code == 202
    verdict = "OPEN" if media_bypassed else "FIXED"
    evidence = (
        f"/v1/jobs -> {r_jobs.status_code} {_errcode(r_jobs)} (global cap {'enforced' if jobs_capped else 'NOT enforced?!'}); "
        f"/v1/media/jobs -> {r_media.status_code} (global cap {'BYPASSED' if media_bypassed else 'enforced'})"
    )
    return verdict, evidence


async def check_3_unbounded_tracker(tmp: Path) -> tuple[str, str]:
    """Submit K media jobs from one tenant; _MediaJobTracker._jobs must grow to K
    (no eviction/TTL) = the DoS/memory surface. (JobRegistry, by contrast, is
    reaped by the job_ttl GC loop; the media tracker has no reaper.)"""
    import koboi.facade as facade

    app, _ = _build_app(tmp)
    K = 25
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        codes = await asyncio.gather(
            *(c.post("/v1/media/jobs", json=MEDIA_BODY, headers=HEADERS) for _ in range(K))
        )
        await asyncio.sleep(0.2)  # let the background _run_media_job tasks settle
    admitted = sum(1 for r in codes if r.status_code == 202)
    tracker_size = len(app.state.media_jobs._jobs)
    unbounded = tracker_size == K  # every accepted media job still present, nothing evicted
    verdict = "OPEN" if unbounded else "FIXED"
    evidence = (
        f"submitted K={K}; admitted={admitted}; tracker._jobs size={tracker_size} "
        f"(no eviction: {'YES — grows unbounded' if unbounded else 'no'})"
    )
    return verdict, evidence


async def check_4_fanout(tmp: Path) -> tuple[str, str]:
    """Each bypassed-admission media submit spawns a REAL background generation
    (agent.media_generate). Count invocations to prove K billed-generations fan
    out from one tenant with zero cap."""
    import koboi.facade as facade

    counts = {"n": 0}
    orig = facade.KoboiAgent.media_generate

    async def counting(self, req):
        counts["n"] += 1
        return await orig(self, req)

    facade.KoboiAgent.media_generate = counting  # type: ignore[assignment]
    try:
        app, _ = _build_app(tmp)
        K = 10
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            await asyncio.gather(
                *(c.post("/v1/media/jobs", json=MEDIA_BODY, headers=HEADERS) for _ in range(K))
            )
            await asyncio.sleep(0.3)  # let the background tasks invoke media_generate
    finally:
        facade.KoboiAgent.media_generate = orig  # type: ignore[assignment]
    verdict = "OPEN" if counts["n"] == K else "FIXED"
    evidence = (
        f"submitted K={K} media jobs; agent.media_generate invoked {counts['n']}x "
        f"({counts['n']} real generations fanned out from ONE tenant, uncapped)"
    )
    return verdict, evidence


async def main() -> int:
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="exp-media-"))
    checks = [
        ("CHECK 1: per-tenant cap (per_tenant_max=5) applied to /v1/media/jobs?", check_1_per_tenant_cap),
        ("CHECK 2: global cap (max_concurrent=2) applied to /v1/media/jobs?", check_2_global_cap),
        ("CHECK 3: _MediaJobTracker bounded (eviction/TTL)?", check_3_unbounded_tracker),
        ("CHECK 4: bypassed submits still fan out real generations?", check_4_fanout),
    ]
    print("=" * 78)
    print("experiment_media_admission_gaps.py — /v1/media/jobs admission-control gap")
    print("(same class as #50; 4th finding from the 2026-07-14 audit pass)")
    print("=" * 78)
    any_open = False
    for title, fn in checks:
        verdict, evidence = await fn(tmp)
        any_open = any_open or verdict == "OPEN"
        print(f"\n{title}")
        print(f"  VERDICT: {verdict}")
        print(f"  EVIDENCE: {evidence}")
    print("\n" + "=" * 78)
    print("SUMMARY:", "OPEN — admission gap reproduces on this build" if any_open else "all FIXED")
    print("=" * 78)
    return 1 if any_open else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
