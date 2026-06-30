"""Shared fixtures for E2E integration tests against live Docker deployment.

Required env vars:
    KOBOI_HOST      Base URL (default: http://localhost)
    KOBOI_API_KEY   Valid API key (required — no default)

Run:
    KOBOI_API_KEY=koboi_xxx pytest tests/e2e/test_smoke.py -v
    KOBOI_API_KEY=koboi_xxx pytest tests/e2e/ -v
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import httpx
import pytest

BASE_URL = os.environ.get("KOBOI_HOST", "http://localhost")
API_KEY = os.environ.get("KOBOI_API_KEY", "")

# Stamp each pytest session with a unique run id so results are preserved per-run
# in tests/e2e/results/run_<timestamp>/ (not overwritten). Override with
# E2E_RUN_ID=... to group/label a run explicitly.
if not os.environ.get("E2E_RUN_ID"):
    from datetime import datetime

    os.environ["E2E_RUN_ID"] = datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _headers(**extra: str) -> dict:
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    h.update(extra)
    return h


@pytest.fixture
async def client():
    """httpx AsyncClient pointed at the live server."""
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=300) as c:
        yield c


@pytest.fixture(autouse=True, scope="session")
def _write_scenario_summary():
    """Aggregate per-scenario JSON into results/summary.json at session end."""
    yield
    try:
        from tests.e2e.framework.scenario import save_summary_from_disk

        save_summary_from_disk()
    except Exception:  # summary is best-effort; never fail the suite on it
        pass


async def create_session(client: httpx.AsyncClient) -> str:
    """Create a new session and return its id."""
    r = await client.post("/v1/sessions", headers=_headers())
    assert r.status_code == 201, f"session create failed: {r.status_code} {r.text}"
    return r.json()["session_id"]


async def stream_chat(
    client: httpx.AsyncClient,
    message: str,
    session_id: str | None = None,
    timeout: float = 180,
) -> list[dict]:
    """POST /chat/stream and collect all SSE events as parsed dicts.

    Returns a list of event dicts; the last element is "[DONE]" (string).
    """
    headers = _headers()
    if session_id:
        headers["X-Session-Id"] = session_id

    events: list = []
    async with client.stream(
        "POST", "/v1/chat/stream", json={"message": message}, headers=headers, timeout=timeout
    ) as r:
        assert r.status_code == 200, f"chat_stream failed: {r.status_code}"
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                events.append("[DONE]")
                break
            events.append(json.loads(payload))
    return events


async def poll_job(
    client: httpx.AsyncClient,
    job_id: str,
    timeout: float = 30,
) -> dict:
    """Poll GET /jobs/:id until terminal status. Returns the final JSON."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = await client.get(f"/v1/jobs/{job_id}", headers=_headers())
        body = r.json()
        if body["status"] in ("completed", "failed", "timed_out", "cancelled"):
            return body
        await asyncio.sleep(0.5)
    pytest.fail(f"Job {job_id} did not reach terminal status within {timeout}s")


def get_content(events: list[dict]) -> str:
    """Extract the 'content' field from the 'complete' event."""
    for ev in events:
        if isinstance(ev, dict) and ev.get("type") == "complete":
            return ev.get("content", "")
    return ""


def get_event_types(events: list[dict]) -> list[str]:
    """Extract the 'type' field from each event."""
    return [ev["type"] if isinstance(ev, dict) else ev for ev in events]
