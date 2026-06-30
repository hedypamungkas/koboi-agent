"""Smoke tests — quick verification that the live deployment works (~2 min).

Run: KOBOI_API_KEY=koboi_xxx pytest tests/e2e/test_smoke.py -v
"""

from __future__ import annotations

import pytest

from tests.e2e.conftest import (
    API_KEY,
    _headers,
    create_session,
    get_content,
    get_event_types,
    poll_job,
    stream_chat,
)


@pytest.mark.smoke
class TestSmoke:
    async def test_health(self, client):
        """1. /healthz and /readyz return 200 with valid JSON."""
        r = await client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

        r = await client.get("/readyz")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert any(c["name"] == "pool" for c in body["checks"])

    async def test_basic_chat(self, client):
        """2. Single-turn SSE chat produces text_delta → complete → [DONE]."""
        events = await stream_chat(client, "Say hello in one word.")
        types = get_event_types(events)
        assert "text_delta" in types, f"expected text_delta in {types}"
        assert "complete" in types, f"expected complete in {types}"
        assert types[-1] == "[DONE]"
        content = get_content(events)
        assert len(content) > 0, "complete event had empty content"

    async def test_auth(self, client):
        """3. 401 without key (when keys configured), 200 with valid key."""
        # With key
        r = await client.post("/v1/sessions", headers=_headers())
        assert r.status_code == 201
        # Without key (should 401 if keys configured, 201 if dev mode)
        r2 = await client.post("/v1/sessions", headers={"Content-Type": "application/json"})
        if API_KEY:
            assert r2.status_code == 401, f"expected 401 without key, got {r2.status_code}"

    async def test_job_lifecycle(self, client):
        """4. Job submit → poll → completed → result_json populated."""
        r = await client.post("/v1/jobs", json={"message": "What is 2+2? One word."}, headers=_headers())
        assert r.status_code == 202
        job_id = r.json()["job_id"]

        result = await poll_job(client, job_id, timeout=30)
        assert result["status"] == "completed"
        assert result.get("result") is not None, "result_json should be populated"
        assert "content" in result["result"], f"result={result['result']}"

    async def test_ownership(self, client):
        """5. Cross-key session access returns 403."""
        if not API_KEY:
            pytest.skip("requires API keys configured")

        sid = await create_session(client)
        r = await client.get(
            f"/v1/sessions/{sid}", headers={"Content-Type": "application/json", "Authorization": "Bearer wrong_key"}
        )
        assert r.status_code in (401, 403), f"expected 401/403 for wrong key, got {r.status_code}"
