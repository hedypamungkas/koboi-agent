"""Autonomous job E2E tests — submit, poll, idempotency, stream, ownership."""

from __future__ import annotations

import pytest

from tests.e2e.conftest import API_KEY, _headers, poll_job


@pytest.mark.e2e
class TestJobs:
    async def test_job_autonomous_complete(self, client):
        """19. Job completes with real LLM answer in result_json."""
        r = await client.post(
            "/v1/jobs",
            json={"message": "What is the capital of France? One word."},
            headers=_headers(),
        )
        assert r.status_code == 202
        job_id = r.json()["job_id"]

        result = await poll_job(client, job_id, timeout=30)
        assert result["status"] == "completed"
        assert result["result"] is not None
        content = result["result"].get("content", "").lower()
        assert "paris" in content, f"expected 'paris' in result: {content}"

    async def test_job_idempotency(self, client):
        """20. Same Idempotency-Key returns same job_id."""
        key = f"idem-e2e-{__import__('time').time_ns()}"
        r1 = await client.post(
            "/v1/jobs",
            json={"message": "Say hello"},
            headers={**_headers(), "Idempotency-Key": key},
        )
        r2 = await client.post(
            "/v1/jobs",
            json={"message": "Say hello"},
            headers={**_headers(), "Idempotency-Key": key},
        )
        assert r1.json()["job_id"] == r2.json()["job_id"]

    async def test_job_stream_replay(self, client):
        """21. GET /jobs/:id/stream replays events + [DONE]."""
        import json

        r = await client.post(
            "/v1/jobs",
            json={"message": "What is 1+1? One word."},
            headers=_headers(),
        )
        job_id = r.json()["job_id"]
        await poll_job(client, job_id, timeout=30)

        events = []
        async with client.stream("GET", f"/v1/jobs/{job_id}/stream", headers=_headers(), timeout=10) as resp:
            assert resp.status_code == 200
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        events.append("[DONE]")
                        break
                    events.append(json.loads(payload))

        types = [e["type"] if isinstance(e, dict) else e for e in events]
        assert "complete" in types, f"expected complete in stream: {types}"
        assert types[-1] == "[DONE]"

    async def test_job_cross_owner_403(self, client):
        """22. Different key cannot access another's job."""
        if not API_KEY:
            pytest.skip("requires API keys configured")

        r = await client.post(
            "/v1/jobs",
            json={"message": "secret job"},
            headers=_headers(),
        )
        job_id = r.json()["job_id"]

        r2 = await client.get(
            f"/v1/jobs/{job_id}",
            headers={"Content-Type": "application/json", "Authorization": "Bearer wrong_key"},
        )
        assert r2.status_code in (401, 403)
