"""Error handling E2E tests — SSE error frames, 404, 409."""

from __future__ import annotations


import pytest

from tests.e2e.conftest import _headers, poll_job


@pytest.mark.e2e
class TestErrors:
    async def test_job_404(self, client):
        """29. Unknown job_id → 404."""
        r = await client.get("/v1/jobs/nonexistent_job_xyz", headers=_headers())
        assert r.status_code == 404

    async def test_409_cancel_terminal(self, client):
        """30. Cancel a completed job → 409."""
        r = await client.post(
            "/v1/jobs",
            json={"message": "Say ok"},
            headers=_headers(),
        )
        job_id = r.json()["job_id"]
        result = await poll_job(client, job_id, timeout=30)
        assert result["status"] == "completed"

        r2 = await client.post(f"/v1/jobs/{job_id}/cancel", headers=_headers())
        assert r2.status_code == 409
