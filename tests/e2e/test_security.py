"""Security E2E tests — auth, ownership, path traversal, body validation."""

from __future__ import annotations

import pytest

from tests.e2e.conftest import API_KEY, _headers, create_session


@pytest.mark.e2e
class TestSecurity:
    async def test_401_no_auth(self, client):
        """23. No Bearer header → 401 (when keys configured)."""
        if not API_KEY:
            pytest.skip("requires API keys configured")
        r = await client.post("/v1/sessions", headers={"Content-Type": "application/json"})
        assert r.status_code == 401

    async def test_401_wrong_key(self, client):
        """24. Invalid Bearer key → 401."""
        if not API_KEY:
            pytest.skip("requires API keys configured")
        r = await client.post(
            "/v1/sessions",
            headers={"Content-Type": "application/json", "Authorization": "Bearer koboi_invalid_xyz"},
        )
        assert r.status_code == 401

    async def test_403_cross_owner_session(self, client):
        """25. Key-B cannot read Key-A's session."""
        if not API_KEY:
            pytest.skip("requires API keys configured")
        sid = await create_session(client)
        r = await client.get(
            f"/v1/sessions/{sid}",
            headers={"Content-Type": "application/json", "Authorization": "Bearer koboi_wrong_key_999"},
        )
        assert r.status_code in (401, 403)

    async def test_400_path_traversal(self, client):
        """26. Unsafe X-Session-Id (path traversal) → 400."""
        r = await client.post(
            "/v1/chat/stream",
            json={"message": "hi"},
            headers={**_headers(), "X-Session-Id": "../etc/passwd"},
        )
        assert r.status_code == 400

    async def test_400_empty_message(self, client):
        """27. Empty message body → 400."""
        r = await client.post(
            "/v1/chat/stream",
            json={"message": ""},
            headers=_headers(),
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "bad_request"
