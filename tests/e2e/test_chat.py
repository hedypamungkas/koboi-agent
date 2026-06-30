"""Multi-turn chat, memory, body shapes, resume — E2E tests."""

from __future__ import annotations

import pytest

from tests.e2e.conftest import (
    create_session,
    get_content,
    get_event_types,
    stream_chat,
)


@pytest.mark.e2e
class TestChat:
    async def test_multi_turn_memory(self, client):
        """6. Second turn references first turn's answer (memory works)."""
        sid = await create_session(client)
        events1 = await stream_chat(client, "I love eating sushi for dinner.", session_id=sid)
        assert "complete" in get_event_types(events1)

        events2 = await stream_chat(client, "What do I love eating?", session_id=sid)
        content = get_content(events2).lower()
        assert "sushi" in content, f"memory lost — expected 'sushi': {content}"

    async def test_messages_array_shape(self, client):
        """7. Body with messages[] (OpenAI shape) is accepted."""
        headers = {"Content-Type": "application/json"}
        from tests.e2e.conftest import API_KEY

        if API_KEY:
            headers["Authorization"] = f"Bearer {API_KEY}"

        async with client.stream(
            "POST",
            "/v1/chat/stream",
            json={"messages": [{"role": "user", "content": "Say yes"}]},
            headers=headers,
            timeout=30,
        ) as r:
            assert r.status_code == 200
            text = (await r.aread()).decode()
        assert "complete" in text

    async def test_session_isolation(self, client):
        """8. Two sessions don't cross-contaminate (verified via messages API)."""
        from tests.e2e.conftest import _headers

        sid_a = await create_session(client)
        sid_b = await create_session(client)
        # Send distinct messages to each session.
        await stream_chat(client, "I love sushi.", session_id=sid_a)
        await stream_chat(client, "I love pizza.", session_id=sid_b)

        # Verify isolation via the messages API (not LLM-dependent).
        msgs_a = (await client.get(f"/v1/sessions/{sid_a}", headers=_headers())).json()["messages"]
        msgs_b = (await client.get(f"/v1/sessions/{sid_b}", headers=_headers())).json()["messages"]

        # Session A should contain "sushi" but NOT "pizza".
        text_a = " ".join(m.get("content", "") for m in msgs_a).lower()
        assert "sushi" in text_a, f"session A missing sushi: {text_a[:200]}"
        assert "pizza" not in text_a, f"session A leaked pizza: {text_a[:200]}"

        # Session B should contain "pizza" but NOT "sushi".
        text_b = " ".join(m.get("content", "") for m in msgs_b).lower()
        assert "pizza" in text_b, f"session B missing pizza: {text_b[:200]}"
        assert "sushi" not in text_b, f"session B leaked sushi: {text_b[:200]}"

    async def test_resume(self, client):
        """9. POST /resume returns content from persisted memory."""
        from tests.e2e.conftest import _headers

        sid = await create_session(client)
        await stream_chat(client, "My name is TestUser99", session_id=sid)

        r = await client.post(f"/v1/sessions/{sid}/resume", headers=_headers())
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert len(body["content"]) > 0
