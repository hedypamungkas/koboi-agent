"""Tool execution E2E tests — calculator, filesystem via SSE events."""

from __future__ import annotations

import pytest

from tests.e2e.conftest import get_event_types, stream_chat


@pytest.mark.e2e
class TestTools:
    async def test_calculator(self, client):
        """17. Calculator tool produces correct result."""
        events = await stream_chat(client, "Use the calculator tool to compute 15 * 23 and tell me the result.")
        types = get_event_types(events)
        assert "complete" in types, f"expected complete event: {types}"
        # Check result in tool_result OR final content (LLM may compute directly).
        found = False
        for ev in events:
            if isinstance(ev, dict) and ev.get("type") in ("tool_result", "complete"):
                if "345" in ev.get("result", "") or "345" in ev.get("content", ""):
                    found = True
                    break
        assert found, (
            f"expected 345 in tool_result or complete: {[e.get('type') for e in events if isinstance(e, dict)]}"
        )

    async def test_filesystem_write_read(self, client):
        """18. Filesystem write + read tools execute and return results."""
        events = await stream_chat(
            client,
            "Create a file called e2e_test.txt with the content 'hello world', then read it back to verify.",
        )
        types = get_event_types(events)
        assert "tool_call" in types, f"expected tool_call events: {types}"
        assert "tool_result" in types, f"expected tool_result events: {types}"
        assert types[-1] == "[DONE]"
