"""Skills discovery + activation E2E tests."""

from __future__ import annotations

import pytest

from tests.e2e.conftest import get_content, stream_chat


@pytest.mark.e2e
class TestSkills:
    async def test_skill_hotel_receptionist(self, client):
        """15. Hotel scenario triggers hotel_receptionist skill behavior."""
        events = await stream_chat(
            client,
            "I want to book a room at Grand Plaza Hotel for 2 adults. What options do you have and what are the rates?",
        )
        content = get_content(events).lower()
        assert "$" in content, f"expected rates with $ in answer: {content}"
        assert any(kw in content for kw in ["room", "night", "standard", "deluxe", "suite"]), (
            f"expected room-related content: {content}"
        )

    async def test_skill_customer_service(self, client):
        """16. Customer service scenario triggers customer_service skill behavior."""
        events = await stream_chat(
            client,
            "I received a damaged item from ShopWave. How do I return it?",
        )
        content = get_content(events).lower()
        assert any(kw in content for kw in ["return", "refund", "replace", "photo"]), (
            f"expected return/refund guidance: {content}"
        )
        assert "48" in content or "30" in content, f"expected time window mention: {content}"
