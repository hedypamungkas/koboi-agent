"""RAG retrieval E2E tests — queries against sample documents."""

from __future__ import annotations

import pytest

from tests.e2e.conftest import get_content, stream_chat


@pytest.mark.e2e
class TestRAG:
    async def test_rag_product_query(self, client):
        """10. Product catalog query returns correct pricing."""
        events = await stream_chat(client, "What is the price of AcmeCRM Business per user per month?")
        content = get_content(events).lower()
        assert "$25" in content or "25/user" in content or "25 per user" in content, (
            f"expected $25/user/month in answer: {content}"
        )

    async def test_rag_policy_query(self, client):
        """11. Company policy query references the policy document."""
        events = await stream_chat(client, "What is the remote work policy at Acme Corp?")
        content = get_content(events).lower()
        assert "remote" in content or "work from home" in content or "wfh" in content, (
            f"expected remote work content: {content}"
        )

    async def test_rag_hotel_query(self, client):
        """12. Hotel operations query returns room info from RAG."""
        events = await stream_chat(
            client,
            "What amenities does the Presidential Suite at Grand Plaza Hotel have?",
        )
        content = get_content(events).lower()
        assert any(kw in content for kw in ["butler", "jacuzzi", "ocean", "terrace", "piano", "$850"]), (
            f"expected Presidential Suite details from RAG: {content}"
        )

    async def test_rag_ecommerce_query(self, client):
        """13. E-commerce KB query returns return policy details."""
        events = await stream_chat(client, "What is the return policy at ShopWave? How many days do I have?")
        content = get_content(events)
        assert "30" in content, f"expected 30-day return window: {content}"
        assert "day" in content.lower(), f"expected 'day' in answer: {content}"
