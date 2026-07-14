"""Tests for B4 -- warm handoff digest (HandoffDigest).

The digest is a side-LLM summary over the transcript, redacted, never-raising. It
enriches ``HandoverEvent.summary`` (when empty) at the chat-path handover site.
Unit tests inject a mock client; the integration test monkeypatches ``digest`` to
verify the wiring (HandoverEvent.summary is enriched) without a real LLM call.
"""

from __future__ import annotations

import pytest

from koboi.server.handoff_digest import HandoffDigest
from koboi.types import AgentResponse, TokenUsage


class _MockJudge:
    """Scripted side-LLM client (mirrors GroundingGuardrail test double)."""

    def __init__(self, content: str, raise_: bool = False):
        self._content = content
        self._raise = raise_
        self.calls = 0

    async def complete(self, messages, tools=None, response_format=None):
        self.calls += 1
        if self._raise:
            raise RuntimeError("judge down")
        return AgentResponse(content=self._content, tool_calls=[], usage=TokenUsage(0, 0))

    async def get_embeddings(self, text):
        return None

    async def close(self):
        pass


def _digest(judge=None) -> HandoffDigest:
    hd = HandoffDigest(provider="openai", model="m", api_key="x")
    if judge is not None:
        hd._client = judge  # bypass lazy create_client
    return hd


class TestHandoffDigest:
    async def test_produces_summary(self):
        hd = _digest(_MockJudge("Customer wants a refund for order #1234."))
        summary = await hd.digest([{"role": "user", "content": "I want a refund"}])
        assert "refund" in summary

    async def test_redacts_secret_shapes(self):
        hd = _digest(_MockJudge("Their key is sk-live-abc123def456ghi789jklmnop"))
        summary = await hd.digest([{"role": "user", "content": "x"}])
        assert "sk-live" not in summary  # the secret shape is masked
        assert "REDACTED" in summary

    async def test_never_raises_on_judge_failure(self):
        hd = _digest(_MockJudge("", raise_=True))
        summary = await hd.digest([{"role": "user", "content": "x"}])
        assert summary == ""  # fail-soft, never breaks the handover

    async def test_empty_transcript_returns_empty(self):
        hd = _digest(_MockJudge("should not be called"))
        assert await hd.digest([]) == ""
        assert hd._client.calls == 0  # cost-gate: no judge call on empty

    async def test_no_client_returns_empty(self):
        # _get_client returns None (build failed) -> "".
        hd = HandoffDigest(provider="badprovider", model="x", api_key="")
        assert await hd.digest([{"role": "user", "content": "x"}]) == ""


class TestHandoffDigestWiring:
    """Integration: a handover with digest.enabled enriches HandoverEvent.summary."""

    async def test_handover_summary_enriched_when_empty(self, monkeypatch):
        pytest.importorskip("fastapi")
        import httpx
        from koboi.config import Config
        from koboi.server import create_app
        from tests.conftest import MockClient, make_mock_response, make_mock_tool_call

        # Monkeypatch the digest to a fixed string (no real LLM call).
        async def _fake_digest(self, messages):
            return "MOCK_DIGEST_CASE_CARD"

        monkeypatch.setattr(HandoffDigest, "digest", _fake_digest)

        cfg = Config.from_dict(
            {
                "agent": {"name": "srv", "system_prompt": "h", "max_iterations": 3},
                "llm": {"provider": "openai", "model": "gpt-4o-mini", "api_key": "test"},
                "memory": {"backend": "in_memory"},
                "sandbox": {"backend": "restricted"},
                "server": {"auth_required": False},
                "tools": {"builtin": ["transfer_to_human"]},
                "handover": {"digest": {"enabled": True}},
            },
            validate=True,
        )
        # B1 tool with summary="" -> the digest enriches it.
        handover_resp = make_mock_response(
            content="transferring",
            tool_calls=[make_mock_tool_call("transfer_to_human", {"reason": "complex case", "summary": ""})],
        )
        app = create_app(cfg, client_factory=lambda: MockClient([handover_resp]), enable_cors=False)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            async with c.stream(
                "POST", "/v1/chat/stream", json={"message": "help", "mode": "act"}, headers={"X-Session-Id": "s-b4"}
            ) as r:
                body = (await r.aread()).decode()
        # Find the handover event in the SSE + assert its summary was enriched.
        import json as _json

        for line in body.split("\n"):
            if line.startswith("data: ") and '"type":"handover"' in line:
                ev = _json.loads(line[6:])
                assert ev["summary"] == "MOCK_DIGEST_CASE_CARD"
                return
        pytest.fail("no handover event in stream")
