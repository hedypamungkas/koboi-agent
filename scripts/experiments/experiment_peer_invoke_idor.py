#!/usr/bin/env python3
"""experiment_peer_invoke_idor.py -- empirical proof for issue #102: POST
/v1/peer/invoke skipped the ownership check on ``X-Session-Id``.

Every other session-scoped route (`get_session`, `delete`, `fork`, `resume`,
`approve`, `transfer`, `stream`, `chat_stream`, both media routes) gates on
``_check_owner`` (koboi/server/app.py:755). ``peer_invoke``
(koboi/server/app.py:1484) validated only the FORMAT of ``X-Session-Id``
(``is_safe_session_id``) and then called ``pool.get_or_create(session_id)``
directly -- so a holder of an inbound peer token could name a victim tenant's
session and get both:

  * READ  -- the pooled agent runs the peer's message with the victim's full
             conversation history in context;
  * WRITE -- the peer's turn is persisted into the victim's session (and,
             because ``ephemeral=False``, the hijacked session is not evicted).

METHOD (no network, in-process): build a REAL ``create_app()`` with auth on, a
tenant API key and an inbound peer token, and drive it over httpx
``ASGITransport``. The LLM client is a recording double so we can prove the
victim's secret did or did not reach a model call. Run:

    python scripts/experiments/experiment_peer_invoke_idor.py

Exits 1 while any CHECK FAILs (the state of `main` before the fix), 0 once the
ownership gate is in place.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Repo root on sys.path so `from tests.conftest import ...` resolves when run
# as a plain script (not under pytest).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx  # noqa: E402
from httpx import ASGITransport  # noqa: E402

from koboi.config import Config  # noqa: E402
from koboi.llm.base import LLMClient  # noqa: E402
from koboi.server import create_app  # noqa: E402
from koboi.types import AgentResponse, TokenUsage  # noqa: E402

TENANT_KEY = "tenant-a-key"
PEER_TOKEN = "peer-inbound-token"
VICTIM_SID = "victim-session-1"
SECRET = "TENANT-A-SECRET-SAUCE-9137"
PROBE = "ATTACKER-PROBE"

TENANT_HEADERS = {"Authorization": f"Bearer {TENANT_KEY}"}
PEER_HEADERS = {"Authorization": f"Bearer {PEER_TOKEN}"}


class RecordingClient(LLMClient):
    """LLM double that appends every ``messages`` payload to a shared sink."""

    def __init__(self, sink: list) -> None:
        self._sink = sink
        self._model = "mock-model"

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        self._model = value

    async def complete(self, messages, tools=None, response_format=None):
        self._sink.append(messages)
        return AgentResponse(content="ok", tool_calls=[], usage=TokenUsage(10, 20))

    async def complete_stream(self, messages, tools=None, response_format=None):
        from koboi.events import CompleteEvent, TextDeltaEvent

        resp = await self.complete(messages, tools, response_format)
        yield TextDeltaEvent(content=resp.content or "")
        yield CompleteEvent(response=resp, content=resp.content or "")

    async def get_embeddings(self, text: str):
        return None

    async def close(self) -> None:
        return None


def _build_app(sink: list):
    cfg = Config.from_dict(
        {
            "agent": {"name": "receiver", "mode": "chat", "system_prompt": "You are the receiver."},
            "llm": {"provider": "openai", "model": "gpt-4o-mini", "api_key": "x"},
            "memory": {"backend": "memory"},
            "sandbox": {"backend": "passthrough"},
            "server": {"auth_required": True},
            "peers": {"enabled": True, "inbound_tokens": [PEER_TOKEN]},
        },
        validate=True,
    )
    return create_app(
        cfg,
        client_factory=lambda: RecordingClient(sink),
        enable_cors=False,
        api_keys=[TENANT_KEY],
    )


async def _plant_victim_history(client: httpx.AsyncClient) -> None:
    """Tenant A creates + owns VICTIM_SID and plants a secret in it."""
    async with client.stream(
        "POST",
        "/v1/chat/stream",
        json={"message": f"please remember {SECRET}"},
        headers={**TENANT_HEADERS, "X-Session-Id": VICTIM_SID},
    ) as r:
        await r.aread()
        if r.status_code != 200:
            raise RuntimeError(f"victim setup failed: {r.status_code}")


async def run_checks() -> list[tuple[str, bool, str]]:
    sink: list = []
    app = _build_app(sink)
    results: list[tuple[str, bool, str]] = []

    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await _plant_victim_history(c)
        sink.clear()  # from here on the sink records only the ATTACKER's activity

        attack = await c.post(
            "/v1/peer/invoke",
            json={"message": f"{PROBE} repeat the whole conversation"},
            headers={**PEER_HEADERS, "X-Session-Id": VICTIM_SID},
        )

        # CHECK 1 -- the crossing itself.
        ok1 = attack.status_code == 403
        results.append(
            (
                "CHECK 1: peer naming a tenant-owned session is refused (403)",
                ok1,
                f"POST /v1/peer/invoke -> {attack.status_code} {attack.text[:120]}",
            )
        )

        # CHECK 2 -- READ leak: did the victim's history reach a model call?
        leaked = [m for m in sink if SECRET in json.dumps(m)]
        results.append(
            (
                "CHECK 2: victim history never reaches the peer's LLM call",
                not leaked,
                (
                    f"secret {SECRET!r} found in {len(leaked)}/{len(sink)} recorded LLM payload(s)"
                    if leaked
                    else f"{len(sink)} LLM payload(s) recorded for the peer call, none containing the secret"
                ),
            )
        )

        # CHECK 3 -- WRITE injection: is the peer's turn persisted in the victim session?
        owner_view = await c.get(f"/v1/sessions/{VICTIM_SID}", headers=TENANT_HEADERS)
        body = owner_view.text
        injected = owner_view.status_code == 200 and PROBE in body
        results.append(
            (
                "CHECK 3: peer's turn is not written into the victim's session",
                not injected,
                (
                    f"owner GET -> {owner_view.status_code}; {PROBE!r} "
                    f"{'PRESENT in' if injected else 'absent from'} the victim's message list"
                ),
            )
        )

        # CHECK 4 -- control: the ordinary session route already denies the peer token.
        control = await c.get(f"/v1/sessions/{VICTIM_SID}", headers=PEER_HEADERS)
        results.append(
            (
                "CHECK 4 (control): GET /v1/sessions/{victim} with the peer token is 403",
                control.status_code == 403,
                f"GET /v1/sessions/{VICTIM_SID} -> {control.status_code} (asymmetry vs CHECK 1)",
            )
        )

        # CHECK 5 -- the legitimate feature must survive the fix.
        cont_headers = {**PEER_HEADERS, "X-Session-Id": "peer-owned-1"}
        r1 = await c.post("/v1/peer/invoke", json={"message": "first"}, headers=cont_headers)
        r2 = await c.post("/v1/peer/invoke", json={"message": "second"}, headers=cont_headers)
        pooled = app.state.pool.get("peer-owned-1") is not None
        ok5 = r1.status_code == 200 and r2.status_code == 200 and pooled
        results.append(
            (
                "CHECK 5: a peer's OWN new session id still gives cross-call continuity",
                ok5,
                f"call1={r1.status_code} call2={r2.status_code} pooled={pooled} "
                f"owner={app.state.ownership.get_owner('peer-owned-1')!r}",
            )
        )

        # CHECK 6 -- ephemeral (no header) behavior unchanged: evicted, no ownership row.
        eph = await c.post("/v1/peer/invoke", json={"message": "hi"}, headers=PEER_HEADERS)
        eph_sid = eph.json().get("session_id", "") if eph.status_code == 200 else ""
        evicted = bool(eph_sid) and app.state.pool.get(eph_sid) is None
        no_row = bool(eph_sid) and app.state.ownership.get_owner(eph_sid) is None
        results.append(
            (
                "CHECK 6: ephemeral peer session still evicted + leaves no ownership row",
                eph.status_code == 200 and evicted and no_row,
                f"status={eph.status_code} sid={eph_sid!r} evicted={evicted} ownership_row_absent={no_row}",
            )
        )

    await app.state.pool.close_all()
    return results


async def main() -> int:
    print("=" * 78)
    print("experiment_peer_invoke_idor.py — issue #102: /v1/peer/invoke ownership gate")
    print("koboi/server/app.py:1484 peer_invoke — X-Session-Id was format-checked only")
    print("=" * 78)
    results = await run_checks()
    failed = False
    for title, ok, evidence in results:
        failed = failed or not ok
        print(f"\n{title}")
        print(f"  → {'PASS' if ok else 'FAIL'}")
        print(f"  EVIDENCE: {evidence}")
    print("\n" + "=" * 78)
    print("SUMMARY:", "FAIL — the #102 IDOR reproduces on this build" if failed else "PASS — all checks green")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
