"""examples/hitl_client.py -- minimal human-in-the-loop SSE client (#7).

A dependency-light (httpx-only, base-install-safe) client that exercises the full
koboi server HITL loop:

    POST /v1/sessions                         -> session_id
    POST /v1/chat/stream  (SSE)               -> iteration/tool_call/...
        on "pending_approval" (concurrently):
    POST /v1/sessions/{id}/approve            -> resolves the approval
        ...stream continues until "data: [DONE]"

The server's interactive SSE path auto-wires an AsyncCallbackApprovalHandler, so
any DESTRUCTIVE tool call emits a ``pending_approval`` event. This client resolves
it (auto-approve by default; ``--deny`` to deny; drop the flag to be prompted).

Run (two terminals):

    # 1. serve a config whose agent can call a destructive tool:
    pip install -e ".[api]"
    koboi serve configs/hitl_demo.yaml          # loopback, auth_required:false

    # 2. drive it:
    python examples/hitl_client.py --message "delete the file workspace/old.log"

If your server uses auth (auth_required:true), set KOBOI_API_KEY to a key minted
via ``koboi keys create`` -- it is sent as ``Authorization: Bearer <key>``.

This example needs only the base install (httpx). It does NOT import click/rich.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import httpx


def _auth_headers() -> dict[str, str]:
    key = os.environ.get("KOBOI_API_KEY", "").strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


def _print(label: str, value: str = "", *, indent: str = "  ") -> None:
    text = f"{indent}{label}" if not value else f"{indent}{label}: {value}"
    print(text[:240])


async def _resolve_approval(
    base_url: str, headers: dict[str, str], session_id: str, approval_id: str, decision: str
) -> None:
    """POST /v1/sessions/{id}/approve on a fresh connection (concurrent with the stream)."""
    body = {"approval_id": approval_id, "decision": decision, "scope": "once"}
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        r = await client.post(f"/v1/sessions/{session_id}/approve", json=body, headers=headers)
    if r.status_code == 200:
        _print("approval resolved", f"{approval_id} -> {decision}")
    else:
        _print("approval resolve FAILED", f"{r.status_code} {r.text[:120]}")


async def _drive(base_url: str, message: str, auto_decision: str | None) -> None:
    headers = _auth_headers()

    async with httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(120.0)) as client:
        # 1. create session
        r = await client.post("/v1/sessions", headers=headers)
        r.raise_for_status()
        session_id = r.json().get("session_id") or r.headers.get("X-Session-Id", "")
        if not session_id:
            print("ERROR: no session_id in create response", file=sys.stderr)
            return
        _print("session", session_id, indent="")

        # 2. stream chat, resolving approvals concurrently as they arrive
        async with client.stream(
            "POST",
            "/v1/chat/stream",
            json={"message": message},
            headers={**headers, "X-Session-Id": session_id},
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                print(f"ERROR: chat/stream {resp.status_code}: {body[:200]!r}", file=sys.stderr)
                return

            tasks: list[asyncio.Task] = []
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or line.startswith(":"):  # keepalive comment or blank
                    continue
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: ") :]
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                etype = event.get("type", "")
                if etype == "iteration":
                    _print(f"iteration {event.get('iteration', '?')}")
                elif etype == "tool_call":
                    _print("tool_call", event.get("tool_name", "?"))
                elif etype == "tool_result":
                    result = str(event.get("result", ""))[:80]
                    _print("tool_result", result)
                elif etype == "pending_approval":
                    approval_id = event.get("approval_id", "")
                    _print(
                        "PENDING APPROVAL",
                        f"{event.get('tool_name')} risk={event.get('risk_level')} "
                        f"args={str(event.get('arguments'))[:60]}",
                    )
                    decision = auto_decision
                    if decision is None:
                        # interactive prompt
                        decision = await asyncio.to_thread(_prompt_decision, event.get("tool_name", "?"))
                    tasks.append(
                        asyncio.create_task(_resolve_approval(base_url, headers, session_id, approval_id, decision))
                    )
                elif etype == "complete":
                    _print("complete", indent="")
                elif etype == "error":
                    _print("ERROR event", str(event)[:200])

            # 3. wait for any in-flight approve POSTs to finish
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)


def _prompt_decision(tool_name: str) -> str:
    try:
        ans = input(f"  approve '{tool_name}'? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "deny"
    return "deny" if ans == "n" else "approve"


def main() -> int:
    p = argparse.ArgumentParser(description="koboi HITL SSE client (httpx-only)")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument(
        "--message",
        "-m",
        default="Delete the file workspace/old.log.",
        help="message to send (should make the agent call a destructive tool)",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--approve", action="store_const", const="approve", dest="decision")
    g.add_argument("--deny", action="store_const", const="deny", dest="decision")
    g.set_defaults(decision="approve")  # auto-approve by default
    args = p.parse_args()

    try:
        asyncio.run(_drive(args.base_url, args.message, args.decision))
    except httpx.ConnectError as e:
        print(f"ERROR: cannot reach {args.base_url}: {e}", file=sys.stderr)
        print("Start the server first:  koboi serve configs/hitl_demo.yaml", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
