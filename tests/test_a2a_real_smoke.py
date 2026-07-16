"""Real-LLM cross-instance A2A smoke (env-gated).

Proves the headline use case with a LIVE LLM (Surplus gateway) over a REAL localhost socket
between two instances: agent A on instance X delegates to agent C on instance Y (served by a
real uvicorn process) via ``call_peer_agent``; C answers with the real LLM; A incorporates it;
and the W3C trace-id lands in BOTH instances' journals. A second test does the same under
verified-only (``org_secret``) with a real card fetch.

Env-gated: skipped unless ``OPENAI_API_KEY`` is set (so the default suite never makes real
calls or binds ports). Costs ~3-6 real calls. Run::

  OPENAI_API_KEY=... pytest tests/test_a2a_real_smoke.py -v -s
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import sqlite3

import httpx
import pytest
import uvicorn

from koboi import tracing_context as tc
from koboi.config import Config
from koboi.server.app import create_app

_LIVE = pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY (real-LLM smoke)")
_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
_DELEGATE_PROMPT = (
    "You are agent A. For ANY factual question from the user you MUST use the call_peer_agent "
    'tool to ask peer C first: pass calls=[{"peer": "C", "message": "<the user\'s question>"}]. '
    "Then report C's exact answer to the user verbatim."
)


def _llm() -> dict:
    llm = {"provider": "openai", "model": _MODEL, "api_key": os.environ["OPENAI_API_KEY"]}
    if os.environ.get("OPENAI_BASE_URL"):
        llm["base_url"] = os.environ["OPENAI_BASE_URL"]
    return llm


def _trace_ids(db_path: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT DISTINCT trace_id FROM steps WHERE trace_id IS NOT NULL").fetchall()
    conn.close()
    return {r[0] for r in rows}


def _used_tools(result) -> list[str]:
    return [c.name for c in result.tool_calls_made]


@contextlib.asynccontextmanager
async def _served(app):
    """Serve ``app`` on a free localhost port; yield its base URL. Real socket, real HTTP."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(server.serve())
    base = f"http://127.0.0.1:{port}"
    try:
        async with httpx.AsyncClient() as c:
            ready = False
            for _ in range(100):
                try:
                    if (await c.get(f"{base}/healthz")).status_code == 200:
                        ready = True
                        break
                except Exception:  # noqa: BLE001 -- not-yet-listening
                    pass
                await asyncio.sleep(0.1)
            if not ready:
                raise RuntimeError("app did not become ready on /healthz")
        yield base
    finally:
        server.should_exit = True
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


@_LIVE
class TestRealDelegation:
    async def test_a_delegates_to_c_over_real_socket_and_trace_spans_both(self, tmp_path):
        db_x, db_y = tmp_path / "x.db", tmp_path / "y.db"
        cfg_y = Config.from_dict(
            {
                "agent": {
                    "name": "C",
                    "mode": "chat",
                    "system_prompt": "You are agent C. Answer concisely.",
                    "max_iterations": 3,
                },
                "llm": _llm(),
                "memory": {"backend": "sqlite", "db_path": str(db_y)},
                "peers": {"enabled": True, "inbound_tokens": ["tok-y"]},
            }
        )
        app_y = create_app(cfg_y)  # real LLM (no client_factory)

        async with _served(app_y) as base_y:
            cfg_x = Config.from_dict(
                {
                    "agent": {"name": "A", "mode": "act", "max_iterations": 5, "system_prompt": _DELEGATE_PROMPT},
                    "llm": _llm(),
                    "memory": {"backend": "sqlite", "db_path": str(db_x)},
                    "peers": {
                        "enabled": True,
                        "allow_private_network": True,  # 127.0.0.1 is private
                        "peers": [{"name": "C", "url": base_y, "token": "tok-y"}],
                    },
                }
            )
            app_x = create_app(cfg_x)

            tc.begin_request("00-" + "a" * 32 + "-" + "b" * 16 + "-01")
            root = tc.current_trace_id()
            agent = await app_x.state.pool.get_or_create("smoke-x")
            result = await agent.run("What is the capital of France?")

            assert "call_peer_agent" in _used_tools(result), f"A did not delegate; tools: {_used_tools(result)}"
            assert "paris" in (result.content or "").lower(), f"C's answer not propagated: {result.content!r}"
            assert root in _trace_ids(str(db_x))  # caller journaled the trace
            assert root in _trace_ids(str(db_y))  # peer continued the SAME trace-id over the real socket

    async def test_verified_only_delegation_over_real_socket(self, tmp_path):
        """Same delegation, under verified-only (shared org_secret) with a real card fetch."""
        db_x, db_y = tmp_path / "x2.db", tmp_path / "y2.db"
        org = "real-smoke-secret"
        cfg_y = Config.from_dict(
            {
                "agent": {
                    "name": "C",
                    "mode": "chat",
                    "system_prompt": "You are agent C. Answer concisely.",
                    "max_iterations": 3,
                },
                "llm": _llm(),
                "memory": {"backend": "sqlite", "db_path": str(db_y)},
                "peers": {
                    "enabled": True,
                    "org": "acme",
                    "org_secret": org,
                    "public_base_url": "http://peer-y:8000",
                    "inbound_tokens": ["tok-y"],
                },
            }
        )
        app_y = create_app(cfg_y)

        async with _served(app_y) as base_y:
            cfg_x = Config.from_dict(
                {
                    "agent": {"name": "A", "mode": "act", "max_iterations": 5, "system_prompt": _DELEGATE_PROMPT},
                    "llm": _llm(),
                    "memory": {"backend": "sqlite", "db_path": str(db_x)},
                    "peers": {
                        "enabled": True,
                        "org": "acme",
                        "org_secret": org,
                        "allow_private_network": True,
                        "peers": [{"name": "C", "url": base_y, "token": "tok-y", "agent_name": "C"}],
                    },
                }
            )
            app_x = create_app(cfg_x)

            # Verified-only: C is gated until verify_all (real card fetch over the socket).
            assert app_x.state.peer_registry.get("C") is None
            n = await app_x.state.peer_registry.verify_all()
            assert n == 1 and app_x.state.peer_registry.get("C") is not None

            agent = await app_x.state.pool.get_or_create("smoke-x2")
            result = await agent.run("What is the capital of Japan?")
            assert "call_peer_agent" in _used_tools(result)
            assert "tokyo" in (result.content or "").lower()
