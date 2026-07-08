"""Server-layer performance benchmarks (FastAPI SSE / jobs / pool / auth).

Measures server overhead with NO network and NO real LLM: the ASGI app is
driven in-process via ``httpx.ASGITransport`` (the repo's canonical pattern --
see tests/test_server_app.py), and the per-session LLM client is swapped for a
canned ``MockClient`` through ``create_app(client_factory=...)``. So these
measure auth middleware + routing + pool-lock + SSE encoding + job admission +
idempotency -- not network or model latency.

These are higher-variance than the pure-CPU micro-benchmarks (full ASGI
round-trip per round), so they are REPORTED in the CI summary + artifact but
gated loosely. Absolute NFR thresholds are intentionally generous (catch
egregious regressions only); the relative-compare layer is the proper gate for
these -- see docs/performance-benchmarking.md.

Async-under-benchmark: each round spins a fresh event loop via asyncio.run
(httpx AsyncClient cannot be reused across loops), mirroring bench_core.py.
"""

from __future__ import annotations

import asyncio
import hashlib

import httpx
from httpx import ASGITransport

from koboi.config import Config
from koboi.server import create_app
from koboi.server.auth import KeyStore
from koboi.server.idempotency import IdempotencyRegistry
from tests.conftest import MockClient, make_mock_response


# ---------------------------------------------------------------------------
# app/client builders
# ---------------------------------------------------------------------------


def _server_config() -> Config:
    """Minimal in-process server config: in-memory control plane, no auth."""
    return Config.from_dict(
        {
            "agent": {"name": "bench-srv", "system_prompt": "h", "max_iterations": 3},
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "test",
                "base_url": "http://localhost:8080/v1",
            },
            "memory": {"backend": "in_memory"},  # sidecar DB -> ":memory:" (no file I/O)
            "sandbox": {"backend": "restricted"},  # required by /v1/jobs admission
            "server": {"auth_required": False},  # dev-open (owner="dev")
        },
        validate=True,
    )


def _canned_client_factory():
    """Instant canned LLM client for in-process benches (no network)."""
    return MockClient([make_mock_response(content="hello")])


def _app(workspace_root: str):
    """A fresh in-process app with a canned MockClient (instant LLM, no network)."""
    return create_app(
        _server_config(),
        client_factory=_canned_client_factory,
        enable_cors=False,
        api_keys=None,
        workspace_root=workspace_root,
    )


# ---------------------------------------------------------------------------
# HTTP round-trip benchmarks (full ASGI path)
# ---------------------------------------------------------------------------


def test_server_healthz(benchmark, tmp_path):
    """Baseline ASGI routing overhead: GET /healthz (stateless, always open)."""
    app = _app(str(tmp_path / "ws"))

    def run_once():
        async def hit():
            async with httpx.AsyncClient(base_url="http://t", transport=ASGITransport(app=app)) as c:
                r = await c.get("/healthz")
                return r.status_code

        return asyncio.run(hit())

    assert benchmark(run_once) == 200


def test_server_chat_stream(benchmark, tmp_path):
    """Full SSE chat round-trip: POST /v1/chat/stream -> drain text/event-stream."""
    app = _app(str(tmp_path / "ws"))
    headers = {"X-Session-Id": "bench-sess"}  # pin session -> steady-state (agent built once)

    def run_once():
        async def hit():
            async with httpx.AsyncClient(base_url="http://t", transport=ASGITransport(app=app)) as c:
                async with c.stream("POST", "/v1/chat/stream", json={"message": "hi"}, headers=headers) as r:
                    await r.aread()
                    return r.status_code

        return asyncio.run(hit())

    assert benchmark(run_once) == 200


def test_server_job_admit(benchmark, tmp_path):
    """Job-record admission write: JobStore.insert (the sqlite write /v1/jobs
    performs on admission). Measured directly, NOT via HTTP: the HTTP path
    spawns a background autonomous job whose execution outlives the per-round
    event loop (asyncio.run closes it) and then blocks on per_tenant_max. Full
    /v1/jobs HTTP throughput (incl. execution) is a relative-compare follow-up.
    """
    app = _app(str(tmp_path / "ws"))
    store = app.state.job_store
    counter = [0]

    def run_once():
        counter[0] += 1
        store.insert(f"job-{counter[0]}", "bench-sess", "dev", "do something")

    benchmark(run_once)


# ---------------------------------------------------------------------------
# Subsystem benchmarks (direct component access -- lower variance)
# ---------------------------------------------------------------------------


def test_server_pool_get_warm(benchmark, tmp_path):
    """Hot-path session reuse: get_or_create on an already-warm session (lock + lookup)."""
    app = _app(str(tmp_path / "ws"))
    pool = app.state.pool
    asyncio.run(pool.get_or_create("warm-sess"))  # pre-create so we measure reuse, not construction

    def run_once():
        return asyncio.run(pool.get_or_create("warm-sess"))

    result = benchmark(run_once)
    assert result is not None


def test_server_idempotency_check(benchmark):
    """Dedup-hit fast path: check_and_record on an already-seen key (returns False).

    This is the production hot path (retried /chat/stream requests). Reusing one
    key keeps ``_seen`` at size 1, which matters because pytest-benchmark auto-runs
    ~10^5 rounds for a microsecond op and ``_purge`` is O(N) -- a fresh key every
    round would be O(N**2) total and never finish.
    """
    reg = IdempotencyRegistry(ttl_seconds=600, max_entries=10000)
    reg.check_and_record("bench-key")  # seed -> seen

    result = benchmark(reg.check_and_record, "bench-key")  # sync -> False (dedup hit)
    assert result is False


def test_server_auth_validate(benchmark):
    """Bearer-key verification cost: KeyStore.validate (hmac.compare_digest on SHA256)."""
    ks = KeyStore()
    token = "koboi_bench_token_0123456789abcdef0123456789abcdef"
    ks._keys[hashlib.sha256(token.encode()).hexdigest()] = "k1"

    # sync method -> hand directly to benchmark (no asyncio.run needed)
    key_id = benchmark(ks.validate, token)
    assert key_id == "k1"
