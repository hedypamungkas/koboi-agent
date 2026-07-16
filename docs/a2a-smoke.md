# A2A smoke tests — real LLM, real network

The cross-instance A2A feature has two **real-world** smoke tests (beyond the 75 unit/in-process
tests). Neither runs in the default `pytest` suite — they need a live LLM and (for the docker one)
Docker. They exist to prove the headline use case: **multiple koboi instances communicating over
real HTTP, with a real LLM deciding to delegate.**

## 1. In-process real-LLM smoke (`tests/test_a2a_real_smoke.py`)

Two koboi apps: X{A} in-process + Y{C} as a **real uvicorn server** on a free localhost port,
both using a **live LLM** (Surplus gateway). Agent A delegates to agent C over a **real socket**;
C answers with the real LLM; A incorporates the answer; and the W3C trace-id lands in **both**
instances' step journals. A second variant does the same under verified-only (`org_secret`) with a
real card fetch.

Env-gated (skipped unless `OPENAI_API_KEY` is set):

```bash
set -a; source .env; set +a   # OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL (Surplus)
pytest tests/test_a2a_real_smoke.py -v -s
```

Costs ~3-6 real LLM calls (~40s). This is the closest-to-prod validation that runs without Docker.

## 2. Multi-container docker smoke (`docker-compose.a2a.yml` + `scripts/a2a_docker_smoke.py`)

Two **separate docker containers** (`koboi-a2a-x` + `koboi-a2a-y`) that resolve each other by
service name on the compose network, each running `koboi serve` with its own config, a shared
`org_secret`, and a real LLM. The smoke drives A's `/v1/chat/stream` and asserts C's answer
streams back across containers — true multi-instance over real docker networking.

```bash
export OPENAI_API_KEY=... OPENAI_BASE_URL=https://api.surplusintelligence.ai/v1 OPENAI_MODEL=gpt-5.4-mini
python scripts/a2a_docker_smoke.py    # brings the stack up, runs, tears down
# or, if you brought it up yourself:
docker compose -f docker-compose.a2a.yml up -d --wait
python scripts/a2a_docker_smoke.py --no-up
```

Needs Docker + the LLM key. Builds the image on first run (the configs `a2a_docker_{x,y}.yaml`
are baked into the image). Manual / CI-nightly.

## What these prove (that the unit tests can't)
- The **LLM actually decides to delegate** (emits `call_peer_agent` with well-formed args) — the
  unit tests script the tool call via `MockClient`.
- **Real HTTP between instances** (localhost socket / docker network), not in-process `ASGITransport`.
- The full **verified-only** path with a real card fetch + HMAC org-claim.
- The **W3C trace-id spanning both instances** (visible by querying both `steps.trace_id` columns).

## Known limitations

### Sync RPC model (orphaned work)
The current A2A model is **synchronous blocking RPC**: A's `call_peer_agent` tool blocks on a single HTTP request to B. There is no progress streaming, heartbeat, or cancellation propagation. If A's `peer.timeout` fires before B finishes:
- A gets `[B] (FAILED: timeout)` and continues (error isolation).
- **B may continue running** (orphaned work + wasted LLM cost) — Starlette does not always cancel non-streaming handlers on client disconnect.
- B's ephemeral session is evicted when the run completes (eventually).

Future fix: **streaming peer-invoke** (Model 1) — B's `/v1/peer/invoke` returns SSE TextDelta events; A's `invoke_peer` reads the stream; A sees real-time progress; B auto-cancels on A's disconnect (Starlette cancels streaming generators). ~50 lines of code change.

### org_secret rotation
To rotate `peers.org_secret` across a mesh:
1. Update `peers.org_secret` in ALL instance configs.
2. Rolling restart (one instance at a time). Each restart re-builds its card (signed with the new secret) + re-verifies peers (who also have the new secret).
3. During the transition, instances with the OLD secret reject instances with the NEW (and vice versa) — brief unavailability per pair.
4. Future: grace period (accept both old + new secrets during the transition window).

## Security model

### Trust boundary: same-org
A2A assumes **same-org trust**: all instances sharing `peers.org_secret` can call each other, read each other's answers, and see each other's data. This is enforced by the HMAC org-claim (the agent-card is signed; a rogue instance without the secret is rejected). Cross-org A2A (different orgs, different trust levels) is a future concern requiring PII redaction + access control.

### Prompt injection from peers
A malicious peer could send a message that tricks B's agent into doing something harmful. **Layered defense**:
1. `AutonomousApprovalHandler` on the peer-invoke path denies DESTRUCTIVE tools (deny-by-default).
2. The C3 sandbox check refuses `passthrough` sandbox for agents configured in autonomous modes (act/auto/yolo).
3. `call_peer_agent` is `RiskLevel.SAFE` (no HITL approval needed — but the receiver's own tools still gate).
4. The receiver runs in its own **configured mode** (the caller cannot override it via `body.mode`).

### Data exfiltration
A peer can ask B for sensitive data (from B's RAG, memory, tools). B's agent may comply → the data flows back to the peer in the response content. **This is by design** for same-org peers (they share the org_secret + are trusted). For cross-org A2A (future): add PII redaction on the inbound message + data-access controls.

### Card enumeration
The `GET /.well-known/agent-card` endpoint is intentionally **open** (no auth) — it follows W3C well-known conventions for discoverability. It advertises `org`, `agent_name`, `agents`, `skills`, `peer_invoke_url`, `issued_at`, `signature`. It contains **no secrets** (no tokens, no org_secret). An attacker can enumerate instances + learn their capabilities, but cannot forge a card (HMAC) or call them (no token). Low risk.

### LLM cost drain
A malicious/compromised peer could flood B with `/v1/peer/invoke` calls to drain B's LLM budget. **Mitigated** (Wave 1): `peers.rate_limit_per_minute` (default 60) bounds calls per minute per peer token; `peers.max_concurrent_inbound` (default 10) bounds simultaneous calls; `PoolFull`→429 is the global backpressure.

## Network edge cases

### Slowloris (slow trickle)
httpx's `timeout` is per-read (the max time between bytes). A peer sending 1 byte every 29 seconds could keep a connection alive beyond the intended 30s timeout. **Mitigated** by the outer `asyncio.wait_for(timeout=peer.timeout)` ceiling — it's a hard total timeout regardless of trickle speed. For cross-internet: also use a reverse proxy (nginx, Caddy) with its own read/body timeouts.

### Redirects (3xx)
`invoke_peer` uses `follow_redirects=False` (httpx default). A 3xx response from a redirecting load balancer → `resp.json()` fails → `ValueError("peer returned non-JSON")` → clear error. If behind a redirecting LB, configure the LB to pass-through (not redirect) for `/v1/peer/invoke`.

### TLS / HTTPS
For cross-internet deployments, use `https://` peer URLs (the token + message content travel in the HTTP body — cleartext over HTTP). Self-signed or expired certs → httpx `SSLError` → `[B] (FAILED: ...)`. Use a CA-signed cert, or configure httpx to accept custom CAs. Future: mTLS (mutual TLS with org-scoped client certs) for machine identity.

