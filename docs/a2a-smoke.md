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
