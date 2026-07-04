#!/usr/bin/env bash
# scripts/reload-model.sh -- apply an .env model change to the running server.
#
# The server resolves llm.model from the OPENAI_MODEL env var ONLY at startup
# (there is no per-request model override), and `docker compose restart` does
# NOT re-read .env. So after editing .env you must RECREATE the container for
# the new model to take effect. This script does that, waits for the container
# healthcheck to turn green, prints the active model, and (optionally) runs a
# one-call chat smoke to prove the new model actually serves -- because /healthz
# can return 200 even when the model name is invalid for the endpoint.
#
# It works regardless of how the server is exposed (nginx or cloudflared) for
# the recreate + healthcheck + model print, which use the container directly.
# The optional chat smoke goes through KOBOI_HOST (default http://localhost).
#
# Usage:
#   1. edit .env  ->  OPENAI_MODEL=<new-model>
#      (and OPENAI_API_KEY / OPENAI_BASE_URL as needed)
#   2. ./scripts/reload-model.sh                         # no smoke
#      KOBOI_API_KEY=koboi_xxx ./scripts/reload-model.sh  # with chat smoke
set -euo pipefail

# Run from the repo root (where docker-compose.yml lives).
cd "$(dirname "$0")/.."

SERVICE="${KOBOI_SERVICE:-koboi}"
CONTAINER="${KOBOI_CONTAINER:-koboi-agent}"

echo ">> Recreating '${SERVICE}' to pick up the new .env (no image rebuild)..."
docker compose up -d "$SERVICE"

echo ">> Waiting for healthcheck to turn 'healthy' (container: ${CONTAINER})..."
ok=0
for _ in $(seq 1 40); do
  st="$(docker inspect -f '{{.State.Health.Status}}' "$CONTAINER" 2>/dev/null || echo "?")"
  if [ "$st" = "healthy" ]; then
    echo "   healthy"
    ok=1
    break
  fi
  sleep 2
done
if [ "$ok" -ne 1 ]; then
  echo "!! '${CONTAINER}' did not become healthy in time." >&2
  echo "   Inspect with: docker compose logs '${SERVICE}'" >&2
  exit 1
fi

echo ">> Active model (resolved by the server from OPENAI_MODEL):"
docker compose exec -T "$SERVICE" printenv OPENAI_MODEL

# Optional chat smoke -- only runs when KOBOI_API_KEY is exported. Makes one LLM
# call to confirm the (newly set) model actually returns content.
if [ -z "${KOBOI_API_KEY:-}" ]; then
  echo ">> (chat smoke skipped -- export KOBOI_API_KEY to enable it)"
else
  HOST="${KOBOI_HOST:-http://localhost}"
  echo ">> Chat smoke via ${HOST} (one LLM call to confirm the model serves)..."
  sid="$(curl -fsS -X POST "${HOST}/v1/sessions" \
        -H "Authorization: Bearer ${KOBOI_API_KEY}" \
        -H "Content-Type: application/json" \
        | sed -n 's/.*"session_id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
        2>/dev/null || true)"
  if [ -z "${sid:-}" ]; then
    echo "   !! could not create session (check KOBOI_API_KEY / KOBOI_HOST / server health)" >&2
    exit 1
  fi
  stream="$(curl -fsS -N --max-time 90 -X POST "${HOST}/v1/chat/stream" \
            -H "Authorization: Bearer ${KOBOI_API_KEY}" \
            -H "Content-Type: application/json" \
            -H "X-Session-Id: ${sid}" \
            -d '{"message":"Reply with exactly: OK"}' 2>/dev/null || true)"
  if printf '%s' "${stream}" | grep -qiE 'data: \[DONE\]|"text"|"delta"|"content"'; then
    echo "   chat smoke: OK (model replied)"
  else
    echo "   !! chat smoke FAILED -- model may be invalid for the endpoint" >&2
    printf '%s\n' "${stream}" | head -c 400 >&2
    echo "" >&2
    exit 1
  fi
fi

echo "Done."
