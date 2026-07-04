#!/usr/bin/env bash
# Tier 1: mount a custom YAML config — no code, no rebuild.
# Proves: the mounted YAML drives the app (openapi title == koboi-tier1).
set -euo pipefail
cd "$(dirname "$0")"

docker run --rm -d --name koboi-tier1 -p 18080:8080 \
  -e KOBOI_CONFIG=/app/agent.yaml \
  -v "$PWD/agent.yaml:/app/agent.yaml:ro" \
  koboi-agent:exp
trap 'docker stop koboi-tier1 >/dev/null 2>&1 || true' EXIT

# wait for readiness
for _ in $(seq 1 30); do curl -sf http://localhost:18080/healthz >/dev/null && break; sleep 1; done

title=$(curl -s http://localhost:18080/openapi.json | python3 -c "import sys,json; print(json.load(sys.stdin)['info']['title'])")
echo "openapi info.title: $title"
if [ "$title" = "koboi-tier1" ]; then
  echo "PROVEN: Tier 1 — mounted YAML drove the app (no rebuild)"
else
  echo "FAIL: expected koboi-tier1, got $title"; exit 1
fi
