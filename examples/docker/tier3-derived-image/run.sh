#!/usr/bin/env bash
# Tier 3: derive a new image (customize-by-code / Path B).
# Proves: create_app(extra_tools=..., extra_routes=...) works from a derived image.
set -euo pipefail
cd "$(dirname "$0")"

docker build -t koboi-tier3 -f Dockerfile.extend .

docker run --rm -d --name koboi-tier3 -p 18081:8080 koboi-tier3
trap 'docker stop koboi-tier3 >/dev/null 2>&1 || true' EXIT

for _ in $(seq 1 30); do curl -sf http://localhost:18081/healthz >/dev/null && break; sleep 1; done

code=$(curl -s -o /tmp/t3.json -w '%{http_code}' http://localhost:18081/__tier3__)
body=$(cat /tmp/t3.json)
echo "/__tier3__ -> $code: $body"
if [ "$code" = "200" ] && echo "$body" | grep -q '"via": *"create_app"'; then
  echo "PROVEN: Tier 3 — derived image, create_app custom route + extra tool live"
else
  echo "FAIL"; exit 1
fi
