#!/usr/bin/env bash
# Quick smoke tests — ~2 min, 5 tests.
# Verifies the live deployment is functional.
#
# Usage:
#   KOBOI_API_KEY=koboi_xxx ./tests/e2e/run_smoke.sh
#   KOBOI_API_KEY=koboi_xxx KOBOI_HOST=http://my-server.com ./tests/e2e/run_smoke.sh

set -euo pipefail

export KOBOI_HOST="${KOBOI_HOST:-http://localhost}"
export KOBOI_API_KEY="${KOBOI_API_KEY:?KOBOI_API_KEY is required}"

echo "Running E2E smoke tests against $KOBOI_HOST ..."
python3 -m pytest tests/e2e/test_smoke.py -v -m smoke --tb=short "$@"
