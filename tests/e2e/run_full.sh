#!/usr/bin/env bash
# Full E2E suite — ~10-15 min, 30 tests covering all features.
# Requires the e2e_full.yaml config deployed to the Docker container.
#
# Usage:
#   KOBOI_API_KEY=koboi_xxx ./tests/e2e/run_full.sh
#   KOBOI_API_KEY=koboi_xxx KOBOI_HOST=http://my-server.com ./tests/e2e/run_full.sh

set -euo pipefail

export KOBOI_HOST="${KOBOI_HOST:-http://localhost}"
export KOBOI_API_KEY="${KOBOI_API_KEY:?KOBOI_API_KEY is required}"

echo "Running full E2E suite against $KOBOI_HOST ..."
echo "  Tests: smoke + chat + rag + skills + tools + jobs + security + errors"
echo ""

python -m pytest tests/e2e/ -v --tb=short "$@"
