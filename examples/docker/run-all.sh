#!/usr/bin/env bash
# Build the koboi image once, then prove all 3 customization tiers (LLM-free).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HERE/../.."

echo "=== building koboi-agent:exp (this takes a few minutes the first time) ==="
docker build -t koboi-agent:exp "$REPO"
echo ""

"$HERE/tier1-config-mount/run.sh"
echo ""
"$HERE/tier2-extension-dir/run.sh"
echo ""
"$HERE/tier3-derived-image/run.sh"
echo ""
echo "=========================="
echo " ALL 3 TIERS PROVEN  ✅"
echo "=========================="
