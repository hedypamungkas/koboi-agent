#!/usr/bin/env bash
# Full E2E suite v2 — ~144 parametrized scenarios + smoke/chat/rag/skills/tools/jobs.
# Each run writes to its own timestamped folder tests/e2e/results/run_<ts>/
# (preserving history); latest.txt points to the most recent run.
#
# Throttling (avoid provider rate limits):
#   E2E_THROTTLE_SECONDS        delay between turns        (default 1.0)
#   E2E_INTER_SCENARIO_SECONDS  delay between scenarios    (default 2.0)
# Strictness:
#   E2E_STRICT=1   keyword/tool misses also fail pytest (default: recorded only)
# Filters:
#   E2E_CATEGORY=rag            run one category (multi_turn|multi_tool|rag|skills|jobs|stress)
#   E2E_NAME=hotel              run scenarios whose name contains substring
#
# Usage:
#   KOBOI_API_KEY=koboi_xxx ./tests/e2e/run_full.sh
#   KOBOI_API_KEY=koboi_xxx E2E_CATEGORY=rag ./tests/e2e/run_full.sh
#   KOBOI_API_KEY=koboi_xxx E2E_STRICT=1 ./tests/e2e/run_full.sh

set -euo pipefail

export KOBOI_HOST="${KOBOI_HOST:-http://localhost}"
export KOBOI_API_KEY="${KOBOI_API_KEY:?KOBOI_API_KEY is required}"
export E2E_THROTTLE_SECONDS="${E2E_THROTTLE_SECONDS:-1.0}"

echo "Running full E2E suite v2 against $KOBOI_HOST ..."
echo "  Throttle: ${E2E_THROTTLE_SECONDS}s/turn  Strict: ${E2E_STRICT:-0}"
echo "  Results:  tests/e2e/results/<scenario>.json + summary.json"
echo ""

# Security-edge + custom tests are fast; run the full directory.
python3 -m pytest tests/e2e/ -v --tb=short -p no:cacheprovider "$@"

echo ""
echo "=== Scenario summary ==="
# Resolve the latest run folder (timestamped subfolder) via latest.txt, falling
# back to the most recently modified run_* dir.
RUN_DIR=""
if [[ -f tests/e2e/results/latest.txt ]]; then
  CAND="tests/e2e/results/$(cat tests/e2e/results/latest.txt)"
  [[ -d "$CAND" ]] && RUN_DIR="$CAND"
fi
if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR=$(ls -dt tests/e2e/results/run_* 2>/dev/null | head -1)
fi
SUMMARY="$RUN_DIR/summary.json"
if [[ -f "$SUMMARY" ]]; then
  echo "Results dir: $RUN_DIR"
  SUMMARY="$SUMMARY" python3 -c "
import json, os
s = json.load(open(os.environ['SUMMARY']))
print(f\"total={s['total']}  passed={s['passed']}  failed={s['failed']}  skipped={s['skipped']}  blocked={s.get('blocked',0)}\")
print(f\"duration={s['total_duration']}s  tokens={s['total_tokens']}\")
for cat, c in sorted(s['categories'].items()):
    print(f\"  {cat:12} {c['passed']}/{c['total']}  ({c['duration']}s)\")
print('Failures:')
for sc in s['scenarios']:
    if not sc['passed'] and not sc['error'].startswith('SKIPPED'):
        print(f\"  - {sc['category']}/{sc['name']}: {sc['error']}\")
" 2>/dev/null || cat "$SUMMARY"
else
  echo "(no summary.json found — did any scenario write results?)"
fi
