#!/usr/bin/env bash
# Quick scenario subset — ~5 min. A representative slice of each category plus
# all security-edge cases. Use to validate infra before the long full run.
#
# Usage:
#   KOBOI_API_KEY=koboi_xxx ./tests/e2e/run_quick.sh

set -euo pipefail

export KOBOI_HOST="${KOBOI_HOST:-http://localhost}"
export KOBOI_API_KEY="${KOBOI_API_KEY:?KOBOI_API_KEY is required}"

echo "Running quick E2E subset against $KOBOI_HOST ..."
# security edge (instant) + a few of each LLM category + custom-orchestration tests
python3 -m pytest tests/e2e/test_security_edge.py \
    -k "calc_basic_add or fs_write_read or rag_acme_crm or skill_hotel_inquiry or job_calc_autonomous or stress_concurrent_3_jobs or multiturn_memory_01 or test_mixed_workload or test_session_create_delete or test_job_idempotency" \
    tests/e2e/test_scenarios.py -v --tb=short -p no:cacheprovider "$@"
