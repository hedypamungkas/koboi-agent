#!/usr/bin/env bash
# Run all CI gates locally with CI-faithful tooling (.venv).
#
# Gate 0 refreshes the editable install + dev toolchain (mirrors ci.yml's
# `pip install -e ".[dev,tui,api]"`). This is load-bearing:
#   - it (re)installs mypy (~=1.19.1) + bandit (>=1.9) so gates 3-4 can run;
#   - it rebuilds the editable dist-info from the CURRENT pyproject, so
#     test_version_matches_pyproject stops false-red-ing on a stale install.
# All 6 gates must pass before tagging a release.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "❌ $PY not found. Create the CI-faithful venv first:"
  echo "   python3 -m venv .venv && .venv/bin/pip install -e \".[dev,tui,api]\""
  exit 1
fi

echo "=== 0. Refresh editable install + dev toolchain (mirrors ci.yml) ==="
"$PY" -m pip install -e ".[dev,tui,api]" --quiet \
  || { echo "❌ editable install / toolchain refresh failed"; exit 1; }

echo "=== 1. ruff check ==="
.venv/bin/ruff check koboi/ --select E,F,W --ignore E501,E402 \
  || { echo "❌ ruff check failed"; exit 1; }

echo "=== 2. ruff format --check ==="
.venv/bin/ruff format --check koboi/ \
  || { echo "❌ ruff format failed"; exit 1; }

echo "=== 3. mypy (~=1.19.1, via dev extra) ==="
"$PY" -m mypy koboi/ --ignore-missing-imports --no-strict-optional \
  || { echo "❌ mypy failed"; exit 1; }

echo "=== 4. bandit (>=1.9, via dev extra — matches CI) ==="
.venv/bin/bandit -r koboi/ -c pyproject.toml -q \
  || { echo "❌ bandit failed"; exit 1; }

echo "=== 5. pytest (e2e self-skips, cov>=80) ==="
KOBOI_HOST=http://127.0.0.1:9 KOBOI_API_KEY= \
  "$PY" -m pytest --tb=short -q --cov=koboi --cov-report=term-missing --cov-fail-under=80 \
  || { echo "❌ pytest failed"; exit 1; }

echo ""
echo "=========================="
echo " ALL 6 GATES GREEN ✅"
echo "=========================="
echo "Ready to tag. Rehearse first with:"
echo "  .claude/skills/koboi-release/scripts/bump-and-tag.sh --dry-run X.Y.Z \"message\""
echo "Then release:"
echo "  .claude/skills/koboi-release/scripts/bump-and-tag.sh X.Y.Z \"message\""
