#!/usr/bin/env bash
# Run all CI gates in a CI-FAITHFUL venv (only .[dev,tui,api], no eval extras).
#
# Why a throwaway venv (not the dev .venv): the dev .venv often has eval/rag
# extras (ragas/pandas/datasets) that pull numpy, whose PEP 695 `type` stubs
# break mypy under `python_version = "3.10"`. CI installs only .[dev,tui,api]
# (no numpy), so to truly mirror CI the gates must run in a matching venv --
# otherwise the gate false-reds locally while CI stays green. The venv is cached
# at $KOBOI_CI_VENV (default /tmp/koboi-release-venv); `rm -rf` it to force a
# clean rebuild (e.g. after a Python or dependency change).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

CI_VENV="${KOBOI_CI_VENV:-/tmp/koboi-release-venv}"

echo "=== 0. Ensure CI-faithful venv ($CI_VENV, .[dev,tui,api]) ==="
if [ ! -x "$CI_VENV/bin/python" ]; then
  # Bootstrap: find any Python 3.10+ (koboi requires it) to create the venv.
  # Bare system `python3` may be <3.10 (e.g. macOS 3.9.6), so prefer .venv/bin/python.
  BOOT_PY=""
  for c in .venv/bin/python python3.13 python3.12 python3.11 python3.10 python3; do
    resolved=$(command -v "$c" 2>/dev/null || true)
    [ -n "$resolved" ] || resolved="$c"
    if [ -x "$resolved" ] && "$resolved" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
      BOOT_PY="$resolved"; break
    fi
  done
  [ -n "$BOOT_PY" ] || {
    echo "❌ No Python 3.10+ found to build the CI venv. Create a .venv first:"
    echo "   python3.13 -m venv .venv && .venv/bin/pip install -e \".[dev,tui,api]\""
    exit 1
  }
  echo "  creating $CI_VENV via $BOOT_PY ($("$BOOT_PY" --version 2>&1))"
  "$BOOT_PY" -m venv "$CI_VENV"
  "$CI_VENV/bin/pip" install -q --upgrade pip
fi
# (Re)install to pick up pyproject changes + refresh the editable dist-info
# (so test_version_matches_pyproject never false-reds on a stale install).
"$CI_VENV/bin/pip" install -e ".[dev,tui,api]" --quiet \
  || { echo "❌ editable install / toolchain refresh failed"; exit 1; }

echo "=== 1. ruff check ==="
"$CI_VENV/bin/ruff" check koboi/ --select E,F,W --ignore E501,E402 \
  || { echo "❌ ruff check failed"; exit 1; }

echo "=== 2. ruff format --check ==="
"$CI_VENV/bin/ruff" format --check koboi/ \
  || { echo "❌ ruff format failed"; exit 1; }

echo "=== 3. mypy (~=1.19.1, via dev extra) ==="
"$CI_VENV/bin/python" -m mypy koboi/ --ignore-missing-imports --no-strict-optional \
  || { echo "❌ mypy failed"; exit 1; }

echo "=== 4. bandit (>=1.9, via dev extra — matches CI) ==="
"$CI_VENV/bin/bandit" -r koboi/ -c pyproject.toml -q \
  || { echo "❌ bandit failed"; exit 1; }

echo "=== 5. pytest (e2e self-skips, cov>=80) ==="
KOBOI_HOST=http://127.0.0.1:9 KOBOI_API_KEY= \
  "$CI_VENV/bin/python" -m pytest --tb=short -q --cov=koboi --cov-report=term-missing --cov-fail-under=80 \
  || { echo "❌ pytest failed"; exit 1; }

echo ""
echo "=========================="
echo " ALL 6 GATES GREEN ✅ (CI-faithful venv: $CI_VENV)"
echo "=========================="
echo "Ready to tag. Rehearse first with:"
echo "  .claude/skills/koboi-release/scripts/bump-and-tag.sh --dry-run X.Y.Z \"message\""
echo "Then release:"
echo "  .claude/skills/koboi-release/scripts/bump-and-tag.sh X.Y.Z \"message\""
