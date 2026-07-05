#!/usr/bin/env bash
# Run all CI gates locally with CI-faithful tooling (.venv ruff/bandit, system mypy).
# All 5 must pass before tagging a release.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "=== 1. ruff check ==="
.venv/bin/ruff check koboi/ --select E,F,W --ignore E501,E402 || { echo "❌ ruff check failed"; exit 1; }

echo "=== 2. ruff format --check ==="
.venv/bin/ruff format --check koboi/ || { echo "❌ ruff format failed"; exit 1; }

echo "=== 3. mypy ==="
python -m mypy koboi/ --ignore-missing-imports --no-strict-optional || { echo "❌ mypy failed"; exit 1; }

echo "=== 4. bandit 1.9.4 ==="
.venv/bin/bandit -r koboi/ -c pyproject.toml || { echo "❌ bandit failed"; exit 1; }

echo "=== 5. pytest (e2e self-skip, cov≥80) ==="
KOBOI_HOST=http://127.0.0.1:9 KOBOI_API_KEY= \
  .venv/bin/python -m pytest --tb=short -q --cov=koboi --cov-report=term-missing --cov-fail-under=80 \
  || { echo "❌ pytest failed"; exit 1; }

echo ""
echo "=========================="
echo " ALL 5 GATES GREEN ✅"
echo "=========================="
echo "Ready to tag. Run: scripts/bump-and-tag.sh X.Y.Z \"message\""
