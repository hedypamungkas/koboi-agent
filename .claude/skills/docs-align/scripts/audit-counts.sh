#!/usr/bin/env bash
# Quick audit of codebase counts for doc alignment.
# Run from repo root. Compare outputs against doc claims.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "=== koboi/*.py count ==="
find koboi -name '*.py' -not -path '*/__pycache__/*' | wc -l

echo "=== test files ==="
find tests -name 'test_*.py' -not -path '*/__pycache__/*' | wc -l

echo "=== configs ==="
ls configs/*.yaml 2>/dev/null | wc -l

echo "=== examples ==="
ls examples/*.py 2>/dev/null | wc -l

echo "=== CLAUDE.md coverage (koboi/ subpackages) ==="
find koboi -name 'CLAUDE.md' | sort

echo "=== subpackages WITHOUT CLAUDE.md ==="
for d in koboi/*/; do
  [ -f "$d/CLAUDE.md" ] || echo "  MISSING: $d"
done

echo "=== CLI subcommands ==="
grep -oE "elif cmd == \"[a-z_-]+\"" koboi/cli.py 2>/dev/null | sed 's/elif cmd == "//  /;s/"//' | sort

echo "=== config sections (Pydantic models) ==="
grep -oE "class \w+Config" koboi/config_models.py 2>/dev/null | sort

echo "=== builtin tools ==="
grep -oE '"[a-z_]+"' koboi/tools/builtin/__init__.py 2>/dev/null | tr -d '"' | sort || echo "(check register_all)"

echo "=== HookEvent values ==="
grep -oE "[A-Z_]+," koboi/hooks/chain.py 2>/dev/null | tr -d ',' | sort || echo "(check chain.py)"

echo "=== TUI screens ==="
ls koboi/tui/screens/*.py 2>/dev/null | xargs -n1 basename | sort

echo "=== TUI widgets ==="
ls koboi/tui/widgets/*.py 2>/dev/null | xargs -n1 basename | sort
