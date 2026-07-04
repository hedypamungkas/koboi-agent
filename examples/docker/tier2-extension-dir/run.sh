#!/usr/bin/env bash
# Tier 2: mount an extensions directory — custom Python tool, no rebuild.
# Proves: the dir is on sys.path in-container (KOBOI_EXTENSIONS_DIR) and a @tool loads.
set -euo pipefail
cd "$(dirname "$0")"

docker run --rm \
  -e KOBOI_EXTENSIONS_DIR=/app/ext \
  -v "$PWD:/app/ext:ro" \
  koboi-agent:exp \
  python -c "import koboi; import sys; assert '/app/ext' in sys.path, 'ext dir not on sys.path'; import my_ext; assert hasattr(my_ext.ext_greeting, '_tool_def'), '@tool not applied'; print('ext_greeting loaded; @tool applied ->', my_ext.ext_greeting('proof'))"

echo "PROVEN: Tier 2 — KOBOI_EXTENSIONS_DIR on sys.path; custom tool module loads (no rebuild)"
