"""tests/test_hitl_client.py -- #7 smoke tests for the HITL client.

The client must be importable on a bare install (httpx + stdlib only) and must
not pull in click/rich/fastapi/koboi. These tests load the module by file path
(so ``examples/`` need not be a package) and check the pure helpers.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "hitl_client.py"

_FORBIDDEN = ("click", "rich", "fastapi", "textual", "koboi")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("hitl_client_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_imports_only_base_deps():
    """No import LINE may reference click/rich/fastapi/textual/koboi (bare-install-safe)."""
    for raw in EXAMPLE.read_text().splitlines():
        line = raw.strip()
        if line.startswith("import ") or line.startswith("from "):
            for bad in _FORBIDDEN:
                assert bad not in line, f"hitl_client imports {bad!r}: {line!r}"


def test_auth_headers_env_driven(monkeypatch, mod):
    monkeypatch.delenv("KOBOI_API_KEY", raising=False)
    assert mod._auth_headers() == {}
    monkeypatch.setenv("KOBOI_API_KEY", "koboi_test")
    assert mod._auth_headers() == {"Authorization": "Bearer koboi_test"}


def test_help_runs_clean():
    """`python examples/hitl_client.py --help` exits 0 (imports + argparse OK)."""
    result = subprocess.run([sys.executable, str(EXAMPLE), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "HITL" in result.stdout
