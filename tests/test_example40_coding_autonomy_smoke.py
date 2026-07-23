"""tests/test_example40_coding_autonomy_smoke.py -- CI smoke test for example 40.

Example 40 (the "invisible engineering" coding-autonomy showcase) runs the full
Wave 0-4 coding stack against a real temp git repo in --mock mode: offline, $0,
deterministic, with the coding tools genuinely executing. Because the example
now exits non-zero when any per-phase verification FAILS (new commit + clean
tree + green tests + PR observed), running it in --mock IS a regression gate --
if the sandbox/policy/patch-parser/git/github wiring breaks, this test fails.

Run as a subprocess (the example is a click script, not an importable module).
Needs the [tui] extra (click + rich); skipped otherwise. git must be on PATH.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "40_coding_autonomy_full_demo.py"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

_HAS_TUI = importlib.util.find_spec("click") is not None and importlib.util.find_spec("rich") is not None
_needs_tui = pytest.mark.skipif(not _HAS_TUI, reason="example 40 needs the [tui] extra (click + rich)")
_needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")


@pytest.fixture(scope="module")
def demo_mod():
    """Load example 40 as a module (it is a click script, not a package)."""
    if not _HAS_TUI:
        pytest.skip("example 40 needs the [tui] extra (click + rich)")
    spec = importlib.util.spec_from_file_location("demo40_mod", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@_needs_tui
def test_help_runs_clean():
    """`--help` exits 0 -- imports + click wiring are OK on the current install."""
    result = subprocess.run(
        [sys.executable, str(EXAMPLE), "--help"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert "Invisible Engineering" in result.stdout or "coding-agent" in result.stdout


@_needs_tui
@_needs_git
def test_mock_run_all_phases_verified_green():
    """The full --mock run exits 0 (all phases independently verified GREEN).

    This is the load-bearing assertion: example 40's own verification tables
    check real effects (commit landed, tree clean, pytest green, PR observed)
    and the script exits 1 if any FAIL. A green exit here means the whole
    coding tool-chain (repo_map/apply_patch/edit_file/run_typecheck/run_shell/
    git_*/github_*) executed correctly against a real repo, offline.
    """
    result = subprocess.run(
        [sys.executable, str(EXAMPLE)],  # default = --mock
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=300,
    )
    combined = result.stdout + result.stderr
    # Exit 0 is the real gate; the message is a human-readable corroboration.
    assert result.returncode == 0, f"example 40 --mock failed (exit {result.returncode}):\n{combined[-3000:]}"
    assert "verified GREEN" in combined, combined[-2000:]
    # No verification row may have FAILed.
    assert "verification checks FAILED" not in combined, combined[-2000:]


class TestTransportHijackGuard:
    """The guard that catches the httpx-hijack regression class (a demo mock
    stealing koboi's LLM transport). It must pass on a clean wiring and fire
    loudly when httpx.AsyncClient is globally monkeypatched."""

    def test_guard_passes_on_real_transport(self, demo_mod):
        from koboi.llm.auth import BearerAuth
        from koboi.llm.http_transport import HttpTransport

        class _Impl:
            _transport = HttpTransport(base_url="https://x", auth=BearerAuth("k"))

        agent = type("A", (), {})()
        agent._core = type("C", (), {})()
        agent._core.client = _Impl()
        # Should not raise.
        demo_mod._assert_llm_transport_not_hijacked(agent)

    def test_guard_catches_global_httpx_hijack(self, demo_mod):
        import httpx

        from koboi.llm.auth import BearerAuth
        from koboi.llm.http_transport import HttpTransport

        class _Impl:
            _transport = HttpTransport(base_url="https://x", auth=BearerAuth("k"))

        agent = type("A", (), {})()
        agent._core = type("C", (), {})()
        agent._core.client = _Impl()

        real = httpx.AsyncClient
        httpx.AsyncClient = lambda *a, **k: real(*a, **k)  # the old bug: a factory, not the class
        try:
            with pytest.raises(AssertionError, match="httpx.AsyncClient has been monkeypatched"):
                demo_mod._assert_llm_transport_not_hijacked(agent)
        finally:
            httpx.AsyncClient = real

    def test_scoped_github_mock_leaves_llm_transport_intact(self, demo_mod):
        """The real _install_mock_github must NOT touch httpx.AsyncClient."""
        import httpx

        before = httpx.AsyncClient
        undo: list = []
        prs: list = []
        demo_mod._install_mock_github(undo, prs)
        try:
            assert httpx.AsyncClient is before, "mock github hijacked global httpx.AsyncClient"
        finally:
            for fn in undo:
                fn()
