"""experiment_env_auth_key_leak.py -- issue #99: server Bearer keys in subprocess env.

Reproduces (or proves fixed) the leak of the server's OWN API keys
(``KOBOI_API_KEYS`` / ``KOBOI_API_KEYS_FILE``) into every subprocess the agent
spawns. Drives the REAL classes only -- ``koboi.harness.env.build_safe_env``, a
real ``RestrictedProcessBackend``, and the real ``run_shell`` tool. No network,
no LLM, no API key needed.

Run: .venv/bin/python scripts/experiments/experiment_env_auth_key_leak.py
Exit code: 0 = all checks PASS (leak closed), 1 = at least one FAIL (leak open).
"""

from __future__ import annotations

import os
import sys
import tempfile

from koboi.harness.env import SECRET_BLOCKLIST, build_safe_env
from koboi.sandbox.restricted import RestrictedProcessBackend
from koboi.tools.builtin.shell import run_shell

SENTINEL = "sk-koboi-admin-MASTER-SENTINEL"
KEYS_FILE = "/data/koboi-keys-SENTINEL.json"

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check_unit_filter() -> tuple[bool, str]:
    """build_safe_env() must drop KOBOI_API_KEYS despite the KOBOI_* allow-glob."""
    env = build_safe_env({})
    survives = env.get("KOBOI_API_KEYS")
    return (survives is None, f"build_safe_env()['KOBOI_API_KEYS'] = {survives!r}")


def check_keys_file_filter() -> tuple[bool, str]:
    """The pointer to the key material is stripped too."""
    env = build_safe_env({})
    survives = env.get("KOBOI_API_KEYS_FILE")
    return (survives is None, f"build_safe_env()['KOBOI_API_KEYS_FILE'] = {survives!r}")


def check_benign_koboi_vars_survive() -> tuple[bool, str]:
    """The agent still gets the non-secret KOBOI_* control vars."""
    env = build_safe_env({})
    kept = {k: env.get(k) for k in ("KOBOI_HOST", "KOBOI_PORT", "KOBOI_VERBOSE")}
    ok = all(v is not None for v in kept.values())
    return (ok, f"kept = {kept}")


def check_passthrough_escape_hatch() -> tuple[bool, str]:
    """The trusted-CI passthrough hatch must NOT restore the auth boundary."""
    env = build_safe_env({"env_passthrough": True})
    leaked = env.get("KOBOI_API_KEYS")
    other = env.get("OPENAI_API_KEY")  # ordinary secrets DO come back (documented)
    return (
        leaked is None and other is not None,
        f"passthrough: KOBOI_API_KEYS={leaked!r}, OPENAI_API_KEY restored={other is not None}",
    )


def check_real_subprocess(workdir: str) -> tuple[bool, str]:
    """End-to-end: a model-authored `echo $KOBOI_API_KEYS` must come back empty."""
    sandbox = RestrictedProcessBackend(workdir=workdir, network="deny")
    out = run_shell("echo LEAKED=[$KOBOI_API_KEYS]", cwd=workdir, _deps={"sandbox": sandbox})
    return (SENTINEL not in out, f"run_shell output = {out.strip()!r}")


def check_blocklist_plural_globs() -> tuple[bool, str]:
    """The plural globs that keep the next sibling var from reintroducing this."""
    missing = [g for g in ("*_KEYS", "*_TOKENS", "*_SECRETS") if g not in SECRET_BLOCKLIST]
    return (not missing, f"missing globs = {missing or 'none'}")


def main() -> int:
    os.environ["KOBOI_API_KEYS"] = SENTINEL
    os.environ["KOBOI_API_KEYS_FILE"] = KEYS_FILE
    os.environ["OPENAI_API_KEY"] = "sk-openai-sentinel"
    os.environ["KOBOI_HOST"] = "127.0.0.1"
    os.environ["KOBOI_PORT"] = "8080"
    os.environ["KOBOI_VERBOSE"] = "1"

    workdir = tempfile.mkdtemp(prefix="koboi_env_leak_")

    checks = [
        ("CHECK 1: build_safe_env strips KOBOI_API_KEYS", check_unit_filter),
        ("CHECK 2: build_safe_env strips KOBOI_API_KEYS_FILE", check_keys_file_filter),
        ("CHECK 3: benign KOBOI_* control vars still pass", check_benign_koboi_vars_survive),
        ("CHECK 4: env_passthrough does not restore the auth keys", check_passthrough_escape_hatch),
        ("CHECK 5: real run_shell subprocess cannot echo the keys", lambda: check_real_subprocess(workdir)),
        ("CHECK 6: plural secret globs present in SECRET_BLOCKLIST", check_blocklist_plural_globs),
    ]

    print("=" * 78)
    print("experiment_env_auth_key_leak.py — issue #99: server Bearer keys in subprocess env")
    print("=" * 78)
    failed = 0
    for title, fn in checks:
        ok, evidence = fn()
        failed += 0 if ok else 1
        print(f"\n{title}")
        print(f"  {PASS if ok else FAIL}")
        print(f"  EVIDENCE: {evidence}")

    print("\n" + "=" * 78)
    if failed:
        print(f"SUMMARY: {failed} check(s) FAILED — the auth-key leak reproduces on this build")
    else:
        print("SUMMARY: all checks PASS — the auth-key leak is closed on this build")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
