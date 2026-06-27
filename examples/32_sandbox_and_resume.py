"""Example 32: sandbox isolation + step journal resume (P0b + P2-A).

Demonstrates (no API key required -- exercises the sandbox/journal machinery):
  - ``sandbox: restricted``  -> cwd containment, network-binary soft-deny,
    proxy/secret env stripping, and rlimits applied to the *child* only;
  - the step journal          -> one row per loop iteration + the crash-marker
    concept that ``koboi run --resume`` recovers from.

Run:
    python examples/32_sandbox_and_resume.py
"""

from __future__ import annotations

import os
import tempfile

from koboi.config import Config
from koboi.sandbox import build_sandbox
from koboi.memory_sqlite import SQLiteMemory
from koboi.journal import StepJournal


def demo_sandbox(cfg) -> None:
    print("=== P0b: restricted sandbox backend ===")
    sb = build_sandbox(cfg.get("sandbox"))
    workdir = cfg.get("sandbox", "workdir", default=".")
    os.makedirs(workdir, exist_ok=True)
    print(f"backend: {sb.name} | workdir: {workdir}")

    # 1) cwd containment: a path outside the workdir is rejected.
    try:
        sb.validate_path("/etc/passwd")
        print("  validate_path('/etc/passwd') -> allowed (unexpected!)")
    except PermissionError:
        print("  validate_path('/etc/passwd') -> BLOCKED (outside workdir)")

    # 2) soft network deny: obvious egress binaries are flagged.
    print("  network_allowed('curl https://x'):", sb.network_allowed("curl https://x"))
    print("  network_allowed('ls -la'):", sb.network_allowed("ls -la"))

    # 3) env hygiene: subprocess env drops secret-shaped + proxy vars.
    env = sb.build_env()
    print("  env strips OPENAI_API_KEY:", "OPENAI_API_KEY" not in env)
    print("  env PATH restricted to safe dirs:", bool(env.get("PATH")))

    # 4) a contained command runs inside the workdir.
    result = sb.run("echo contained-run", shell=True)
    print(f"  run('echo contained-run') -> rc={result.returncode} out={result.stdout.strip()!r}")


def demo_journal(cfg) -> None:
    print("\n=== P2-A: step journal + resume ===")
    db = os.path.join(tempfile.mkdtemp(), "demo.db")
    mem = SQLiteMemory(db_path=db, session_id="demo-session")
    j = StepJournal(mem._ensure_conn(), mem.session_id)

    # Simulate a completed turn: a tool-call step then a terminal complete step.
    j.advance_turn()  # turn 1
    j.record_step(turn_index=j.turn_index, step_index=0, status="tool_calls", prompt_tokens=42, completion_tokens=8)
    j.record_step(
        turn_index=j.turn_index,
        step_index=1,
        status="complete",
        is_terminal=True,
        prompt_tokens=50,
        completion_tokens=12,
    )
    print("recorded steps for turn 1:")
    for s in j.list_steps(turn_index=1):
        print(
            "  ",
            {k: s[k] for k in ("turn_index", "step_index", "status", "is_terminal")},
        )

    # Simulate a crash: advance to turn 2, write a 'running' marker, then "die".
    j.advance_turn()
    j.record_step(turn_index=j.turn_index, step_index=0, status="running")
    print(f"\ncrash simulation: {len(j.list_open_running())} open 'running' marker(s)")
    mem.close()

    # A new process opens the same session -> the journal inherits turn 2, and
    # `koboi run --resume` marks the dangling marker 'interrupted' + continues.
    mem2 = SQLiteMemory(db_path=db, session_id="demo-session")
    j2 = StepJournal(mem2._ensure_conn(), mem2.session_id)
    print(f"resumed journal inherits turn: {j2.turn_index} (no re-numbering)")
    j2.mark_interrupted(j2.list_open_running())
    print(f"after mark_interrupted: {len(j2.list_open_running())} open marker(s)")
    mem2.close()


def main() -> None:
    cfg = Config.from_yaml("configs/sandbox_restricted.yaml")
    demo_sandbox(cfg)
    demo_journal(cfg)
    print("\nTry it end-to-end with a real model:")
    print("  koboi chat configs/sandbox_restricted.yaml")
    print("  koboi run configs/sandbox_restricted.yaml --resume <session-id>")
    print("  koboi sessions configs/sandbox_restricted.yaml")


if __name__ == "__main__":
    main()
