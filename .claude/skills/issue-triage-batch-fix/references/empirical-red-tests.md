# Empirical RED tests + reproducible harness

How to write a test that **fails today for the bug's reason** and passes after the fix — deterministically, not flakily — and how to ship a standalone harness anyone can re-run.

## The RED test (in the repo test suite)

Asserts the **fixed** behavior, so it flips red→green without rewriting. Write it first, run it, confirm the failure is the bug (not a setup error).

### Race / concurrency bugs (TOCTOU, missing caps)
A naive concurrent test is flaky. Force the worst-case schedule deterministically with a gate so every concurrent request parks before any proceeds:

```python
gate = asyncio.Event(); parked = {"n": 0}
real = app.state.pool.get_or_create
async def slow(sid):
    parked["n"] += 1
    if parked["n"] >= N: gate.set()
    await gate.wait()
    return await real(sid)
monkeypatch.setattr(AgentPool, "get_or_create", slow)  # or app.state.pool.get_or_create = slow
resp = await asyncio.gather(*(c.post("/endpoint", ...) for _ in range(N)))
assert sum(r.status_code == 202 for r in resp) <= CAP   # fails today: all N admitted
```
The gate makes it scheduling-independent: every submitter runs its admission check before any mutates state — exactly the window the race exploits.

### Missing-cap / bypass bugs
Pre-seed the registry to the cap, then one more request must be rejected while the buggy path admits it:
```python
for i in range(CAP):
    rec = registry.register(f"s{i}", "sess", owner); rec.status = "running"
r = await c.post("/v1/jobs", json={...}, headers=auth)
assert r.status_code == 429 and r.json()["error"]["code"] == "too_many_jobs_per_tenant"
```

### Platform-gated code (Linux-only, no-local-runtime)
When the real feature can't run locally (e.g. seccomp on macOS, Docker build), test the **logic** with a fake module injected into `sys.modules` (picked up by a function-local `import`):
```python
fake = type("M", (), {})()
fake.ALLOW = "ALLOW"; fake.ERRNO = lambda e: ("ERRNO", e); fake.KILL = None
fake.SyscallFilter = _FakeFilter
monkeypatch.setitem(sys.modules, "seccomp", fake)
monkeypatch.setattr("pkg.mod._HAS_SECCOMP", True)
with pytest.raises(RuntimeError, match="connect"): backend._seccomp_preexec()
```
Today: the swallowed error + `load()` returns normally (RED). After fix: it raises (GREEN). No subprocess, no Linux, fully deterministic.

### Identity / auth derivation
When a test needs a specific owner/tenant, derive it exactly as the code does (e.g. `owner = "env:" + hashlib.sha256(token.encode()).hexdigest()[:12]`) so assertions match.

## The reproducible harness (`experiment_<topic>.py`)

A standalone script (not a pytest) anyone runs to see the bug red / fix green. Hard rules:
- Drives **real** production classes — never mock the system under test; mock only external I/O (LLM, HTTP, the platform-gated module).
- No network, no real API keys.
- Deterministic (gates/pre-seeds, never `sleep`-races).
- Prints `CHECK N: <name>` → `VERDICT: OPEN|FIXED` → `EVIDENCE: <concrete status/counts>`.
- `sys.exit(1)` if any CHECK is OPEN (bug present), `0` if all FIXED — so it's a CI-able red/green gate that auto-flips when the fix lands.

Match the repo's probe convention if one exists (e.g. a `experiment_*.py` family with a standard header). See `examples/experiment_template.py`.

## Common RED-test pitfalls
- **Fails for the wrong reason**: an import/fixture error is not a reproduced bug. Always read the red output; if it's not the bug's symptom, fix the test.
- **Flaky concurrency**: never rely on timing; use an `asyncio.Event` gate or pre-seeding.
- **Mocks the wrong layer**: mock the LLM/HTTP/platform module, not the route/registry under test.
- **Asserts current (buggy) behavior**: assert the FIXED behavior so the test transitions cleanly.
