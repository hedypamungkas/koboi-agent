# Release gotchas — lessons from v0.3.0–v0.4.2

Each release surfaced a distinct issue. This catalog ensures the same mistakes aren't repeated.

## v0.3.0 — First release (agent-hardening)

### Gotcha: `release.yml` test job installed `.[dev]` (missing `tui` + `api`)
- **Symptom**: release.yml test job FAILED with `ModuleNotFoundError: No module named 'textual'/'click'/'fastapi'`.
- **Root cause**: release.yml's test job `pip install -e ".[dev]"` — missing the `tui` extra (textual/rich/click) and `api` extra (fastapi). pytest collects ALL test files including TUI + server tests → collection errors.
- **Fix**: release.yml test job: `.[dev]` → `.[dev,tui,api]`.
- **Lesson**: CI test job and release test job MUST install the SAME extras. ci.yml uses `.[dev,tui]` (+ now `[api]`); release.yml must match.

### Gotcha: Trusted Publishing not configured
- **Symptom**: PyPI publish failed with `invalid-publisher: valid token, but no corresponding publisher`.
- **Root cause**: PyPI's Trusted Publishing (OIDC) wasn't configured for the repo/workflow/environment.
- **Fix**: Register on PyPI (one-time): koboi-agent → Settings → Publishing → Add publisher: repo=`hedypamungkas/koboi-agent`, workflow=`release.yml`, environment=`pypi`.
- **Lesson**: Trusted Publishing must be configured ONCE before the first release.yml publish. Subsequent releases work automatically.

## v0.4.0 — Serving layer (feature/sse-server)

### Gotcha: e2e tests committed but no self-skip
- **Symptom**: CI test job would fail on `tests/e2e/` with `httpx.ConnectError` (no live server in CI).
- **Root cause**: `tests/e2e/conftest.py` had no skip-when-no-server guard. e2e tests are deployment-integration (need a live Docker server + API key).
- **Fix**: Added `_require_live_server` autouse fixture in `tests/e2e/conftest.py` that pings `/healthz`; if unreachable, `pytest.skip()` the whole suite.
- **Lesson**: Deployment-integration tests MUST self-skip in CI (no live server). Guard with a healthz ping, not just an env-var check (a dev server might be up without the key).

### Gotcha: CI test job missing `[api]` extra
- **Symptom**: Server test modules error on `import fastapi` (ModuleNotFoundError).
- **Root cause**: ci.yml test job installed `.[dev,tui]` but NOT `[api]` (fastapi). Server tests need it.
- **Fix**: ci.yml all 3 jobs + release.yml test job: `.[dev,tui]` → `.[dev,tui,api]`.
- **Lesson**: Any new extra required by tests MUST be in the CI install line.

## v0.4.1 — Container customization (experiment branch)

### Gotcha: GHCR image tag naming
- **Symptom**: `docker pull ghcr.io/.../koboi-agent:0.4.1` → "not found".
- **Root cause**: `docker/metadata-action` with `type=ref,event=tag` produces the GIT TAG name (`v0.4.1` with `v`), not the PyPI version (`0.4.1` without `v`).
- **Fix**: Use `docker pull ghcr.io/.../koboi-agent:v0.4.1` (with `v`) or `:latest`.
- **Lesson**: GHCR image tags mirror git tags (`vX.Y.Z`). PyPI versions don't have the `v`. Document both clearly.

### Gotcha: GHCR packages private by default
- **Symptom**: `docker pull` fails for non-authenticated users.
- **Root cause**: GHCR packages are private on creation. Must set public in package settings.
- **Fix**: Package settings → Danger Zone → Change visibility → Public. (Turned out it was already public for this repo — likely inherited repo visibility.)
- **Lesson**: After first GHCR push, verify the package is public (check settings or `docker manifest inspect` without login).

## v0.4.2 — Docker default-config fix (patch)

### Gotcha: Docker default config crashes on bare run
- **Symptom**: `docker run ghcr.io/.../koboi-agent:v0.4.1` (no args) → `ValueError: Config validation failed: llm.model is required`.
- **Root cause**: Default config was `e2e_full.yaml` whose `llm.model: ${OPENAI_MODEL:}` is empty without that env var → Pydantic validation fails.
- **Fix**: Dockerfile default config: `e2e_full.yaml` → `server_simple.yaml` (concrete `model: gpt-4o-mini`). Bare run now gives clean C1 fail-closed guidance ("set KOBOI_API_KEYS") instead of a validation traceback.
- **Lesson**: The Docker image's DEFAULT config must be runnable (concrete llm.model). Env-parameterized configs are for deployments with env vars set, not for the bare-run default.

## General gotcha: Bandit version mismatch
- **Symptom**: Local bandit says "green" but CI bandit fails (or vice versa).
- **Root cause**: CI installs latest bandit (1.9.4+); local system might have 1.8.6 (fewer rules). 1.9.4 added B105 on `ANTHROPIC_AUTH_TOKEN` env-var name (false positive, needs nosec).
- **Fix**: Always use `.venv/bin/bandit` (matches CI version). If a new CI-only bandit rule fires, add a `# nosec BXXX` with justification (like `nosec B105` on env-var names).
- **Lesson**: Local validation tooling MUST match CI versions. Use `.venv` (Python 3.13 + current ruff/bandit), not system Python 3.9 + stale tools.
