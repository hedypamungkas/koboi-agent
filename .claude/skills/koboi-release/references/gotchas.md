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

## v0.4.3 — Security release (PR #5) + release-skill reliability pass

The release shipped cleanly, but running it surfaced 5 reliability gaps in THIS skill. All are
fixed (python3/venv interpreters, Gate 0 venv refresh, `--dry-run`, `verify-release.sh`, etc.).

### Gotcha: release scripts called bare `python` (aborts on macOS)
- **Symptom**: `pre-release-check.sh` (mypy) and `bump-and-tag.sh` (pyproject bump) aborted with `python: command not found` on macOS.
- **Root cause**: scripts invoked bare `python`; macOS ships only `python3` (the user's `python` is a zsh alias, invisible to non-interactive bash under `set -e`).
- **Fix**: scripts now prefer `.venv/bin/python`, fall back to `python3` — never bare `python`.
- **Lesson**: release scripts MUST use `python3` or a venv interpreter. A `python` alias is NOT inherited by non-interactive bash.

### Gotcha: stale editable install false-red the version test
- **Symptom**: `test_version_matches_pyproject` failed locally (`koboi.__version__=0.4.0` vs pyproject `0.4.2`).
- **Root cause**: the `.venv` editable install's dist-info lagged pyproject; nothing refreshed it.
- **Fix**: `pre-release-check.sh` Gate 0 now runs `pip install -e ".[dev,tui,api]"` (mirrors ci.yml) — refreshes the dist-info AND ensures mypy/bandit are present.
- **Lesson**: a local pre-release gate MUST refresh the editable install so `importlib.metadata` sees the current version.

### Gotcha: bandit was in NO optional-dependency group
- **Symptom**: a fresh `.venv` from `pip install -e ".[dev,tui,api]"` lacked bandit → Gate 4 aborts.
- **Root cause**: CI installs bandit separately (`pip install bandit`); the `dev` extra never declared it.
- **Fix**: added `bandit>=1.9` to the `dev` + `all` extras in pyproject.toml.
- **Lesson**: every tool a release gate invokes MUST be reachable via an extras group the gate installs.

### Gotcha: SKILL.md cited scripts at the wrong path + no dry-run
- **Symptom**: SKILL.md referenced `scripts/pre-release-check.sh` (repo-root relative) but the scripts live at `.claude/skills/koboi-release/scripts/`. Rehearsing the irreversible tag push was impossible.
- **Root cause**: stale relative paths; no `--dry-run` mode.
- **Fix**: SKILL.md uses repo-root-relative real paths; `bump-and-tag.sh --dry-run` bumps+reverts without commit/tag/push.
- **Lesson**: cite script paths unambiguously; offer a no-push rehearsal for irreversible actions.

### Gotcha: `gh run list --limit 1` raced + recovery doc was harmful
- **Symptom**: `gh run list --workflow=... --limit 1` could return the PREVIOUS release's run (false green); the recovery doc told operators to re-tag after any workflow failure (harmful if PyPI already succeeded).
- **Fix**: new `verify-release.sh` polls for the run filtered by `--branch vX.Y.Z`; recovery doc now prefers `gh run rerun <job>` and reserves re-tagging for a bad-version situation (PyPI is immutable).
- **Lesson**: identify runs by the triggering tag, not "latest"; never recommend re-tagging blindly.

## General gotcha: Bandit version mismatch
- **Symptom**: Local bandit says "green" but CI bandit fails (or vice versa).
- **Root cause**: CI installs latest bandit (1.9.4+); local system might have 1.8.6 (fewer rules). 1.9.4 added B105 on `ANTHROPIC_AUTH_TOKEN` env-var name (false positive, needs nosec).
- **Fix**: Always use `.venv/bin/bandit` (matches CI version). If a new CI-only bandit rule fires, add a `# nosec BXXX` with justification (like `nosec B105` on env-var names).
- **Lesson**: Local validation tooling MUST match CI versions. Use `.venv` (Python 3.13 + current ruff/bandit), not system Python 3.9 + stale tools.
