# koboi/workflows/ -- Deterministic workflow export/import

## What this is
Freeze a koboi run into a self-contained, re-runnable config bundle (`WorkflowDefinition`), and
optionally capture its LLM response cache so a re-run is byte-identical **offline** (no API key).
The `workflow:` envelope carries provenance + a `DeterminismProfile`; it is stripped before `Config`
loads, so a bundle IS a valid koboi config plus metadata. `WORKFLOW_SCHEMA_VERSION = "1.0"`.

Three determinism tiers: **export** (frozen config bundle) → **capture** (bundle + response-cache
sidecar) → **replay** (offline, raise-on-cache-miss). Design: `docs/deterministic-workflow-export-strategy.md`.

## Key files
```
definition.py    WorkflowDefinition envelope (schema_version/name/description/provenance/config) +
                 DeterminismProfile (temperature/seed/top_p/model_pin/replay_mode; .merge()/.to_llm_overrides()) +
                 WorkflowProvenance; build_from_config_path/_text (read UN-interpolated source + redact) +
                 validate_workflow (honest determinism warnings) + parse_determinism + build_graph_snapshot
store.py         FileWorkflowStore -- filesystem store for the CLI; resolve_workflows_dir(scope) honors
                 KOBOI_WORKFLOWS_DIR, else ~/.koboi/workflows (user) or cwd/.koboi/workflows (project);
                 save/load/load_with_cache/list/delete; atomic writes
capture.py       capture_from_run (record -> freeze -> bundle; optionally freeze ResponseCache as a
                 sidecar) + prepare_captured_bundle (inject replay.mode/cache_dir) + validate_capture
cache_sidecar.py CacheSidecar Protocol + CacheSidecarManifest; DirectoryCacheSidecar (CLI; the sidecar
                 dir IS a valid cache_dir) + SqliteCacheSidecar (server; owner+name-scoped)
__init__.py      Re-exports + WORKFLOW_SCHEMA_VERSION
```

## Determinism knobs (config-side)
There is **no top-level `workflows:` config section**. Determinism is set in ordinary config:
- `orchestration.determinism:` `{temperature, seed, top_p, model_pin, replay_mode}` -- workflow-level
  default; a per-node `determinism:` on an `AgentDef` overrides via `DeterminismProfile.merge` (node
  wins). Applied to each node's `llm_config` (`setdefault` preserves explicit values); a string
  `llm_config` (a `providers:` ref) opts out.
- `orchestration.agents[].output_schema:` + `force_response_format_with_tools:` -- structured output
  on nodes (Gap A+B).
- `replay:` `{mode: live|cache|replay, cache_dir}` -- set by `koboi run --replay-mode`; `Config.with_replay`.

`model_pin` is part of the cache key, so a pinned model never collides with a different one.

## CLI / server / TUI surfaces
- CLI: `koboi export <config>` (-> bundle), `koboi import <bundle>`, `koboi workflows list|show|delete`,
  `koboi capture <config> --with-cache`, `koboi run --workflow <name> --replay-mode {live|cache|replay}`
  (`--clear-cache` in cache mode).
- Server: `POST/GET/GET{name}/DELETE /v1/workflows`; `POST /v1/jobs/{id}/capture`; `workflow_ref` +
  `replay_mode` on `POST /v1/jobs`. Stored in `koboi/server/workflow_store.py` (SQLite, owner-scoped).
- TUI: `/capture [name] [--with-cache] [--redact-cache]`.

## Conventions
- The envelope wins: a `workflow:` key always overrides a stale same-named body key (`to_bundle_dict`).
- `export` reads the **un-interpolated** merged source (keeps `${VAR}` templates); the capture/server
  path uses `build_from_config_text` on the raw text.
- Both export and capture redact via `koboi.redact.redact_config_for_export` before persisting.

## Gotchas
- **`replay` mode needs no API key** but raises `CacheMissError` on any uncached call
  (`CacheMissPolicy.RAISE`); `cache` mode live-fetches + stores on miss; `live` (default) is uncached.
  Precedence: `replay.mode` > `orchestration.determinism.replay_mode` > `live`.
- **Anthropic has no `seed`** -- `validate_workflow` flags it; only caching + record-replay give true
  reproducibility on hosted APIs.
- **Plain (non-`workflow_ref`) jobs cannot isolate a run cache** -- only `workflow_ref` jobs freeze a
  per-job cache sidecar; `replay_mode` on a plain job builds a fresh per-job agent.
- **Two separate stores**: server `workflow_store` is SQLite in `koboi_memory.db` (owner-scoped); the
  CLI store is filesystem (`KOBOI_WORKFLOWS_DIR`).
