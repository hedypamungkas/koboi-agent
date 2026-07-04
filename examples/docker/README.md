# Container customization — 3 tiers

The published koboi image (`ghcr.io/hedypamungkas/koboi-agent:<version>`) is a **base
layer**, not a lock-in. All three customization tiers work without rebuilding koboi
itself. Each subdirectory below is a standalone, LLM-free proof you can run locally.

First build the image once (or `docker pull` the published one):
```bash
docker build -t koboi-agent:exp .          # from repo root
```

## Tier 1 — mount a YAML config (built-in path)
Change behavior with **zero code and no rebuild** — just mount your agent YAML.
The entrypoint honors `KOBOI_CONFIG`, `KOBOI_HOST`, `KOBOI_PORT`.
```bash
cd tier1-config-mount && ./run.sh
```
Proves: the mounted YAML drove the app (`/openapi.json` title == `koboi-<agent.name>`).

## Tier 2 — mount an extensions directory (custom Python modules)
Add custom tools / RAG retrievers / context strategies via `tools.custom` /
`rag.custom_modules` / `context.custom_modules`. Set `KOBOI_EXTENSIONS_DIR` and the
dir is added to `sys.path` at `import koboi` (see `koboi/_extensions_path.py`), so the
`importlib.import_module` loaders find your modules — **no rebuild, no manual PYTHONPATH**.
```bash
cd tier2-extension-dir && ./run.sh
```
Proves: the mounted dir is on `sys.path` inside the container and a `@tool` module loads.

## Tier 3 — derive a new image (customize-by-code, Path B)
Full programmatic control via `create_app(config, extra_tools=…, extra_hooks=…,
approval_handler=…, extra_routes=…)`. `FROM` the koboi image, add your entrypoint.
```bash
cd tier3-derived-image && ./run.sh
```
Proves: a derived image with a custom `create_app` entrypoint exposes a custom route
(`/__tier3__` → 200) and wires an extra tool — no LLM call needed.

## Run all three
```bash
./run-all.sh        # builds the image + proves all 3 tiers
```

## When to use which
| Need | Tier |
|---|---|
| Different prompts/tools/mode/limits via YAML | 1 (mount config) |
| Custom tool / RAG retriever / context strategy as a Python module | 2 (mount extensions dir) |
| Custom approval handler, extra routes, `create_app(...)` composition, installed plugin packages | 3 (derive image) |

> Tiers 1–2 are mount-based (no rebuild). Tier 3 derives an image — the standard
> "extend the official image" pattern (like `FROM python:3.12`). Editing koboi's
> *internal* code isn't supported in any tier — use the extension points.
