# Contributing to koboi-agent

## Dev setup

```bash
git clone <repo>
cd koboi-agent
pip install -e ".[dev,tui]"
pytest  # should pass
```

## Running tests

```bash
pytest                        # all
pytest tests/test_config.py   # single file
pytest -k "hook"              # by keyword
pytest --cov=koboi            # with coverage
```

## Lint, type, and security gates (required by CI)

```bash
ruff check koboi/             # lint (E,F,W,B,UP,C4)
ruff format --check koboi/    # format check
mypy koboi/                   # type check
bandit -r koboi/ -c pyproject.toml   # static security scan
pip-audit --strict            # dependency vulnerability audit
```

## Branch naming

```
feature/<short-description>
fix/<issue-number>-<short-description>
refactor/<what-is-refactored>
```

## Commit format

```
<type>: <short summary>
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `style`, `ci`, `perf`, `build`

## Adding an LLM provider

1. Create `koboi/llm/<provider>_adapter.py` implementing `LLMClient` ABC
2. Add a `_create_<provider>()` factory function in `factory.py` and register it via `ProviderRegistry.register(ProviderDescriptor(...))` in `koboi/llm/registry.py:register_builtin_providers()`
3. Declare env var fallbacks on the `ProviderDescriptor` (`env_key_api`, `env_key_base_url`, `env_key_model`) when registering the provider in `koboi/llm/registry.py:register_builtin_providers()` — `_build_client()` in facade.py no longer resolves env vars.
4. Add tests in `tests/test_llm_<provider>_adapter.py`
5. Add example in `examples/`

## Adding a builtin tool

1. Create `koboi/tools/builtin/<name>.py`
2. Use `@tool()` decorator with JSON Schema parameters
3. Register in `koboi/tools/builtin/__init__.py:register_all()`
4. Add tests in `tests/test_<name>_tools.py`

See `.claude/skills/creating-tools.md` for the full pattern.

## Adding a hook

See `.claude/skills/creating-hooks.md` for the full pattern.

## Adding an eval scorer

See `koboi/eval/CLAUDE.md` for the pattern.

## Understanding the codebase

For a comprehensive architecture overview (subsystem dependency graph, agent loop lifecycle, hook event flow, tool pipeline, extension points, and all major subsystems), see **[docs/architecture.md](docs/architecture.md)**.
