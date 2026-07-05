# Documentation audit checklist

## What to verify per file type

### README.md
- [ ] Features list: every feature has a bullet; no missing features; no removed features listed.
- [ ] Builtin tools list: count matches `koboi/tools/builtin/__init__.py:register_all()`.
- [ ] Install command: correct extras (`[dev,tui]`, `[api]`, `[all]`).
- [ ] CLI commands: every `koboi <subcommand>` in `cli.py` is documented.
- [ ] Config example: sections match `config_models.py` (`agent`, `llm`, `tools`, `context`, `rag`, `guardrails`, `harness`, `sandbox`, `journal`, `server`, `jobs`).
- [ ] Examples table: count matches `ls examples/*.py | wc -l`; ranges cover all numbered scripts.
- [ ] Architecture subsystems: every `koboi/<dir>/` is listed; every root-level module is mentioned.
- [ ] License: matches `LICENSE` file.
- [ ] Links: all `[text](path)` resolve to real files.

### CLAUDE.md (root)
- [ ] "What this is": mentions all major capabilities (CLI, TUI, server, RAG, hooks, sandbox, etc.).
- [ ] Directory map: `.py` file count matches `find koboi -name '*.py' | wc -l`.
- [ ] Directory map: every `koboi/<dir>/` is listed with a one-liner.
- [ ] Directory map: `tests/` count, `configs/` count, `examples/` count — all current.
- [ ] Quick commands: every CLI subcommand from `cli.py:main()` is listed.
- [ ] Code conventions: match the actual code style (check a few files).
- [ ] Gotchas: new conventions from recent merges are present.

### koboi/<subpackage>/CLAUDE.md (each)
- [ ] "Key files" block: every `.py` file in the subpackage is listed.
- [ ] No phantom files listed (files that were removed/renamed).
- [ ] Conventions/gotchas: match the actual code.
- [ ] Counts (hooks, screens, widgets, scorers, etc.): match `ls <dir>/*.py | wc -l`.

### configs/CLAUDE.md
- [ ] Available configs: every `configs/*.yaml` is listed.
- [ ] Top-level sections: every section in `config_models.py` (KoboiConfig) is listed.

### docs/architecture.md
- [ ] Subsystem dependency graph: includes all subsystems (incl. server, sandbox, journal).
- [ ] Config sections table: count + sections match `config_models.py`.
- [ ] Extension points table: includes all registries + `@register_*` decorators.
- [ ] Built-in providers/tools/hooks: counts match the code.
- [ ] References table ("I want to..."): links resolve + cover all subpackages with CLAUDE.md.

## Counting commands
```bash
# .py files in koboi/
find koboi -name '*.py' -not -path '*/__pycache__/*' | wc -l

# test files
find tests -name 'test_*.py' | wc -l

# configs
ls configs/*.yaml | wc -l

# examples
ls examples/*.py | wc -l

# CLAUDE.md coverage
find koboi -name 'CLAUDE.md' | sort

# CLI subcommands (from cli.py)
grep -oP "elif cmd == .(\w+)." koboi/cli.py | sort

# config sections (from config_models.py)
grep -oP "class (\w+Config)" koboi/config_models.py | sort
```
