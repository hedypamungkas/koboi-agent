# Documentation writing guide

## Core principles
1. **Clear** — a new developer can understand it without prior context.
2. **Compact** — no padding. Every sentence carries information.
3. **Jargon-free** — no buzzwords ("enterprise-grade", "seamless", "robust"). Use plain language.
4. **LLM-friendly** — exact paths, code blocks, tables. LLMs parse structure better than prose.
5. **Accurate** — every claim verified against the code. No guessing.

## Good vs bad examples

### Bad (vague, jargon, padding)
> The framework provides a robust, enterprise-grade solution for seamlessly orchestrating
> multi-agent workflows with comprehensive guardrail integration and various extension points
> for enhanced flexibility.

### Good (specific, compact, accurate)
> koboi-agent runs multi-agent workflows via `koboi/orchestration/` (keyword/LLM/hybrid routing).
> Guardrails (input/output validation, rate limiting) are in `koboi/guardrails/`. Extend with
> `@tool()`, `@register_retriever()`, or `@register_hook` — see `koboi/server/CLAUDE.md`.

### Bad (wrong count, stale)
> `examples/` contains 28 numbered scripts.

### Good (verified count)
> `examples/` contains 32 numbered scripts (01–32) + `server_built_in.py` / `server_customize.py`.

## File-length targets
| File | Target | Max |
|------|--------|-----|
| README.md | ~150 lines | 250 |
| CLAUDE.md (root) | ~200 lines | 350 |
| `koboi/<dir>/CLAUDE.md` | ~60 lines | 150 |
| `configs/CLAUDE.md` | ~40 lines | 80 |
| `docs/architecture.md` | ~600 lines | 1000 |

## Imperative form (for CLAUDE.md / instructions)
```
# Correct (imperative)
Register new tools in `__init__.py:register_all()`.
Tool functions return `str`.

# Incorrect (second person)
You should register new tools in __init__.py.
Your tool functions should return str.
```

## What to avoid
- "etc." / "and more" / "various" — be specific or omit.
- Marketing language ("powerful", "cutting-edge", "revolutionary").
- Long paragraphs — use bullets or tables.
- Duplicated content across files — link instead.
- Unverified counts — always run `find`/`ls`/`wc -l` before writing a number.
- References to removed files/commands — check the code, not memory.
