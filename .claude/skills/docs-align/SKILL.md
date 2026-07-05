---
name: docs-align
description: >-
  This skill should be used when the user asks to "align docs", "sync documentation",
  "update README", "update CLAUDE.md", "fix stale docs", "audit documentation",
  "ensure docs match code", or mentions any discrepancy between documentation files
  (README, CLAUDE.md, architecture.md, configs/CLAUDE.md, folder CLAUDE.mds, etc.)
  and the actual codebase. Spawns parallel exploration agents to map the code +
  audit every doc, then revises all stale documentation to match — compact, jargon-free,
  human-and-LLM-friendly. Gates new-doc creation via AskUserQuestion.
---

# docs-align — Align all documentation with the codebase

## Overview
Audit every documentation file (`.md`) against the actual codebase, identify discrepancies,
and revise stale content so all docs accurately reflect the current code. Documentation must
be **clear, compact, jargon-free** — friendly to both humans and LLMs.

## When to use
Triggered by: "align docs", "sync documentation", "update README", "fix stale docs", "ensure
docs match code", "audit documentation", or any mention of docs being out of date vs the code.
Typically run after a major feature merge, a refactor, or before a release.

## Phase 1 — Parallel exploration (3 agents, simultaneously)

Launch 3 `Explore` agents in a **single message** (parallel) to build a complete picture:

### Agent 1: Code map
Map the current codebase structure. Focus on:
- Directory tree (`koboi/`, `tests/`, `configs/`, `examples/`, `docs/`, top-level files).
- File/module counts (use `find ... | wc -l`).
- New modules, new config sections, new CLI commands, new endpoints since the last doc update.
- Key public APIs (classes, functions, decorators, entry points).

### Agent 2: Doc inventory + staleness audit
Read every `.md` file that represents the codebase and cross-check against the code:
- `README.md`, `CLAUDE.md` (root), every `koboi/<subpackage>/CLAUDE.md`.
- `configs/CLAUDE.md`, `docs/architecture.md`, `docs/*.md` (skip pure planning/gap-analysis docs).
- For each file: list concrete discrepancies (wrong counts, missing modules, stale commands,
  missing config sections, dead references, broken examples).
- Verify counts with shell commands (`find`, `ls`, `grep`).

### Agent 3: Missing-coverage assessment
Identify subpackages/modules that have NO `CLAUDE.md` or doc coverage but should:
- Which `koboi/<dir>/` lack a `CLAUDE.md`?
- Are there new features with zero documentation?
- Are there examples that no longer exist or new examples not listed?

## Phase 2 — Consolidate findings

After agents return, synthesize into a **prioritized gap list**:
- **P0**: Wrong facts (counts, commands, module names) — factual errors that mislead.
- **P1**: Missing coverage (new subsystem with no docs; missing from directory map).
- **P2**: Stale style/examples (code examples that no longer work; references to removed files).
- **P3**: Optional polish (comment density, wording).

## Phase 3 — Gate new docs via AskUserQuestion

If the audit reveals that **new documentation files** should be created (e.g., a subpackage
missing a `CLAUDE.md`, or a new doc section), present the proposal via `AskUserQuestion`:
- List the proposed new files.
- Explain WHY each is needed (what gap it fills).
- Let the user approve/reject/modify before creating.

Do NOT create new doc files without this gate. Updating existing files does not require the gate.

## Phase 4 — Revise documentation

Apply the consolidated findings. For each file:

### Writing rules (non-negotiable)
- **Clear, compact, jargon-free.** No buzzwords. Write for a developer who is new to the project.
- **Imperative form** in procedural docs (CLAUDE.md, instructions). Third-person in descriptions.
- **Verify every claim** against the code before writing it. If unsure, check with `grep`/`find`/`Read`.
- **No duplication.** Information lives in ONE place. If a detail is in `architecture.md`, don't
  repeat it in `README.md` — link to it.
- **Compact.** README < 200 lines. CLAUDE.md < 300 lines. Folder CLAUDE.md < 100 lines.
- **LLM-friendly.** Use exact file paths, code blocks for structure, tables for quick reference.
  Avoid vague phrases ("various", "etc.", "and more") — be specific.

### What to fix per file type
- **README.md**: Features list, quickstart commands, config examples, examples table, architecture
  subsystem list, install instructions. The front door — must be accurate.
- **CLAUDE.md (root)**: Directory map (counts + new modules), Quick commands (all CLI subcommands),
  Gotchas (new conventions/gotchas from recent merges). The AI's primary reference.
- **`koboi/<subpackage>/CLAUDE.md`**: Module list, key abstractions, conventions, gotchas. Must
  match the actual files in the subpackage.
- **`configs/CLAUDE.md`**: Config list (all `.yaml` files), top-level sections list.
- **`docs/architecture.md`**: Subsystem graph, config-sections table, extension-points table,
  data-types. Must include new subsystems/sections.

### What NOT to touch
- Planning/gap-analysis docs (`docs/rest-sse-requirements.md`, `docs/server-adoption-gap-*.md`,
  `docs/eve-*.md`, `docs/skills-architecture-research.md`) — these are historical specs, not
  codebase docs. Leave them.
- Generated files (`docs/one-pager-api-platform.md` if marked SUPERSEDED).
- `benchmarks/results.json` (183MB — never read).

## Phase 5 — Validate

After revisions:
1. **Verify counts**: re-run `find koboi -name '*.py' | wc -l`, `ls configs/*.yaml | wc -l`,
   `ls examples/*.py | wc -l` — ensure the docs match.
2. **Check links**: any `docs/` or `examples/` references resolve to real files.
3. **Read each revised file** to confirm clarity + no jargon.
4. **Run `ruff` + `mypy`** — doc changes shouldn't affect code, but verify nothing broke.

## Phase 6 — Summary

Present:
- Files modified (list).
- Key changes per file (one-line summary).
- Any new files created (from the Phase 3 gate).
- Remaining gaps (if any were deferred).

## Additional resources

### Reference files
- **`references/audit-checklist.md`** — The exhaustive checklist for auditing each doc type.
- **`references/writing-guide.md`** — Detailed writing rules, examples of good vs bad doc prose.

### Scripts
- **`scripts/audit-counts.sh`** — Quick count of .py files, test files, configs, examples, CLAUDE.md coverage.
