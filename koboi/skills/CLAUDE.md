# koboi/skills/ -- Skill discovery and activation (agentskills.io standard)

## What this is
Progressive-disclosure skills: metadata (name + description) is always loaded into
the system prompt; the SKILL.md **body** is loaded on-demand when the LLM emits
`[ACTIVATE_SKILL: name]`; resource files load lazily on first access. Driven by the
`skills:` YAML section (search paths). No ABC here -- `SkillRegistry` is the whole
surface. Implements the agentskills.io Level 8 universal standard.

## Key files
```
registry.py    SkillRegistry + discover_skills + activate_skill + load_resource
               + build_discovery_prompt + parse_frontmatter (custom, no PyYAML)
__init__.py    Re-exports SkillRegistry, discover_skills, activate_skill, load_resource
```

## SkillDefinition (`koboi/types.py`)
`name`, `description`, `skill_dir`, `body` (None until activated), `license`,
`compatibility`, `metadata`, `allowed_tools`, `disable_model_invocation` (False),
`user_invocable` (True), `disallowed_tools`, `allow_shell` (False). Frontmatter keys
are kebab-case (`allowed-tools`, `disable-model-invocation`, `user-invocable`,
`allow-shell`) and mapped to the snake_case dataclass fields by `parse_frontmatter`.

## Discovery and activation
- `SkillRegistry(logger=None, budget_chars=8000)` scans standard locations:
  - `PROJECT_SKILLS` = `["./skills", ".claude/skills"]` (flat scan)
  - `USER_SKILLS` = `["~/.claude/skills"]` (flat scan)
  - `PLUGIN_SKILLS` = `["~/.claude/plugins/cache"]` (recursive -- nested plugin cache)
  - `discover_all()` runs all three in that order; **first occurrence wins** (dedup by name).
- A `SKILL.md` without both `name` and `description` is silently skipped.
- `activate(name, run_shell=False)` strips frontmatter and stores the body; `body`
  is loaded only once (lazy). ``!`cmd` `` preprocessing requires BOTH `run_shell=True`
  AND `skill.allow_shell=True` (issue #46 fail-closed gate; default False everywhere).
  `route(query, top_k=3)` does TF-IDF keyword matching (name match weighted 3x
  description; includes Indonesian stopwords).
- `get_discovery_prompt()` / `get_routed_discovery_prompt(query)` build the
  `<available-skills>` block (truncated to `budget_chars`, dropping least-relevant).

## Conventions
- SKILL.md lives in its own directory; `skill_dir` resolves to that dir's absolute path.
- `disable_model_invocation=True` skills render with a `[user-only]` suffix and are
  **excluded from model auto-routing** (`route()`) unless `include_model_disabled=True`.
- `user_invocable` defaults True; set False to hide from user-triggered paths.
- `build_discovery_prompt()` accepts `budget_chars`; `SkillRegistry` defaults to 8000.

## Gotchas
- **Shell injection on activation (issue #46, fail-closed)**: ``!`command` `` blocks
  are NOT executed unless BOTH the caller passes `run_shell=True` AND the skill carries
  `allow_shell=True` (set via `allow-shell: true` frontmatter). Both default False, so an
  untrusted SKILL.md cannot run shell on the activation path. When execution IS enabled,
  deny-listed commands and sensitive paths are still refused (reuses
  `_check_command_blocked` + `build_safe_env`) as defense-in-depth.
- **`parse_frontmatter` is a custom mini-parser** (no PyYAML): it handles block scalars
  (`>`, `|`, `|-`...) and kebab-case fields, but exotic YAML may not parse -- keep
  frontmatter simple.
- **`SkillPersistenceHook` (priority 45) re-injects** activated skills after
  `POST_COMPACT`, so activation survives context compression (lives in `koboi/hooks/`).
- **Discovery is metadata-only**: `body=None` until `activate()`. A skill with a great
  description but empty/weak body will route well but deliver little.
