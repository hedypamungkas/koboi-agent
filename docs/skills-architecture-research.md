# Skills Architecture Deep Research: Industry Analysis & koboi-agent Comparison

> Research date: 2026-06-23
> Author: AI Research Assistant
> Scope: Skills implementation patterns across 8+ AI agent frameworks

---

## 1. Executive Summary

Skills in AI agents follow a converging architectural pattern called **progressive disclosure** — a 3-phase lifecycle (Discovery → Activation → Execution) designed to minimize context window overhead. The koboi-agent implementation is architecturally sound and aligned with the emerging [agentskills.io](https://agentskills.io) open standard, but has several areas where empirical improvements can be made based on patterns from Claude Code, Gemini CLI, OpenAI Codex, and other production-grade agents.

**Key Finding**: The user's assumption is **partially correct** — in Claude Code, skills ARE injected into the LLM context, but **not as tools**. They are injected as **prompt extensions** (markdown content in the conversation), with a critical distinction: only skill *descriptions* are always in context (~100 tokens each), while full skill *body* content loads on-demand and persists across turns. This is fundamentally different from tools, which are defined as JSON Schema in every LLM call.

---

## 2. The agentskills.io Standard (Industry Convergence)

### 2.1 What It Is

[Agent Skills](https://agentskills.io) is an **open standard** originally developed by Anthropic, now adopted by 30+ AI coding tools including:

| Agent | Adoption Status |
|-------|----------------|
| Claude Code | Native (extends standard with extras) |
| Gemini CLI | Native |
| OpenAI Codex | Native |
| GitHub Copilot | Native |
| Cursor | Native |
| VS Code | Native |
| Roo Code | Native |
| OpenHands | Native |
| Goose (Block) | Native |
| JetBrains Junie | Native |
| Amp | Native |
| Kiro (AWS) | Native |

### 2.2 Core Architecture

```
skill-name/
├── SKILL.md          # Required: frontmatter (metadata) + instructions
├── scripts/          # Optional: executable code
├── references/       # Optional: documentation
└── assets/           # Optional: templates, resources
```

**SKILL.md Frontmatter** (required fields):
```yaml
---
name: skill-name           # Max 64 chars, lowercase + hyphens
description: What it does   # Max 1024 chars, used for matching
---
```

### 2.3 Progressive Disclosure (The Key Pattern)

This is the **universal architecture** across all implementations:

```
Phase 1: DISCOVERY (always loaded, ~100 tokens/skill)
┌─────────────────────────────────────────┐
│  name + description only                │
│  Injected into system prompt            │
│  Used for routing/matching              │
└─────────────────────────────────────────┘
         │
         ▼ (when task matches)
Phase 2: ACTIVATION (on-demand, ~500-5000 tokens)
┌─────────────────────────────────────────┐
│  Full SKILL.md body loaded              │
│  Injected as context message            │
│  Persists across turns                  │
└─────────────────────────────────────────┘
         │
         ▼ (when referenced in instructions)
Phase 3: EXECUTION (lazy, variable tokens)
┌─────────────────────────────────────────┐
│  scripts/, references/, assets/         │
│  Loaded only when explicitly needed     │
│  Read via file tools or shell exec      │
└─────────────────────────────────────────┘
```

**Why this matters**: A user with 50 skills would need ~5,000 tokens just for descriptions, but only ~100-200 tokens for the 1-2 skills actually relevant to the current task.

---

## 3. How Major Implementations Work (Detailed)

### 3.1 Claude Code (Anthropic) — The Reference Implementation

**Architecture**: Skills are **prompt extensions**, NOT tools.

**How it works**:
1. **Discovery**: At startup, Claude Code scans `~/.claude/skills/`, `.claude/skills/`, and plugin skills. It reads only `name` + `description` from each SKILL.md frontmatter. These are compiled into a **skill listing** that's always in context.

2. **Context injection**: Skill descriptions are injected into the system prompt. The budget is **1% of the model's context window** (configurable via `skillListingBudgetFraction`). If many skills exist, descriptions get shortened or dropped (least-used first).

3. **Activation**: When Claude decides a skill is relevant (or user types `/skill-name`), the full SKILL.md content is injected as a **single message** in the conversation. It stays for the rest of the session.

4. **After compaction**: When context fills up and auto-compaction runs, skills are re-attached after the summary — first 5,000 tokens of each, with a combined budget of 25,000 tokens. Most-recently-invoked skills get priority.

**Key differences from tools**:
| Aspect | Tools | Skills |
|--------|-------|--------|
| Always in context? | Yes (JSON Schema every call) | Only descriptions (~100 tokens each) |
| Invocation | LLM calls tool function | LLM loads skill content into context |
| Execution | Deterministic code | LLM follows instructions |
| Token cost | Fixed per call | Variable (0 if not used) |
| Registration | `@tool()` decorator with schema | SKILL.md file with frontmatter |

**Claude Code extensions beyond the standard**:
- `disable-model-invocation: true` — only user can invoke
- `user-invocable: false` — only Claude can invoke
- `context: fork` — run in isolated subagent
- `allowed-tools` — pre-approve tools for the skill
- `disallowed-tools` — restrict tools while skill active
- `model` / `effort` — override model settings
- Dynamic context injection: `` !`git diff HEAD` `` runs shell before Claude sees content
- String substitution: `$ARGUMENTS`, `$0`, `$1`, `${CLAUDE_SESSION_ID}`

### 3.2 Gemini CLI (Google)

**Architecture**: Similar progressive disclosure, with a **consent step**.

**Unique features**:
- 4-tier precedence: Built-in → Extension → User (`~/.gemini/skills/`) → Workspace (`.gemini/skills/`)
- `.agents/skills/` is an alias for `.gemini/skills/`
- **Consent prompt**: When a skill activates, user sees name/purpose/path and must approve
- Skill directory is added to allowed file paths after activation

**Context injection**: Name + description injected into system prompt at startup. Full SKILL.md content loaded only on `activate_skill` tool call.

### 3.3 OpenAI Codex

**Architecture**: Two-phase loading with **character budget caps**.

**Unique features**:
- Initial list: names + descriptions, **at most 2% of context window or 8,000 chars**
- `allow_implicit_invocation` policy flag per skill
- `agents/openai.yaml` for UI metadata (icons, colors, brand)
- Built-in `$skill-creator` and `$skill-installer`
- Admin-level skills at `/etc/codex/skills`

### 3.4 Roo Code

**Architecture**: Mode-targeted skills with **8-level priority hierarchy**.

**Unique features**:
- Skills scoped to modes via `skills-{mode}/` directories (e.g., `skills-code/`, `skills-architect/`)
- 8-level priority: Project `.roo/` mode-specific → Project `.roo/` generic → Project `.agents/` mode-specific → ... → Global `.agents/` generic
- Uses `read_file` tool to load SKILL.md on activation (not direct injection)

### 3.5 OpenHands

**Architecture**: Two distinct loading models.

**Unique features**:
- **Always-on context**: `AGENTS.md` files injected into system prompt at start
- **On-demand skills**: Loaded via keyword matching or agent decision
- Repository-level, user-level, organization-level, and community/global scopes
- SDK enables dynamic skill creation with full lifecycle control

---

## 4. koboi-agent Implementation Analysis

### 4.1 Current Architecture

```
┌─────────────────────────────────────────────────────┐
│                    SKILL LIFECYCLE                    │
├─────────────────────────────────────────────────────┤
│                                                      │
│  1. DISCOVERY (SkillRegistry.discover_all())         │
│     - Scans: ./skills, .claude/skills, ~/.claude/    │
│     - Parses frontmatter (name + description)        │
│     - Stores as SkillDefinition objects              │
│                                                      │
│  2. ROUTING (SkillRegistry.route())                  │
│     - TF-IDF inspired keyword matching               │
│     - Stopword filtering (EN + ID)                   │
│     - Name match boosted 3x                          │
│     - Returns top-k skills                           │
│                                                      │
│  3. INJECTION (AgentCore._get_managed_messages())    │
│     - Discovery prompt appended to system message    │
│     - Format: <available-skills>...</available-skills>│
│     - Only relevant skills (routed) or all           │
│                                                      │
│  4. ACTIVATION (AgentCore._activate_skill())         │
│     - LLM outputs [ACTIVATE_SKILL: name]             │
│     - Full SKILL.md body loaded                      │
│     - Injected as context message: <skill>...</skill>│
│                                                      │
│  5. EXECUTION                                        │
│     - LLM follows skill instructions                 │
│     - Uses existing tools (read_file, web_search, etc)│
│     - load_resource() for lazy file loading           │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 4.2 What koboi-agent Does Well

1. **Follows agentskills.io standard**: SKILL.md format, frontmatter parsing, progressive disclosure
2. **Smart routing**: TF-IDF based matching with stopword filtering (including Indonesian)
3. **Name boost**: 3x weight for name matches vs description matches
4. **Resource lazy-loading**: `load_resource()` with path traversal protection
5. **Plugin discovery**: Recursive scan of plugin cache directories
6. **No PyYAML dependency**: Custom frontmatter parser

### 4.3 Gaps vs Industry (with empirical justification)

#### Gap 1: No Character Budget for Skill Descriptions

**Problem**: `build_discovery_prompt()` includes ALL discovered skills without budget control.

**Industry standard**:
- Claude Code: 1% of context window (configurable)
- OpenAI Codex: 2% of context window or 8,000 chars max
- Descriptions truncated at 200 chars in koboi (`desc[:200]`) but no total budget

**Impact**: With 50 skills × 200 chars = 10,000 chars of skill descriptions always in context. For a 128K context model, this is ~2,500 tokens wasted when only 1-2 skills are relevant.

**Recommendation**: Implement a `skill_listing_budget_chars` config (default: 1% of max_context_tokens × 4 chars/token). When exceeded, drop least-relevant skills first.

#### Gap 2: No Skill Lifecycle Management After Compaction

**Problem**: After context compaction, activated skills may be lost.

**Industry standard**:
- Claude Code: Re-attaches skills after compaction (first 5,000 tokens each, 25K total budget)
- Skills persist across turns once activated

**Impact**: Long conversations lose skill context after compaction, causing the agent to "forget" skill instructions.

**Recommendation**: Track activated skills and re-inject after compaction, similar to Claude Code's approach.

#### Gap 3: No Invocation Control

**Problem**: All skills are always available for both user and model invocation.

**Industry standard**:
- `disable-model-invocation: true` — user-only skills (deploy, commit)
- `user-invocable: false` — background knowledge skills
- `allowed-tools` — pre-approve tools per skill
- `disallowed-tools` — restrict tools per skill

**Impact**: No way to prevent the model from auto-triggering dangerous skills (e.g., deployment).

**Recommendation**: Add frontmatter fields: `disable-model-invocation`, `user-invocable`, `allowed-tools`, `disallowed-tools`.

#### Gap 4: No Dynamic Context Injection

**Problem**: Skills are static markdown — no way to inject live data.

**Industry standard**:
- Claude Code: `` !`git diff HEAD` `` runs shell before Claude sees content
- ````!` ``` `` blocks for multi-line commands

**Impact**: Skills can't include live context (current git diff, environment state, etc.)

**Recommendation**: Implement `` !`command` `` preprocessing in skill content.

#### Gap 5: No Subagent/Fork Execution

**Problem**: Skills always run in the main conversation context.

**Industry standard**:
- Claude Code: `context: fork` runs skill in isolated subagent
- Skill content becomes the subagent's task prompt

**Impact**: Long-running or isolated skills pollute the main conversation context.

**Recommendation**: Add `context: fork` support with configurable agent type.

#### Gap 6: No Evaluation/Benchmarking Framework

**Problem**: No way to measure skill effectiveness.

**Industry standard**:
- Claude Code: `skill-creator` plugin with evals, benchmarks, A/B testing
- Pass rate, token count, duration metrics
- With-skill vs without-skill comparison

**Impact**: No data-driven way to improve skills.

**Recommendation**: Implement basic skill evaluation: trigger accuracy, output quality, token overhead.

---

## 5. Empirical Data & Benchmarks

### 5.1 Token Overhead Analysis

| Scenario | koboi-agent | Claude Code | Gemini CLI | Codex |
|----------|-------------|-------------|------------|-------|
| 0 skills active | 0 tokens | 0 tokens | 0 tokens | 0 tokens |
| 10 skills discovered | ~2,000 tokens | ~1,000 tokens (budgeted) | ~1,000 tokens | ~2,000 tokens |
| 50 skills discovered | ~10,000 tokens | ~2,000 tokens (budgeted) | ~2,000 tokens | ~4,000 tokens (capped) |
| 1 skill activated | ~500-5,000 tokens | ~500-5,000 tokens | ~500-5,000 tokens | ~500-5,000 tokens |
| After compaction | Lost | Re-attached (5K each) | N/A | N/A |

*Estimates based on average skill description of 200 chars and SKILL.md body of 2,000 chars*

### 5.2 Routing Accuracy

koboi-agent's TF-IDF routing vs alternatives:

| Method | Precision@3 | Recall@3 | Notes |
|--------|-------------|----------|-------|
| koboi TF-IDF | ~70-80% | ~60-70% | Keyword-based, no semantic understanding |
| Claude Code LLM | ~90-95% | ~85-90% | Uses LLM to match (expensive) |
| Gemini CLI keyword | ~75-85% | ~65-75% | Similar to koboi approach |
| Codex LLM routing | ~90-95% | ~85-90% | LLM-based matching |

*Note: These are estimated ranges based on architectural analysis, not published benchmarks. No framework has published formal routing accuracy metrics.*

### 5.3 Context Window Impact

For a 128K context window model (Claude Sonnet class):

```
System prompt:           ~2,000 tokens
CLAUDE.md:               ~500 tokens
Skill descriptions (10): ~500 tokens  ← koboi: ~2,000 tokens (no budget)
Tool definitions (9):    ~1,800 tokens
Conversation history:    variable
─────────────────────────────────────
Available for reasoning: ~123,000 tokens (with budget)
                         ~121,500 tokens (koboi, no budget)
```

The difference is ~1,500 tokens for 10 skills — negligible at small scale but becomes significant with 50+ skills or smaller context windows.

---

## 6. How Skills Differ from Tools (The Truth)

### 6.1 The User's Assumption

> "skills ini dimasukkan ke setiap context pada LLM call sehingga treatment nya spt tool"

**Verdict**: Partially correct. Here's the precise breakdown:

### 6.2 Tools (Always in Context)

```json
// Every LLM call includes this JSON Schema
{
  "type": "function",
  "function": {
    "name": "read_file",
    "description": "Read a file from the filesystem",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {"type": "string"}
      },
      "required": ["path"]
    }
  }
}
```

**Cost**: ~200-400 tokens per tool definition × 9 tools = ~1,800-3,600 tokens EVERY call.

### 6.3 Skills (Progressive Disclosure)

```
Always in context (discovery phase):
  <available-skills>
  - code-review: Systematic code review focusing on security...
  - search-and-summarize: Research a topic by searching...
  </available-skills>
```

**Cost**: ~100-200 tokens per skill description, only when skills are discovered.

**On activation** (one-time injection):
```
<skill name="code-review" dir="/path/to/skills/code_review">
# Code Review Skill
## Instructions
When this skill is activated, review code systematically...
</skill>
```

**Cost**: ~500-5,000 tokens, injected once, persists across turns.

### 6.4 The Key Difference

| Aspect | Tool | Skill |
|--------|------|-------|
| **When in context** | Every LLM call | Description always; body on-demand |
| **What LLM does** | Calls function with JSON args | Reads instructions, uses existing tools |
| **Token cost** | Fixed per call (schema) | Variable (0 if not used) |
| **Execution** | Deterministic code | LLM reasoning + tool use |
| **Composability** | Atomic operations | Multi-step workflows |
| **Abstraction level** | "What can I do" | "How should I do it" |

**Skills are NOT tools.** Skills are **procedural knowledge** that tell the LLM how to use existing tools for specific domains. A "code review" skill doesn't add a new tool — it instructs the LLM to use `read_file`, `grep`, and `web_search` in a specific pattern.

---

## 7. Recommendations for koboi-agent

### 7.1 Priority 1: Add Character Budget for Discovery Prompt

```python
def build_discovery_prompt(skills: list[SkillDefinition], budget_chars: int = 8000) -> str:
    """Generate discovery metadata text with character budget."""
    if not skills:
        return ""

    lines = [
        "",
        "<available-skills>",
        "You have skills that can be activated. "
        "When the user request matches, respond with format: [ACTIVATE_SKILL: skill-name]",
        "",
    ]
    
    total_chars = sum(len(lines) for lines in lines)
    for skill in skills:
        desc = skill.description[:200]
        entry = f"- {skill.name}: {desc}"
        if total_chars + len(entry) > budget_chars:
            lines.append(f"  ... and {len(skills) - len(lines) + 3} more skills (truncated)")
            break
        lines.append(entry)
        total_chars += len(entry)

    lines.append("</available-skills>")
    return "\n".join(lines)
```

### 7.2 Priority 2: Add Invocation Control Frontmatter

```python
@dataclass
class SkillDefinition:
    name: str
    description: str
    skill_dir: str
    body: str | None = None
    # New fields:
    disable_model_invocation: bool = False
    user_invocable: bool = True
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = None
```

### 7.3 Priority 3: Skill Persistence After Compaction

```python
class AgentCore:
    def _reinject_skills_after_compaction(self):
        """Re-inject activated skills after context compaction."""
        for skill_name in self.skills._activated:
            skill = self.skills.get(skill_name)
            if skill and skill.body:
                self.memory.add_context_message(
                    f'<skill name="{skill_name}">\n{skill.body[:5000]}\n</skill>',
                    label=skill_name,
                )
```

### 7.4 Priority 4: Dynamic Context Injection

```python
def activate_skill(skill: SkillDefinition) -> str:
    """Load SKILL.md body with shell preprocessing."""
    skill_path = Path(skill.skill_dir) / "SKILL.md"
    content = skill_path.read_text(encoding="utf-8")
    body = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, flags=re.DOTALL).strip()
    
    # Process !`command` placeholders
    def run_shell(match):
        import subprocess
        cmd = match.group(1)
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            return result.stdout.strip()
        except Exception:
            return f"[command failed: {cmd}]"
    
    body = re.sub(r"!`([^`]+)`", run_shell, body)
    skill.body = body
    return body
```

### 7.5 Priority 5: Basic Skill Evaluation

```python
class SkillEvaluator:
    """Evaluate skill effectiveness with test cases."""
    
    def evaluate(self, skill: SkillDefinition, test_cases: list[dict]) -> dict:
        results = {
            "trigger_accuracy": 0.0,  # Did it activate on matching queries?
            "output_quality": 0.0,     # Did output match expectations?
            "token_overhead": 0,       # How many tokens did the skill add?
            "latency_ms": 0,           # How much slower with skill?
        }
        # Implementation...
        return results
```

---

## 8. Conclusion

The koboi-agent skill system is architecturally sound and follows the emerging industry standard. The core progressive disclosure pattern is correctly implemented. The main gaps are:

1. **No budget control** for skill descriptions in context
2. **No persistence** after context compaction
3. **No invocation control** (all skills always available)
4. **No dynamic context** injection
5. **No evaluation framework**

These are incremental improvements, not fundamental redesigns. The existing TF-IDF routing with Indonesian stopword support is a unique advantage for the Indonesian market that other frameworks lack.

**The user's assumption about skills being treated like tools is incorrect** — skills are prompt extensions that load on-demand, not tool definitions that are always in context. This is the key architectural insight that makes skills scalable.

---

## Sources

- [Agent Skills Specification](https://agentskills.io/specification)
- [Claude Code Skills Documentation](https://code.claude.com/docs/en/skills)
- [Gemini CLI Skills](https://geminicli.com/docs/cli/skills/)
- [OpenHands Skills](https://docs.openhands.dev/overview/skills)
- [Roo Code Skills](https://roocodeinc.github.io/Roo-Code/features/skills)
- [OpenAI Codex Skills](https://developers.openai.com/codex/skills/)
- [Agent Skills GitHub](https://github.com/agentskills/agentskills)
