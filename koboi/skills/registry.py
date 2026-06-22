"""
Skills system -- agentskills.io universal standard (Level 8).

Progressive disclosure:
  1. Discovery: metadata (name+description) always loaded
  2. Activation: SKILL.md body loaded on-demand
  3. Resources: files in scripts/, references/, assets/ lazy-loaded
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from koboi.types import SkillDefinition

if TYPE_CHECKING:
    from koboi.logger import AgentLogger


def parse_frontmatter(content: str) -> dict:
    """Parse YAML frontmatter from SKILL.md (no PyYAML needed)."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        return {}

    raw = match.group(1)
    result = {}
    current_key = None
    current_key_continuation = None
    metadata_lines = []

    for line in raw.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Handle YAML block scalar continuation lines (>, |, |-, >-, |+, >+)
        if current_key_continuation and line.startswith("  "):
            existing = result.get(current_key_continuation, "")
            result[current_key_continuation] = (existing + " " + stripped).strip()
            continue
        elif current_key_continuation:
            current_key_continuation = None

        if current_key == "metadata" and line.startswith("  "):
            if ":" in stripped:
                mk, mv = stripped.split(":", 1)
                mv = mv.strip()
                if mv.startswith('"') and mv.endswith('"'):
                    mv = mv[1:-1]
                elif mv.startswith("'") and mv.endswith("'"):
                    mv = mv[1:-1]
                metadata_lines.append((mk.strip(), mv))
            continue

        if ":" in stripped:
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()

            if key == "metadata":
                current_key = "metadata"
                metadata_lines = []
                continue

            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            elif value in (">", "|", "|-", ">-", "|+", ">+"):
                current_key_continuation = key
                result[key] = ""
                continue

            result[key] = value

    if metadata_lines:
        result["metadata"] = {k: v for k, v in metadata_lines}

    if "allowed-tools" in result:
        result["allowed-tools"] = result["allowed-tools"].split()

    if "disallowed-tools" in result:
        result["disallowed-tools"] = result["disallowed-tools"].split()

    # Boolean fields: parse string "true"/"false" to bool
    for bool_key in ("disable-model-invocation", "user-invocable"):
        if bool_key in result:
            val = result[bool_key]
            if isinstance(val, str):
                result[bool_key] = val.lower() in ("true", "1", "yes")

    return result


def discover_skills(search_paths: list[str | Path], recursive: bool = False) -> list[SkillDefinition]:
    """Scan directories for SKILL.md files. Returns skills with metadata only (body=None).

    Args:
        search_paths: directories to scan (flat scan per path).
        recursive: if True, recursively walk subdirs to find SKILL.md files
                   (needed for plugin cache dirs with nested structure).
    """
    skills = []
    seen_names: set[str] = set()

    for search_path in search_paths:
        search_dir = Path(search_path).expanduser()
        if not search_dir.is_dir():
            continue

        if recursive:
            # Walk deep -- find ALL SKILL.md files, skip node_modules
            for skill_file in sorted(search_dir.rglob("SKILL.md")):
                if "node_modules" in skill_file.parts:
                    continue
                _try_register_skill(skill_file, skills, seen_names)
        else:
            # Flat scan -- direct children only (project skills pattern)
            for entry in sorted(search_dir.iterdir()):
                if not entry.is_dir():
                    continue
                skill_file = entry / "SKILL.md"
                if skill_file.is_file():
                    _try_register_skill(skill_file, skills, seen_names)

    return skills


def _try_register_skill(
    skill_file: Path,
    skills: list[SkillDefinition],
    seen_names: set[str],
) -> None:
    """Parse a SKILL.md and append to skills list if valid. Dedup by name."""
    try:
        content = skill_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return

    frontmatter = parse_frontmatter(content)

    name = frontmatter.get("name", "")
    description = frontmatter.get("description", "")

    if not name or not description:
        return

    # Dedup: first occurrence wins
    if name in seen_names:
        return
    seen_names.add(name)

    skills.append(
        SkillDefinition(
            name=name,
            description=description,
            skill_dir=str(skill_file.parent.resolve()),
            license=frontmatter.get("license"),
            compatibility=frontmatter.get("compatibility"),
            metadata=frontmatter.get("metadata"),
            allowed_tools=frontmatter.get("allowed-tools"),
            disable_model_invocation=frontmatter.get("disable-model-invocation", False),
            user_invocable=frontmatter.get("user-invocable", True),
            disallowed_tools=frontmatter.get("disallowed-tools"),
        )
    )


def activate_skill(skill: SkillDefinition, run_shell: bool = True) -> str:
    """Load SKILL.md body (strip frontmatter). Sets skill.body and returns it.

    Args:
        skill: The skill definition to activate.
        run_shell: If True, preprocess `` !`command` `` blocks by executing
            them and injecting stdout. Commands time out after 10 seconds.
    """
    skill_path = Path(skill.skill_dir) / "SKILL.md"
    content = skill_path.read_text(encoding="utf-8")
    body = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, flags=re.DOTALL).strip()

    if run_shell:
        body = _preprocess_shell_commands(body)

    skill.body = body
    return body


def _preprocess_shell_commands(body: str) -> str:
    """Replace `` !`command` `` blocks with their stdout output.

    Commands are executed with a 10-second timeout. On failure, the block
    is replaced with ``[command failed: <cmd>]``.
    """
    import subprocess

    def _run(match: re.Match) -> str:
        cmd = match.group(1).strip()
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = result.stdout.strip()
            if result.returncode != 0 and not output:
                return f"[command failed: {cmd}]"
            return output if output else f"[command produced no output: {cmd}]"
        except subprocess.TimeoutExpired:
            return f"[command timed out: {cmd}]"
        except Exception:
            return f"[command failed: {cmd}]"

    # Match `` !`command` `` (with optional multiline backtick fence)
    return re.sub(r"!`([^`]+)`", _run, body)


def load_resource(skill: SkillDefinition, relative_path: str) -> str | None:
    """Lazy-load a resource file from skill directory. Returns None if not found."""
    resource_path = Path(skill.skill_dir) / relative_path
    if not resource_path.is_file():
        return None
    if not str(resource_path.resolve()).startswith(str(Path(skill.skill_dir).resolve())):
        return None
    return resource_path.read_text(encoding="utf-8")


def build_discovery_prompt(skills: list[SkillDefinition], budget_chars: int | None = None) -> str:
    """Generate discovery metadata text for system prompt injection.

    Args:
        skills: List of skills to include (should be pre-sorted by relevance).
        budget_chars: Maximum characters for the prompt. If exceeded,
            least-relevant skills (those at the end of the list) are dropped.
            None means no limit.
    """
    if not skills:
        return ""

    header_lines = [
        "",
        "<available-skills>",
        "You have skills that can be activated. "
        "When the user request matches, respond with format: [ACTIVATE_SKILL: skill-name]",
        "",
    ]
    footer = "</available-skills>"

    if budget_chars is None:
        # No budget: include all skills
        lines = list(header_lines)
        for skill in skills:
            desc = skill.description[:200]
            suffix = ""
            if skill.disable_model_invocation:
                suffix = " [user-only]"
            lines.append(f"- {skill.name}: {desc}{suffix}")
        lines.append(footer)
        return "\n".join(lines)

    # Budget-aware: track chars and truncate when exceeded
    # Reserve space for header + footer
    header_text = "\n".join(header_lines) + "\n"
    footer_text = "\n" + footer
    reserved = len(header_text) + len(footer_text) + 80  # 80 chars for truncation message
    remaining = budget_chars - reserved

    if remaining <= 0:
        return ""

    lines = list(header_lines)
    total_chars = 0
    included = 0

    for skill in skills:
        desc = skill.description[:200]
        suffix = ""
        if skill.disable_model_invocation:
            suffix = " [user-only]"
        entry = f"- {skill.name}: {desc}{suffix}"
        entry_chars = len(entry) + 1  # +1 for newline

        if total_chars + entry_chars > remaining:
            dropped = len(skills) - included
            if dropped > 0:
                lines.append(f"  ... and {dropped} more skills (budget limit)")
            break

        lines.append(entry)
        total_chars += entry_chars
        included += 1

    lines.append(footer)
    return "\n".join(lines)


class SkillRegistry:
    """Manage discovered skills. Pattern mirrors ToolRegistry."""

    # Default search paths (agentskills.io standard locations)
    PROJECT_SKILLS = ["./skills", ".claude/skills"]
    USER_SKILLS = ["~/.claude/skills"]
    PLUGIN_SKILLS = ["~/.claude/plugins/cache"]

    def __init__(self, logger: AgentLogger | None = None, budget_chars: int | None = 8000):
        self._skills: dict[str, SkillDefinition] = {}
        self._activated: set[str] = set()
        self.logger = logger
        self.budget_chars = budget_chars

    def discover(self, search_paths: list[str | Path], recursive: bool = False) -> list[str]:
        """Scan paths for skills and register them. Returns list of names."""
        skills = discover_skills(search_paths, recursive=recursive)
        for skill in skills:
            self._skills[skill.name] = skill
        names = [s.name for s in skills]
        if self.logger:
            self.logger.log(f"Skills discovered: {names}")
        return names

    def discover_all(self) -> list[str]:
        """Discover skills from all standard locations: project, user, plugins.

        Returns list of all discovered skill names.
        """
        all_names = []
        # 1. Project-level skills (flat scan)
        all_names += self.discover(self.PROJECT_SKILLS)
        # 2. User-level skills (flat scan)
        all_names += self.discover(self.USER_SKILLS)
        # 3. Plugin skills (recursive -- nested structure)
        all_names += self.discover(self.PLUGIN_SKILLS, recursive=True)
        return all_names

    def register(self, skill: SkillDefinition) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> SkillDefinition | None:
        return self._skills.get(name)

    def list_skills(self) -> list[SkillDefinition]:
        return list(self._skills.values())

    def activate(self, name: str) -> str | None:
        """Activate a skill by name. Loads body if not yet loaded."""
        skill = self._skills.get(name)
        if not skill:
            return None
        if skill.body is None:
            activate_skill(skill)
        self._activated.add(name)
        if self.logger:
            self.logger.log(f"Skill activated: {name} ({len(skill.body or '')} chars)")
        return skill.body

    def is_activated(self, name: str) -> bool:
        return name in self._activated

    def load_resource(self, skill_name: str, relative_path: str) -> str | None:
        skill = self._skills.get(skill_name)
        if not skill:
            return None
        return load_resource(skill, relative_path)

    def get_discovery_prompt(self) -> str:
        return build_discovery_prompt(self.list_skills(), budget_chars=self.budget_chars)

    # Common words that appear in almost every skill description -- useless for matching
    _STOPWORDS = frozenset(
        {
            "a",
            "an",
            "the",
            "and",
            "or",
            "for",
            "to",
            "of",
            "in",
            "on",
            "at",
            "is",
            "are",
            "was",
            "it",
            "this",
            "that",
            "with",
            "from",
            "by",
            "as",
            "use",
            "using",
            "used",
            "when",
            "should",
            "can",
            "needs",
            "work",
            "skill",
            "skills",
            "create",
            "also",
            "all",
            "any",
            "has",
            "have",
            "nya",
            "di",
            "dan",
            "ke",
            "dari",
            "yang",
            "ini",
            "itu",
            "dengan",
            "untuk",
            "pada",
            "adalah",
            "akan",
            "bisa",
            "atau",
            "juga",
            "saat",  # Indonesian stopwords kept for multilingual support
            "load",
            "loaded",
            "help",
            "like",
            "want",
            "need",
            "make",
            "just",
        }
    )

    def route(self, query: str, top_k: int = 3, include_model_disabled: bool = False) -> list[SkillDefinition]:
        """Return top-k skills whose description matches query keywords.

        Scoring: TF-IDF inspired -- rare words in query score higher
        than common words that appear in many descriptions.

        Args:
            query: User query to match against.
            top_k: Maximum number of skills to return.
            include_model_disabled: If True, include skills with
                disable_model_invocation=True (for user-triggered routing).
                If False (default), exclude them from auto-routing.
        """
        q_words = set(re.findall(r"\w+", query.lower())) - self._STOPWORDS
        if not q_words:
            return []

        # IDF: count how many skills each query word appears in
        word_doc_count: dict[str, int] = {}
        for skill in self._skills.values():
            # Skip model-disabled skills for IDF calculation unless explicitly included
            if skill.disable_model_invocation and not include_model_disabled:
                continue
            name_words = set(re.findall(r"\w+", skill.name.replace("-", " ").replace("_", " ").lower()))
            desc_words = set(re.findall(r"\w+", skill.description.lower())) - self._STOPWORDS
            s_words = name_words | desc_words
            for w in q_words:
                if w in s_words:
                    word_doc_count[w] = word_doc_count.get(w, 0) + 1

        scored = []
        for skill in self._skills.values():
            # Skip model-disabled skills unless explicitly included
            if skill.disable_model_invocation and not include_model_disabled:
                continue
            name_words = set(re.findall(r"\w+", skill.name.replace("-", " ").replace("_", " ").lower()))
            desc_words = set(re.findall(r"\w+", skill.description.lower())) - self._STOPWORDS
            s_words = name_words | desc_words
            overlap = q_words & s_words
            if not overlap:
                continue
            # IDF score: rarer words contribute more
            score = sum(1.0 / (1 + word_doc_count.get(w, 1)) for w in overlap)
            # Boost: name match counts 3x more than description match
            name_overlap = overlap & name_words
            if name_overlap:
                score += sum(3.0 / (1 + word_doc_count.get(w, 1)) for w in name_overlap)
            scored.append((score, skill))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [skill for _, skill in scored[:top_k]]

    def get_routed_discovery_prompt(self, query: str) -> str:
        """Discovery prompt with only skills relevant to query."""
        matched = self.route(query)
        return build_discovery_prompt(matched, budget_chars=self.budget_chars)
