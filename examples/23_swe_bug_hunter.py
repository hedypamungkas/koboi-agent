"""Example 23: SWE-Agent Bug Hunter -- complex multi-feature example.

Clone of SWE-Agent/OpenHands: autonomous code analysis agent that finds bugs,
classifies severity, and proposes fixes. Demonstrates ALL koboi-agent features
working together in a realistic production scenario.

Features exercised:
- RAG: buggy code files as knowledge base for analysis
- Tools: shell, search, filesystem, memory, calculator
- Skills: progressive disclosure (code-review, incident-response, bug-hunter)
- Guardrails: input/output validation, rate limiting
- Policy engine: deny destructive commands (rm -rf, curl, chmod)
- Custom Hooks: BugTrackerHook, SeverityAssessmentHook, CodeAnalysisHook
- Harness: telemetry, carryover state, doom loop detection
- Context management: smart_truncation with carryover for long sessions
- Orchestration: multi-agent audit mode (static_analyzer, test_generator, fix_proposer)
- Eval suite: 6 cases with custom scorers (BugDetection, Severity, CodeReference)

Three run modes:
  --run-mode bug_hunt  Single agent with all features (default)
  --run-mode audit     Multi-agent orchestrated audit
  --run-mode eval      Full evaluation suite

Run:
    python examples/23_swe_bug_hunter.py                                # bug_hunt automatic
    python examples/23_swe_bug_hunter.py -m interactive                 # bug_hunt interactive
    python examples/23_swe_bug_hunter.py --run-mode audit               # orchestrated audit
    python examples/23_swe_bug_hunter.py --run-mode eval                # eval suite
    python examples/23_swe_bug_hunter.py -m interactive --run-mode bug_hunt -v
"""

from __future__ import annotations

import asyncio
import re
import time

import click
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from conftest import (
    console,
    setup_example,
    dual_mode_options,
    create_agent,
    automatic_batch,
    interactive_loop,
    run_async,
)

ensure_path = __import__("conftest").ensure_path
ensure_path()

from koboi.hooks.chain import Hook, HookContext, HookEvent  # noqa: E402
from koboi.eval.scorers import BaseScorer  # noqa: E402


# ---------------------------------------------------------------------------
# Custom Hooks
# ---------------------------------------------------------------------------


class BugTrackerHook(Hook):
    """Tracks bugs found during analysis sessions.

    Subscribes to POST_TOOL_USE and POST_LLM_CALL to extract bug descriptions
    from tool output and LLM responses. Maintains a structured registry of
    findings with file, line, description, and category.
    """

    _SEV_WORD = r"(?:CRITICAL|HIGH|MEDIUM|LOW|INFO|WARNING)"
    _SEV_LEVEL = r"P[0-3]"
    _EMOJI = r"(?:🔴|🟠|🟡|🟢|⚠️|🐛)"
    _SEP = r"[ \t]*[—–:-][ \t]*"

    BUG_PATTERNS = [
        # Emoji + severity word: 🔴 CRITICAL — Title
        re.compile(rf"{_EMOJI}[ \t]+{_SEV_WORD}{_SEP}(.+?)(?:\n|$)", re.IGNORECASE),
        # Emoji + severity level: 🔴 P0 — Title  (most common LLM format)
        re.compile(rf"{_EMOJI}[ \t]+{_SEV_LEVEL}{_SEP}(.+?)(?:\n|$)", re.IGNORECASE),
        # Emoji + BUG N: 🔴 BUG 1 — Title
        re.compile(rf"{_EMOJI}[ \t]+BUG\s*#?\d+{_SEP}(.+?)(?:\n|$)", re.IGNORECASE),
        # Emoji alone + title (no severity word/level): 🔴 — Title
        re.compile(rf"{_EMOJI}{_SEP}(.+?)(?:\n|$)", re.IGNORECASE),
        # Numbered list: 1. Title (Lines ...) or N. Title (Lines ...)
        re.compile(r"^\d+\.[ \t]+(.+?)(?:\s*\(Lines?\s|(?:\n|$))", re.MULTILINE),
        # Finding N: Title
        re.compile(r"Finding\s*#?\d+[:\.]\s*(.+?)(?:\n|$)", re.IGNORECASE),
        # Interaction N: Title
        re.compile(r"Interaction\s*#?\d+[:\.]\s*(.+?)(?:\n|$)", re.IGNORECASE),
        # Severity word + Finding: CRITICAL Finding 1: Title
        re.compile(rf"{_SEV_WORD}[ \t]+Finding\s*\d+[:\.]?\s*(.+?)(?:\n|$)", re.IGNORECASE),
        # Severity level + Finding: P0 Finding 1: Title
        re.compile(rf"{_SEV_LEVEL}[ \t]+Finding\s*\d+[:\.]?\s*(.+?)(?:\n|$)", re.IGNORECASE),
        # Bug N: Title
        re.compile(r"Bug\s*#?\d+[:\.]?\s*[—–:-]?\s*(.+?)(?:\n|$)", re.IGNORECASE),
        # Issue/Vuln N: Title
        re.compile(r"(?:Issue|Vuln(?:erability)?)\s*#?\d+[:\.]?\s*(.+?)(?:\n|$)", re.IGNORECASE),
        # Fallback: ### Bug N: Title
        re.compile(r"###\s*(?:Bug|Issue|Vuln)\s*#?\d*[:\s-]+(.+?)(?:\n###|\n\n|\Z)", re.IGNORECASE | re.DOTALL),
        # Fallback: **Bug N** - Title
        re.compile(r"\*\*(?:Bug|Issue|Vuln)\s*#?\d*\*\*[:\s-]+(.+?)(?:\n\n|\n\*\*|\Z)", re.IGNORECASE | re.DOTALL),
    ]
    # Descriptions matching these patterns are NOT bug titles
    FALSE_POSITIVE_PATTERNS = [
        re.compile(r"^\(?\w+\s+(swallow|drops?|crashes?|hides?|bypass|leak|exceed)", re.IGNORECASE),
        re.compile(r"^\(?(?:off-by-one|silent|division|race|float)", re.IGNORECASE),
        re.compile(r"^\(?(?:no\s|missing\s|unvalidated)", re.IGNORECASE),
        re.compile(r"→\s*(?:attacker|user|malicious|adversary|exploit|Bug|P[0-3])", re.IGNORECASE),
        re.compile(
            r"(?:attacker|adversary)\s+(?:can|gains|manipulates|injects|bypasses|executes|overwrites)", re.IGNORECASE
        ),
        re.compile(
            r"^\*\*(?:Wrap|Fix|Remove|Use|Evict|Apply|Implement|Clamp|Replace|Update|Introduce|Refactor|Ensure|Avoid|Set|Return|Check|Validate)\b",
            re.IGNORECASE,
        ),
        re.compile(r"^\*\*(?:I |Common |Categories |My |Our |The analysis|This )", re.IGNORECASE),
    ]
    # Section headers that indicate FIX recommendations, not bugs
    _FIX_SECTION_PATTERNS = [
        re.compile(r"Recommended\s+Fixes?\b", re.IGNORECASE),
        re.compile(r"Remediation", re.IGNORECASE),
        re.compile(r"Suggested\s+Fixes?\b", re.IGNORECASE),
        re.compile(r"Fix\s+Proposals?\b", re.IGNORECASE),
        re.compile(r"How\s+to\s+Fix", re.IGNORECASE),
        re.compile(r"\bFix\s+Order\b", re.IGNORECASE),
        re.compile(r"Priority\s+Recommendation", re.IGNORECASE),
        re.compile(r"Top\s+Recommendations?\b", re.IGNORECASE),
    ]
    # Structural break that ends a fix section
    _SECTION_BREAK = re.compile(r"\n(?:─{3,}|-{3,}|#{1,4}\s)", re.MULTILINE)
    FILE_PATTERN = re.compile(r"(\w+\.py)(?::(\d+))?")
    LINE_PATTERN = re.compile(r"(?:Line|line)\s+#?(\d+)")
    TABLE_ROW_PATTERN = re.compile(r"^\s*(\d+(?:[–-]\d+)?)\s{2,}(.+?)\s{2,}P[0-3]", re.MULTILINE)
    SEVERITY_KEYWORDS = {"critical": "P0", "high": "P1", "medium": "P2", "low": "P3"}
    EMOJI_SEVERITY = {"🔴": "P0", "🟠": "P1", "🟡": "P2", "🟢": "P3", "⚠️": "P1", "🐛": "P2"}

    def __init__(self):
        self._bugs: list[dict] = []
        self._seen_keys: set[str] = set()

    def handles(self) -> list[HookEvent]:
        return [HookEvent.POST_TOOL_USE, HookEvent.POST_LLM_CALL, HookEvent.SESSION_END]

    async def execute(self, ctx: HookContext) -> HookContext:
        if ctx.event == HookEvent.POST_TOOL_USE and ctx.tool_result:
            self._extract_bugs(ctx.tool_result, ctx.tool_name)
        elif ctx.event == HookEvent.POST_LLM_CALL and ctx.llm_response:
            content = getattr(ctx.llm_response, "content", "") or ""
            self._extract_bugs(content, "llm")
        elif ctx.event == HookEvent.SESSION_END:
            ctx.metadata["bug_tracker_summary"] = {
                "total_bugs": len(self._bugs),
                "bugs": self._bugs,
                "files_affected": list({b["file"] for b in self._bugs if b.get("file")}),
            }
        return ctx

    _SEV_WORD_TO_LEVEL = {
        "critical": "P0",
        "high": "P1",
        "medium": "P2",
        "low": "P3",
    }
    _EMOJI_TO_LEVEL = {
        "🔴": "P0",
        "🟠": "P1",
        "🟡": "P2",
        "🟢": "P3",
        "⚠️": "P1",
        "🐛": "P2",
        "🔵": "P3",
    }

    def _build_severity_map(self, text: str) -> list[tuple[int, str]]:
        """Pre-scan text for severity section headers and return sorted (position, level) pairs."""
        sections: list[tuple[int, str]] = []
        for m in re.finditer(
            r"Severity:\s*(?:[🔴🟠🟡🟢⚠️🐛🔵])?\s*(CRITICAL|HIGH|MEDIUM|LOW)",
            text,
            re.IGNORECASE,
        ):
            sections.append((m.start(), self._SEV_WORD_TO_LEVEL.get(m.group(1).lower(), "unclassified")))
        for m in re.finditer(
            r"([🔴🟠🟡🟢⚠️🐛🔵])\s*(Critical|High|Medium|Low|Info|Warning)\s+Bugs?",
            text,
            re.IGNORECASE | re.MULTILINE,
        ):
            sections.append((m.start(), self._EMOJI_TO_LEVEL.get(m.group(1), "unclassified")))
        sections.sort(key=lambda x: x[0])
        return sections

    @staticmethod
    def _lookup_severity(sections: list[tuple[int, str]], position: int) -> str:
        """Find which severity section contains `position`."""
        result = "unclassified"
        for sec_pos, sec_sev in sections:
            if sec_pos <= position:
                result = sec_sev
            else:
                break
        return result

    def _build_table_severity_map(self, text: str) -> dict[int, str]:
        """Extract bug number → severity from summary tables with emoji+word format."""
        mapping: dict[int, str] = {}
        for m in re.finditer(
            r"^\s*(\d+)\s+([🔴🟠🟡🟢⚠️🐛🔵])\s*(Critical|High|Medium|Low|Info|Warning)",
            text,
            re.IGNORECASE | re.MULTILINE,
        ):
            bug_num = int(m.group(1))
            emoji = m.group(2)
            mapping[bug_num] = self._EMOJI_TO_LEVEL.get(emoji, "unclassified")
        return mapping

    def _extract_bugs(self, text: str, source: str) -> None:
        severity_map = self._build_severity_map(text)
        table_sev_map = self._build_table_severity_map(text)

        # Build positions of fix/recommendation sections to exclude
        fix_positions: list[tuple[int, int]] = []
        for fp in self._FIX_SECTION_PATTERNS:
            for m in fp.finditer(text):
                end_pos = len(text)
                end_match = self._SECTION_BREAK.search(text, m.end())
                if end_match:
                    end_pos = end_match.start()
                fix_positions.append((m.start(), end_pos))

        def _in_fix_section(pos: int) -> bool:
            return any(start <= pos <= end for start, end in fix_positions)

        # Phase 1: table row format (LineNum   Description   P0-P3)
        for match in self.TABLE_ROW_PATTERN.finditer(text):
            if _in_fix_section(match.start()):
                continue
            line_str = match.group(1).split("–")[0].split("-")[0]
            description = match.group(2).strip()[:200]
            dedup_key = description[:40].lower()
            if dedup_key in self._seen_keys or len(description) < 10:
                continue
            self._seen_keys.add(dedup_key)
            window = text[max(0, match.start() - 300) : min(len(text), match.end() + 300)]
            file_match = self.FILE_PATTERN.search(window)
            # Extract severity from the P[0-3] in the table row
            sev_match = re.search(r"P([0-3])", match.group(0))
            severity = f"P{sev_match.group(1)}" if sev_match else "unclassified"
            self._bugs.append(
                {
                    "description": description,
                    "file": file_match.group(1) if file_match else None,
                    "line": int(line_str) if line_str.isdigit() else None,
                    "severity": severity,
                    "source": source,
                    "found_at": time.time(),
                }
            )

        # Phase 2: regex patterns
        for pattern in self.BUG_PATTERNS:
            for match in pattern.finditer(text):
                if _in_fix_section(match.start()):
                    continue
                description = match.group(1).strip().split("\n")[0][:200]
                # Strip leading separators from description
                description = re.sub(r"^[—–:-]+\s*", "", description)
                if not description or len(description) < 10:
                    continue
                # False positive filter
                if any(fp.search(description) for fp in self.FALSE_POSITIVE_PATTERNS):
                    continue
                dedup_key = description[:40].lower()
                if dedup_key in self._seen_keys:
                    continue
                self._seen_keys.add(dedup_key)
                start = max(0, match.start() - 300)
                end = min(len(text), match.end() + 300)
                window = text[start:end]
                file_match = self.FILE_PATTERN.search(window)
                line_match = self.LINE_PATTERN.search(window)
                file_name = file_match.group(1) if file_match else None
                line_num = None
                if file_match and file_match.group(2):
                    line_num = int(file_match.group(2))
                elif line_match:
                    line_num = int(line_match.group(1))
                # Infer severity: P-level → emoji → section header → table → keywords
                severity = "unclassified"
                match_line = text[max(0, match.start() - 5) : match.end()]
                sev_level_match = re.search(r"\bP([0-3])\b", match_line)
                if sev_level_match:
                    severity = f"P{sev_level_match.group(1)}"
                else:
                    for emoji, level in self.EMOJI_SEVERITY.items():
                        if emoji in match_line:
                            severity = level
                            break
                if severity == "unclassified":
                    severity = self._lookup_severity(severity_map, match.start())
                if severity == "unclassified" and table_sev_map:
                    bug_num_match = re.search(r"(?:Bug|Issue|Vuln|#)\s*#?(\d+)", match.group(0))
                    if bug_num_match:
                        table_sev = table_sev_map.get(int(bug_num_match.group(1)))
                        if table_sev:
                            severity = table_sev
                if severity == "unclassified":
                    context_lower = window.lower()
                    for kw, level in self.SEVERITY_KEYWORDS.items():
                        if kw in context_lower:
                            severity = level
                            break
                self._bugs.append(
                    {
                        "description": description,
                        "file": file_name,
                        "line": line_num,
                        "severity": severity,
                        "source": source,
                        "found_at": time.time(),
                    }
                )

    @property
    def bugs(self) -> list[dict]:
        return self._bugs

    @property
    def summary(self) -> dict:
        sev_counts = {"P0": 0, "P1": 0, "P2": 0, "P3": 0, "unclassified": 0}
        for b in self._bugs:
            sev = b.get("severity", "unclassified")
            sev_counts[sev] = sev_counts.get(sev, 0) + 1
        return {
            "total_bugs": len(self._bugs),
            "files_affected": list({b["file"] for b in self._bugs if b.get("file")}),
            "severity_counts": sev_counts,
            "by_source": {
                s: len([b for b in self._bugs if b["source"] == s]) for s in {b["source"] for b in self._bugs}
            },
        }


class SeverityAssessmentHook(Hook):
    """Classifies detected bugs by severity (P0-P3).

    Subscribes to POST_OUTPUT to scan the final response for severity
    indicators and build a classification map.
    """

    SEVERITY_MAP = {
        "P0": [
            "critical",
            "security vulnerability",
            "injection",
            "data breach",
            "hardcoded secret",
            "remote code execution",
        ],
        "P1": [
            "high",
            "logic error",
            "race condition",
            "resource leak",
            "off-by-one",
            "division by zero",
            "null pointer",
            "key error",
        ],
        "P2": [
            "medium",
            "precision",
            "rounding",
            "missing validation",
            "swallowed exception",
            "edge case",
            "floating point",
        ],
        "P3": ["low", "cosmetic", "style", "naming", "documentation"],
    }

    def __init__(self):
        self._classifications: list[dict] = []
        self._seen_hashes: set[int] = set()

    def handles(self) -> list[HookEvent]:
        return [HookEvent.POST_OUTPUT, HookEvent.SESSION_END]

    async def execute(self, ctx: HookContext) -> HookContext:
        if ctx.event == HookEvent.POST_OUTPUT:
            content = ""
            if ctx.llm_response:
                content = getattr(ctx.llm_response, "content", "") or ""
            if content:
                text_hash = hash(content[:500])
                if text_hash not in self._seen_hashes:
                    self._seen_hashes.add(text_hash)
                    self.classify(content)
        elif ctx.event == HookEvent.SESSION_END:
            ctx.metadata["severity_summary"] = self.summary
        return ctx

    _EMOJI_SEV_MAP = {
        "🔴": "P0",
        "🟠": "P1",
        "🟡": "P2",
        "🟢": "P3",
        "⚠️": "P1",
        "🐛": "P2",
    }

    def classify(self, output: str) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {"P0": [], "P1": [], "P2": [], "P3": [], "unclassified": []}
        output_lower = output.lower()

        # Pass 1: explicit P0-P3 markers (P0: Title, P0 — Title)
        for level in ["P0", "P1", "P2", "P3"]:
            for match in re.finditer(rf"\b{level}\b\s*[—–:.-]\s*(.+?)(?:\n|$)", output, re.IGNORECASE):
                desc = match.group(1).strip()[:150]
                if desc and len(desc) >= 5:
                    result[level].append(desc)

        has_explicit = any(result[level] for level in ["P0", "P1", "P2", "P3"])

        if not has_explicit:
            # Pass 2: emoji + severity word/level: 🔴 CRITICAL — Title or 🔴 P0 — Title
            emoji_sev_word = {
                "🔴": "P0",
                "🟠": "P1",
                "🟡": "P2",
                "🟢": "P3",
                "⚠️": "P1",
                "🐛": "P2",
            }
            sev_words = {"CRITICAL": "P0", "HIGH": "P1", "MEDIUM": "P2", "LOW": "P3", "WARNING": "P1", "INFO": "P3"}
            for match in re.finditer(
                r"(🔴|🟠|🟡|🟢|⚠️|🐛)\s*(?:CRITICAL|HIGH|MEDIUM|LOW|INFO|WARNING|P[0-3])?\s*[—–:.-]\s*(.+?)(?:\n|$)",
                output,
                re.IGNORECASE,
            ):
                emoji = match.group(1)
                desc = match.group(2).strip()[:150]
                if desc and len(desc) >= 5:
                    level = emoji_sev_word.get(emoji, "unclassified")
                    result[level].append(desc)

        if not any(result[level] for level in ["P0", "P1", "P2", "P3"]):
            # Pass 3: keyword presence
            for level in ["P0", "P1", "P2", "P3"]:
                for kw in self.SEVERITY_MAP[level]:
                    if kw in output_lower:
                        result[level].append(f"Keyword match: {kw}")
                        break

        self._classifications.append(result)
        return result

    @property
    def summary(self) -> dict:
        total = {"P0": 0, "P1": 0, "P2": 0, "P3": 0}
        for cls in self._classifications:
            for level in total:
                total[level] += len(cls.get(level, []))
        return {"severity_counts": total, "sessions_analyzed": len(self._classifications)}

    @property
    def classifications(self) -> list[dict]:
        return self._classifications


class CodeAnalysisHook(Hook):
    """Tracks code analysis coverage metrics.

    Subscribes to tool execution events to record which files were scanned,
    which tools were used, and compute analysis coverage.
    """

    def __init__(self):
        self._files_scanned: set[str] = set()
        self._tools_invoked: dict[str, int] = {}
        self._start_time: float = 0
        self._total_tool_calls: int = 0
        self._file_sources: dict[str, str] = {}

    def handles(self) -> list[HookEvent]:
        return [HookEvent.SESSION_START, HookEvent.PRE_TOOL_USE, HookEvent.POST_TOOL_USE, HookEvent.SESSION_END]

    async def execute(self, ctx: HookContext) -> HookContext:
        if ctx.event == HookEvent.SESSION_START:
            self._start_time = time.time()

        elif ctx.event == HookEvent.PRE_TOOL_USE and ctx.tool_name:
            self._tools_invoked[ctx.tool_name] = self._tools_invoked.get(ctx.tool_name, 0) + 1
            self._total_tool_calls += 1
            if ctx.tool_arguments:
                args_str = str(ctx.tool_arguments)
                for match in re.finditer(r"(\w+\.py)", args_str):
                    self._files_scanned.add(match.group(1))
                    self._file_sources[match.group(1)] = ctx.tool_name

        elif ctx.event == HookEvent.POST_TOOL_USE and ctx.tool_result:
            if ctx.tool_name in ("read_file", "glob_find"):
                for match in re.finditer(r"(\w+\.py)", str(ctx.tool_result)):
                    self._files_scanned.add(match.group(1))
                    self._file_sources[match.group(1)] = ctx.tool_name

        elif ctx.event == HookEvent.SESSION_END:
            elapsed = time.time() - self._start_time if self._start_time else 0
            ctx.metadata["analysis_coverage"] = {
                "files_scanned": sorted(self._files_scanned),
                "files_count": len(self._files_scanned),
                "tools_used": dict(self._tools_invoked),
                "total_tool_calls": self._total_tool_calls,
                "elapsed_seconds": round(elapsed, 2),
            }
        return ctx

    @property
    def coverage(self) -> dict:
        return {
            "files_scanned": sorted(self._files_scanned),
            "files_count": len(self._files_scanned),
            "tools_used": dict(self._tools_invoked),
            "total_tool_calls": self._total_tool_calls,
        }


# ---------------------------------------------------------------------------
# Specialist Agent Prompts (for orchestrated audit mode)
# ---------------------------------------------------------------------------

STATIC_ANALYZER_PROMPT = """\
You are a static code analysis specialist. Your job is to find bugs by reading
source code and identifying patterns that indicate defects.

Focus areas:
- Security vulnerabilities (injection, hardcoded secrets, missing validation)
- Logic errors (off-by-one, wrong operators, missing null checks)
- Resource leaks (unclosed files, connections)
- Concurrency issues (missing locks, race conditions)

Output a structured list of findings with file, line, category, and description."""

TEST_GENERATOR_PROMPT = """\
You are a test generation specialist. Given a description of bugs found in code,
generate unit tests that would expose those bugs.

For each bug:
1. Write a test function that triggers the buggy behavior
2. Include an assertion that would fail with the current code
3. Describe what the correct behavior should be

Output test code in Python with clear comments explaining each test case."""

FIX_PROPOSER_PROMPT = """\
You are a fix proposal specialist. Given descriptions of bugs found in code,
propose concrete fixes.

For each bug:
1. Show the problematic code section
2. Explain what's wrong
3. Show the corrected code with a diff-like format
4. Note any considerations for the fix (breaking changes, performance)

Output structured fix proposals with before/after code snippets."""

AUDIT_KEYWORDS = {
    "static_analyzer": [
        "analyze",
        "scan",
        "find bugs",
        "static",
        "review",
        "audit",
        "check",
        "inspect",
        "examine",
        "look for",
        "vulnerability",
        "security",
    ],
    "test_generator": [
        "test",
        "generate test",
        "unit test",
        "pytest",
        "test case",
        "write test",
        "coverage",
        "expose bug",
    ],
    "fix_proposer": [
        "fix",
        "repair",
        "patch",
        "resolve",
        "correct",
        "solution",
        "suggestion",
        "how to fix",
        "proposal",
    ],
}


# ---------------------------------------------------------------------------
# Orchestrated Audit Mode
# ---------------------------------------------------------------------------


def _build_audit_orchestrator():
    """Create Orchestrator with specialist agents for audit mode."""
    from koboi.client import Client
    from koboi.orchestration.orchestrator import Orchestrator
    from koboi.orchestration.router import KeywordRouter
    from koboi.logger import AgentLogger
    from koboi.memory import ConversationMemory
    from koboi.loop import AgentCore
    from koboi.tools.builtin import register_all
    from koboi.tools.registry import ToolRegistry
    from koboi.rag.chunker import SentenceChunker
    from koboi.rag.retriever import KeywordRetriever
    from koboi.rag.augmentation import OnTheFlyAugmentation
    from koboi.rag.types import Document
    from pathlib import Path

    client = Client(provider="anthropic")
    logger = AgentLogger(session_id="ex23_audit")

    # Load buggy code as RAG knowledge
    buggy_dir = Path(__file__).parent / "data" / "buggy_code"
    chunker = SentenceChunker(max_chunk_size=400)
    all_chunks = []
    for py_file in sorted(buggy_dir.glob("*.py")):
        content = py_file.read_text()
        doc = Document(id=py_file.stem, title=py_file.stem, content=content)
        all_chunks.extend(chunker.chunk(doc))

    retriever = KeywordRetriever(all_chunks) if all_chunks else None

    def _make_agent(system_prompt: str) -> AgentCore:
        registry = ToolRegistry()
        register_all(registry)
        registry.keep_only(["read_file", "grep_search", "run_shell", "calculate", "memory_store", "memory_recall"])
        aug = OnTheFlyAugmentation(retriever=retriever, top_k=5, logger=logger) if retriever else None
        return AgentCore(
            client=client,
            memory=ConversationMemory(logger=logger, system_prompt=system_prompt),
            tools=registry,
            max_iterations=15,
            logger=logger,
            augmentation=aug,
            max_context_tokens=12000,
        )

    # Build specialist agents
    specialist_agents = {
        "static_analyzer": _make_agent(STATIC_ANALYZER_PROMPT),
        "test_generator": _make_agent(TEST_GENERATOR_PROMPT),
        "fix_proposer": _make_agent(FIX_PROPOSER_PROMPT),
    }

    # Custom router
    router = KeywordRouter()
    router.KEYWORD_MAP = AUDIT_KEYWORDS

    # Subclass Orchestrator to route to our specialist agents
    class AuditOrchestrator(Orchestrator):
        def __init__(self, specialists, **kwargs):
            super().__init__(**kwargs)
            self._specialists = specialists

        async def _run_single(self, agent_name, query):
            from koboi.types import AgentResult
            from koboi.tokens import estimate_tokens

            if agent_name in self._specialists:
                agent = self._specialists[agent_name]
                if self.logger:
                    self.logger.log_agent_dispatch(agent_name, query, "specialist")
                start = time.time()
                tokens_before = estimate_tokens(agent.memory.get_messages())
                failed = False
                try:
                    answer = await agent.run(query)
                except Exception as e:
                    import logging

                    logging.getLogger(__name__).error("Agent %s failed: %s", agent_name, e, exc_info=True)
                    answer = f"Error: {e}"
                    failed = True
                elapsed = time.time() - start
                tokens_after = estimate_tokens(agent.memory.get_messages())
                tokens = tokens_after - tokens_before
                return AgentResult(
                    agent_name=agent_name,
                    answer=answer,
                    elapsed_seconds=elapsed,
                    tokens_used=tokens,
                    failed=failed,
                )
            return await super()._run_single(agent_name, query)

    orchestrator = AuditOrchestrator(
        specialists=specialist_agents,
        client=client,
        router=router,
        logger=logger,
    )

    return orchestrator, router


# ---------------------------------------------------------------------------
# Custom Eval Scorers
# ---------------------------------------------------------------------------


class BugDetectionScorer(BaseScorer):
    """Scores whether the output contains structured bug reports."""

    PATTERNS = {
        "severity": re.compile(r"P[0-3]", re.IGNORECASE),
        "location": re.compile(r"\w+\.py:\d+"),
        "category": re.compile(
            r"(?:Category|Type|Class|category|type|class)\s*:\s*(Security|Logic|Resource|Concurrency|Data)",
            re.IGNORECASE,
        ),
        "description": re.compile(r"(?:bug|issue|vuln|error|flaw)", re.IGNORECASE),
    }

    async def score(self, case, output: str, context: dict):
        from koboi.types import EvalScore

        matched = sum(1 for p in self.PATTERNS.values() if p.search(output))
        ratio = matched / len(self.PATTERNS)
        details = [name for name, p in self.PATTERNS.items() if p.search(output)]
        missing = [name for name, p in self.PATTERNS.items() if not p.search(output)]
        reason = f"{matched}/{len(self.PATTERNS)} patterns found: {details}"
        if missing:
            reason += f", missing: {missing}"
        return EvalScore("bug_detection", round(ratio, 3), reason)


class SeverityAccuracyScorer(BaseScorer):
    """Scores whether the output uses proper P0-P3 severity classification."""

    async def score(self, case, output: str, context: dict):
        from koboi.types import EvalScore

        p_counts = {}
        for level in ["P0", "P1", "P2", "P3"]:
            matches = re.findall(rf"\b{level}\b", output, re.IGNORECASE)
            p_counts[level] = len(matches)

        total_mentions = sum(p_counts.values())
        if total_mentions == 0:
            return EvalScore("severity_accuracy", 0.2, "No P0-P3 classifications found")

        levels_used = sum(1 for c in p_counts.values() if c > 0)
        score = min(1.0, levels_used / 3.0)
        reason = f"Severity mentions: {p_counts}, {levels_used} levels used"
        return EvalScore("severity_accuracy", round(score, 3), reason)


class CodeReferenceScorer(BaseScorer):
    """Scores whether the output references specific code files and line numbers."""

    FILE_REFS = re.compile(
        r"(auth_service|data_processor|cache_manager|order_calculator"
        r"|payment_gateway|inventory_manager)\.py"
    )
    FILE_LINE = re.compile(r"(\w+)\.py:(\d+)")
    LINE_NUM = re.compile(r"(?:line|lines?)\s+#?(\d+(?:\s*[-,]\s*\d+)*)", re.IGNORECASE)

    async def score(self, case, output: str, context: dict):
        from koboi.types import EvalScore

        file_matches = set(self.FILE_REFS.findall(output))
        line_from_file = set(self.FILE_LINE.findall(output))
        line_from_text = self.LINE_NUM.findall(output)
        total_lines = len(line_from_file) + len(line_from_text)

        file_score = min(1.0, len(file_matches) / 4.0)
        line_score = min(1.0, total_lines / 5.0)
        combined = (file_score * 0.6) + (line_score * 0.4)

        reason = f"Files: {sorted(file_matches)}, file:lines: {len(line_from_file)}, text lines: {len(line_from_text)}"
        return EvalScore("code_reference", round(min(1.0, combined), 3), reason)


class GroundTruthScorer(BaseScorer):
    """Scores output against KNOWN_BUGS.md ground truth for accuracy."""

    _known_bugs: list[dict] | None = None

    def _load_known_bugs(self) -> list[dict]:
        if self._known_bugs is not None:
            return self._known_bugs
        from pathlib import Path

        bugs_file = Path(__file__).parent / "data" / "buggy_code" / "KNOWN_BUGS.md"
        if not bugs_file.exists():
            return []
        content = bugs_file.read_text()
        bugs = []
        current_file = ""
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("## "):
                current_file = line.lstrip("# ").strip()
            elif line.startswith("- Line ") and current_file:
                parts = line.split("|")
                if len(parts) >= 3:
                    line_num = parts[0].replace("- Line", "").split(":")[0].strip()
                    severity = parts[-2].strip() if len(parts) >= 3 else ""
                    bugs.append({"file": current_file, "line": line_num, "severity": severity})
        self.__class__._known_bugs = bugs
        return bugs

    async def score(self, case, output: str, context: dict):
        from koboi.types import EvalScore

        known = self._load_known_bugs()
        if not known:
            return EvalScore("ground_truth", 0.5, "No ground truth data available")
        found = 0
        for bug in known:
            if bug["file"].replace(".py", "") in output and str(bug["line"]) in output:
                found += 1
        recall = found / len(known)
        reason = f"Found {found}/{len(known)} known bugs ({recall:.0%} recall)"
        return EvalScore("ground_truth", round(recall, 3), reason)


# ---------------------------------------------------------------------------
# Eval Suite Builder
# ---------------------------------------------------------------------------


def _build_eval_cases():
    """Build 6 evaluation test cases for bug-hunting scenarios."""
    from koboi.types import EvalCase

    return [
        EvalCase(
            name="SQL injection detection",
            user_message="Analyze auth_service.py for security vulnerabilities",
            expected_keywords=["injection", "SQL", "concatenation", "auth_service"],
            expected_tools=["read_file", "grep_search"],
            max_iterations=15,
        ),
        EvalCase(
            name="Off-by-one detection",
            user_message="Find logic errors in data_processor.py, especially in batch processing",
            expected_keywords=["off-by-one", "range", "data_processor", "batch"],
            expected_tools=["read_file", "grep_search"],
            max_iterations=15,
        ),
        EvalCase(
            name="Race condition detection",
            user_message="Check cache_manager.py for concurrency issues and race conditions",
            expected_keywords=["race", "lock", "concurrent", "cache_manager"],
            expected_tools=["read_file", "grep_search"],
            max_iterations=15,
        ),
        EvalCase(
            name="Currency precision errors",
            user_message="Review order_calculator.py for pricing and calculation bugs",
            expected_keywords=["float", "precision", "rounding", "order_calculator"],
            expected_tools=["read_file", "calculate"],
            max_iterations=15,
        ),
        EvalCase(
            name="Full security audit",
            user_message="Perform a complete security audit on all files in the buggy_code directory",
            expected_keywords=["vulnerability", "security", "injection", "hardcoded"],
            expected_tools=["grep_search", "read_file"],
            max_iterations=20,
        ),
        EvalCase(
            name="Bug summary report",
            user_message="Find all bugs across all buggy code files and provide a severity-ranked summary",
            expected_keywords=["P0", "P1", "severity", "bug"],
            expected_tools=["grep_search", "read_file"],
            max_iterations=20,
        ),
    ]


def _build_scorers(client):
    """Build custom + built-in scorers for the eval suite."""
    from koboi.eval.scorers import (
        ToolUsageScorer,
        KeywordPresenceScorer,
        OutputLengthScorer,
        IterationEfficiencyScorer,
        LLMJudgeScorer,
    )

    return [
        BugDetectionScorer(),
        SeverityAccuracyScorer(),
        CodeReferenceScorer(),
        GroundTruthScorer(),
        KeywordPresenceScorer(),
        ToolUsageScorer(),
        OutputLengthScorer(),
        IterationEfficiencyScorer(),
        LLMJudgeScorer(client=client),
    ]


# ---------------------------------------------------------------------------
# Display Helpers
# ---------------------------------------------------------------------------


def _print_bug_summary(bug_tracker: BugTrackerHook) -> None:
    summary = bug_tracker.summary
    if summary["total_bugs"] == 0:
        console.print("  [dim]No bugs tracked yet[/dim]\n")
        return

    table = Table(title=f"Bug Tracker ({summary['total_bugs']} found)", show_header=True, header_style="bold yellow")
    table.add_column("File", style="cyan", width=20)
    table.add_column("Line", width=6)
    table.add_column("Description", style="dim", ratio=1)
    table.add_column("Sev", width=4)
    table.add_column("Source", width=10)

    for bug in bug_tracker.bugs[-10:]:
        table.add_row(
            bug.get("file", "-"),
            str(bug.get("line", "-")),
            bug["description"][:80],
            bug.get("severity", "?")[:2],
            bug["source"],
        )
    console.print(table)
    console.print()


def _print_severity(severity: SeverityAssessmentHook) -> None:
    summary = severity.summary
    counts = summary["severity_counts"]
    console.print("  [dim]Severity:[/dim] ", end="")
    for level in ["P0", "P1", "P2", "P3"]:
        c = counts.get(level, 0)
        color = {"P0": "red", "P1": "yellow", "P2": "blue", "P3": "dim"}[level]
        console.print(f"[{color}]{level}:{c}[/{color}]", end="  ")
    console.print("\n")


def _print_severity_from_tracker(bug_tracker: BugTrackerHook) -> None:
    counts = bug_tracker.summary["severity_counts"]
    console.print("  [dim]Severity:[/dim] ", end="")
    for level in ["P0", "P1", "P2", "P3"]:
        c = counts.get(level, 0)
        color = {"P0": "red", "P1": "yellow", "P2": "blue", "P3": "dim"}[level]
        console.print(f"[{color}]{level}:{c}[/{color}]", end="  ")
    console.print("\n")


def _print_coverage(analysis: CodeAnalysisHook) -> None:
    cov = analysis.coverage
    files = ", ".join(cov["files_scanned"]) or "-"
    tools = ", ".join(f"{k}({v})" for k, v in cov["tools_used"].items()) or "-"
    console.print(f"  [dim]Coverage: {cov['files_count']} files ({files})[/dim]")
    console.print(f"  [dim]Tools: {tools}[/dim]\n")


def _print_hooks(agent) -> None:
    hooks_info = agent.core.hooks.list_hooks()
    table = Table(title="Registered Hooks", show_header=True, header_style="bold cyan")
    table.add_column("Hook", style="cyan")
    table.add_column("Events", style="green", max_width=60)
    for info in hooks_info:
        table.add_row(info["name"], ", ".join(info["events"]))
    console.print(table)
    console.print()


def _find_telemetry(agent):
    hook = agent.core.hooks.find_hook(lambda h: hasattr(h, "telemetry") and hasattr(h.telemetry, "snapshot"))
    return hook.telemetry if hook else None


# ---------------------------------------------------------------------------
# Bug Hunt Mode (single agent with all features)
# ---------------------------------------------------------------------------

HUNT_QUESTIONS = [
    "Analyze auth_service.py for security vulnerabilities",
    "Find all bugs in data_processor.py, especially in batch processing",
    "Check cache_manager.py for concurrency issues and race conditions",
    "Review order_calculator.py for pricing and calculation bugs",
    "Perform a security audit on payment_gateway.py -- check for hardcoded credentials and injection vulnerabilities",
    "Find logic errors in inventory_manager.py -- focus on deadlock potential and data consistency",
    "Cross-reference bugs across payment_gateway.py and order_calculator.py that might interact",
]


def run_bug_hunt_automatic(agent, bug_tracker, severity, analysis):
    """Run predefined bug-hunting queries."""
    start = time.time()

    def _post_answer(result, q, i, total):
        _print_bug_summary(bug_tracker)
        _print_severity_from_tracker(bug_tracker)
        _print_coverage(analysis)

        telemetry = _find_telemetry(agent)
        if telemetry:
            snap = telemetry.snapshot
            console.print(
                f"  [dim]telemetry: iters={snap.total_iterations} "
                f"tools={snap.total_tool_calls} "
                f"health={telemetry.health_score():.0f}[/dim]\n"
            )

    automatic_batch(agent, HUNT_QUESTIONS, post_answer=_post_answer)

    # Final session summary
    elapsed = time.time() - start
    telemetry = _find_telemetry(agent)

    summary_lines = [
        f"Duration: {elapsed:.1f}s",
        f"Bugs Found: {bug_tracker.summary['total_bugs']}",
        f"Files Scanned: {analysis.coverage['files_count']}",
    ]

    bt_sev = bug_tracker.summary["severity_counts"]
    summary_lines.append(
        f"Severity Breakdown: P0={bt_sev.get('P0', 0)} P1={bt_sev.get('P1', 0)} "
        f"P2={bt_sev.get('P2', 0)} P3={bt_sev.get('P3', 0)}"
    )

    if telemetry:
        summary_lines.append(f"Health Score: {telemetry.health_score()}/100")
        summary_lines.append(f"Total Tool Calls: {telemetry.snapshot.total_tool_calls}")

    console.print(Panel("\n".join(summary_lines), title="Bug Hunt Summary", border_style="green"))


def run_bug_hunt_interactive(agent, bug_tracker, severity, analysis):
    """Interactive bug hunting with custom commands."""
    messages = 0

    def _post_receive(result, a):
        nonlocal messages
        messages += 1
        _print_severity(severity)
        _print_coverage(analysis)

    extra_commands = {
        "bugs": lambda a: _print_bug_summary(bug_tracker),
        "severity": lambda a: console.print(severity.summary),
        "coverage": lambda a: _print_coverage(analysis),
        "hooks": lambda a: _print_hooks(a),
        "reset": lambda a: (
            a.reset(),
            console.print("[yellow]Conversation reset. Bug tracking preserved.[/yellow]\n"),
        ),
    }

    console.print("[dim]Commands: bugs, severity, coverage, hooks, reset, quit[/dim]\n")
    interactive_loop(
        agent,
        extra_commands=extra_commands,
        post_receive=_post_receive,
    )

    console.print(
        Panel(
            f"Messages: {messages}\n"
            f"Bugs Found: {bug_tracker.summary['total_bugs']}\n"
            f"Files Scanned: {analysis.coverage['files_count']}",
            title="Session Summary",
            border_style="green",
        )
    )


# ---------------------------------------------------------------------------
# Audit Mode (multi-agent orchestrated)
# ---------------------------------------------------------------------------

AUDIT_QUERIES = [
    "Analyze auth_service.py for security vulnerabilities and propose fixes",
    "Find race conditions in cache_manager.py and generate tests to expose them",
    "Review data_processor.py for logic errors, suggest fixes, and write tests",
]


def run_audit_automatic():
    """Run orchestrated audit queries."""
    orchestrator, router = _build_audit_orchestrator()

    # Show routing table
    console.print("\n[bold]Audit Routing Table:[/bold]")
    route_table = Table(show_header=True, header_style="bold magenta")
    route_table.add_column("Query", style="green", max_width=50)
    route_table.add_column("Agents", style="cyan")
    route_table.add_column("Confidence", style="yellow")
    route_table.add_column("Method", style="dim")

    for q in AUDIT_QUERIES:
        decision = run_async(router.route(q))
        route_table.add_row(
            q[:50],
            ", ".join(decision.agents),
            f"{decision.confidence:.2f}",
            decision.method,
        )
    console.print(route_table)
    console.print()

    # Execute
    for i, q in enumerate(AUDIT_QUERIES, 1):
        console.rule(f"[bold]Audit {i}: {q}[/bold]")
        try:
            result = run_async(orchestrator.run(q, mode="sequential"))
            console.print(
                f"  Routing: {result.routing.method} -> "
                f"[cyan]{result.routing.agents}[/cyan] "
                f"(confidence: {result.routing.confidence:.2f})"
            )
            for ar in result.agent_results:
                status = "[red]FAILED[/red]" if ar.failed else "[green]OK[/green]"
                console.print(
                    f"  Agent [cyan]{ar.agent_name}[/cyan]: {ar.elapsed_seconds:.1f}s, {ar.tokens_used} tokens {status}"
                )
            console.print(
                Panel(
                    Markdown(result.final_answer[:2000]),
                    title=f"Synthesized Answer ({result.total_elapsed_seconds:.1f}s)",
                )
            )
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
        console.print()


def run_audit_interactive():
    """Interactive audit with routing display."""
    orchestrator, router = _build_audit_orchestrator()
    console.print("[dim]Available specialists: static_analyzer, test_generator, fix_proposer[/dim]")
    console.print("[dim]Type any analysis request. Routing is automatic.[/dim]\n")

    from rich.prompt import Prompt

    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("[dim]Bye![/dim]")
            break

        if user_input.strip().lower() in ("quit", "exit", "q"):
            console.print("[dim]Bye![/dim]")
            break
        if not user_input.strip():
            continue

        decision = run_async(router.route(user_input))
        console.print(
            f"[dim]Routed: {decision.method} -> "
            f"[cyan]{', '.join(decision.agents)}[/cyan] "
            f"(confidence: {decision.confidence:.2f})[/dim]"
        )

        try:
            result = run_async(orchestrator.run(user_input, mode="sequential"))
            console.print(
                Panel(
                    Markdown(result.final_answer[:2000]),
                    title="Audit Result",
                    border_style="green",
                )
            )
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


# ---------------------------------------------------------------------------
# Eval Mode
# ---------------------------------------------------------------------------


def run_eval(verbose: bool):
    """Run the full evaluation suite."""
    from koboi.eval.runner import EvalRunner

    agent = create_agent("23_swe_bug_hunter", verbose=verbose)
    console.print(f"[dim]Agent: {agent.config.agent_name} | Model: {agent.config.model}[/dim]\n")

    cases = _build_eval_cases()
    scorers = _build_scorers(agent.core.client)

    # Show test cases
    case_table = Table(title="Bug Hunter Eval Cases", show_header=True, header_style="bold cyan")
    case_table.add_column("#", width=3)
    case_table.add_column("Name", style="cyan", width=25)
    case_table.add_column("Query", ratio=1)
    case_table.add_column("Keywords", style="yellow", max_width=30)

    for i, c in enumerate(cases, 1):
        case_table.add_row(
            str(i),
            c.name,
            c.user_message[:60],
            ", ".join(c.expected_keywords[:3]),
        )
    console.print(case_table)
    console.print()

    runner = EvalRunner(
        harness_factory=lambda: create_agent("23_swe_bug_hunter", verbose=verbose),
        scorers=scorers,
    )

    console.print("[bold]Running evaluation suite...[/bold]\n")
    results = run_async(runner.run_suite(cases))

    # Print detailed results
    console.print()
    for r in results:
        status = "PASS" if r.overall_score >= 0.6 else "FAIL"
        border = "green" if r.overall_score >= 0.6 else "red"

        detail_table = Table(
            show_header=True, header_style="bold", title=f"[{status}] {r.case_name} -- {r.overall_score:.1%}"
        )
        detail_table.add_column("Scorer", style="cyan", width=22)
        detail_table.add_column("Score", width=8)
        detail_table.add_column("Reason", style="dim", ratio=1)

        for s in r.scores:
            filled = int(s.value * 10)
            bar = "+" * filled + "-" * (10 - filled)
            detail_table.add_row(s.name, f"[{bar}] {s.value:.2f}", s.reason[:80])

        console.print(detail_table)
        console.print()

    total = len(results)
    passed = sum(1 for r in results if r.overall_score >= 0.6)
    avg = sum(r.overall_score for r in results) / total if total else 0

    console.print(
        Panel(
            f"[bold]Evaluation Summary[/bold]\n\n"
            f"Passed: {passed}/{total}\n"
            f"Average Score: {avg:.1%}\n"
            f"Total Duration: {sum(r.duration_seconds for r in results):.1f}s",
            title="Bug Hunter Eval",
            border_style="green" if passed == total else "yellow",
        )
    )


# ---------------------------------------------------------------------------
# Stress Mode (guardrails, policy, rate limits)
# ---------------------------------------------------------------------------


def run_stress(verbose: bool):
    """Run guardrail and policy stress tests."""
    from pathlib import Path
    from koboi.eval.runner import EvalRunner
    from koboi.eval.scorers import KeywordPresenceScorer

    agent = create_agent("23_swe_bug_hunter", verbose=verbose)
    console.print(f"[dim]Agent: {agent.config.agent_name} | Model: {agent.config.model}[/dim]")
    console.print("[bold yellow]Running guardrail and policy stress tests...[/bold]\n")

    stress_file = Path(__file__).parent / "data" / "eval_cases" / "guardrail_stress_eval.yaml"
    if not stress_file.exists():
        console.print("[red]guardrail_stress_eval.yaml not found[/red]")
        return

    import yaml
    from koboi.types import EvalCase

    raw = yaml.safe_load(stress_file.read_text())
    cases = []
    for c in raw.get("cases", []):
        cases.append(
            EvalCase(
                name=c["name"],
                user_message=c["input"],
                expected_keywords=c.get("expected_keywords", []),
                expected_tools=c.get("expected_tools", []),
                max_iterations=c.get("max_iterations", 5),
            )
        )

    case_table = Table(title="Guardrail Stress Cases", show_header=True, header_style="bold yellow")
    case_table.add_column("#", width=3)
    case_table.add_column("Name", style="yellow", width=30)
    case_table.add_column("Input", ratio=1)
    for i, c in enumerate(cases, 1):
        case_table.add_row(str(i), c.name, c.user_message[:60])
    console.print(case_table)
    console.print()

    scorers = [KeywordPresenceScorer()]
    runner = EvalRunner(
        harness_factory=lambda: create_agent("23_swe_bug_hunter", verbose=verbose),
        scorers=scorers,
    )
    results = run_async(runner.run_suite(cases))

    for r in results:
        border = "yellow" if r.overall_score >= 0.3 else "red"
        console.print(
            f"  [{border}]{'PASS' if r.overall_score >= 0.3 else 'FAIL'}[/{border}] "
            f"{r.case_name} -- {r.overall_score:.1%}"
        )
    console.print()


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


@click.command()
@dual_mode_options
@click.option(
    "--run-mode",
    "-r",
    type=click.Choice(["bug_hunt", "audit", "eval", "stress"]),
    default="bug_hunt",
    help="Run mode: bug_hunt (single agent), audit (orchestrated), eval (evaluation suite), stress (guardrail stress test)",
)
def main(mode: str, verbose: bool, run_mode: str):
    """Example 23: SWE-Agent Bug Hunter -- complex multi-feature example."""
    setup_example(
        "Example 23: SWE-Agent Bug Hunter",
        "Autonomous code analysis agent (SWE-Agent/OpenHands clone).\n"
        "Combines RAG + Tools + Skills + Guardrails + Policy + Hooks + Harness +\n"
        "Context + Orchestration + Eval in one complex example.\n\n"
        "[dim]Modes: --run-mode bug_hunt|audit|eval|stress[/dim]\n"
        "[dim]Interactive commands: bugs, severity, coverage, hooks, reset[/dim]",
    )

    if run_mode == "eval":
        run_eval(verbose)
        return

    if run_mode == "stress":
        run_stress(verbose)
        return

    if run_mode == "audit":
        if mode == "interactive":
            run_audit_interactive()
        else:
            run_audit_automatic()
        return

    # bug_hunt mode -- single agent with ALL features
    agent = create_agent("23_swe_bug_hunter", verbose=verbose)
    console.print(f"[dim]Agent: {agent.config.agent_name} | Model: {agent.config.model}[/dim]")

    # Add custom hooks
    bug_tracker = BugTrackerHook()
    severity = SeverityAssessmentHook()
    analysis = CodeAnalysisHook()

    agent.core.hooks.add(bug_tracker)
    agent.core.hooks.add(severity)
    agent.core.hooks.add(analysis)

    console.print("[dim]Added BugTrackerHook, SeverityAssessmentHook, CodeAnalysisHook[/dim]")

    # Show registered hooks
    _print_hooks(agent)

    if mode == "interactive":
        run_bug_hunt_interactive(agent, bug_tracker, severity, analysis)
    else:
        run_bug_hunt_automatic(agent, bug_tracker, severity, analysis)


if __name__ == "__main__":
    main()
