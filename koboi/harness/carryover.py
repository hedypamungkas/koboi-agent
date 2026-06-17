"""koboi/harness/carryover.py -- Metadata that survives context compaction.

When ContextManager truncates or summarizes messages, CarryoverState persists
outside the message history and is re-injected after compaction completes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class WorkLogEntry:
    iteration: int
    action: str  # "tool_call" | "llm_response" | "skill_activation"
    detail: str
    success: bool = True


@dataclass
class CarryoverState:
    user_goals: list[str] = field(default_factory=list)
    completed_goals: list[str] = field(default_factory=list)
    active_artifacts: dict[str, str] = field(default_factory=dict)
    verified_work: list[str] = field(default_factory=list)
    work_log: list[WorkLogEntry] = field(default_factory=list)
    invoked_tools: dict[str, int] = field(default_factory=dict)
    skills_used: list[str] = field(default_factory=list)
    custom_metadata: dict = field(default_factory=dict)

    max_log_entries: int = 50
    max_goals: int = 10
    max_artifacts: int = 20
    max_verified: int = 20

    def add_goal(self, goal: str) -> None:
        if goal not in self.user_goals:
            self.user_goals.append(goal)
            if len(self.user_goals) > self.max_goals:
                self.user_goals = self.user_goals[-self.max_goals :]

    def complete_goal(self, goal: str) -> None:
        if goal in self.user_goals:
            self.user_goals.remove(goal)
            self.completed_goals.append(goal)
            if len(self.completed_goals) > self.max_goals:
                self.completed_goals = self.completed_goals[-self.max_goals :]

    def add_artifact(self, name: str, description: str) -> None:
        self.active_artifacts[name] = description
        if len(self.active_artifacts) > self.max_artifacts:
            oldest = next(iter(self.active_artifacts))
            del self.active_artifacts[oldest]

    def mark_verified(self, work_description: str) -> None:
        if work_description not in self.verified_work:
            self.verified_work.append(work_description)
            if len(self.verified_work) > self.max_verified:
                self.verified_work = self.verified_work[-self.max_verified :]

    def record_tool_use(
        self, tool_name: str, arguments: str, result: str, *, iteration: int = 0, success: bool = True
    ) -> None:
        self.invoked_tools[tool_name] = self.invoked_tools.get(tool_name, 0) + 1
        self.work_log.append(
            WorkLogEntry(
                iteration=iteration,
                action="tool_call",
                detail=f"{tool_name}({arguments[:80]}) -> {result[:80]}",
                success=success,
            )
        )
        if len(self.work_log) > self.max_log_entries:
            self.work_log = self.work_log[-self.max_log_entries :]

    def record_skill(self, skill_name: str) -> None:
        if skill_name not in self.skills_used:
            self.skills_used.append(skill_name)

    def to_context_message(self) -> str:
        """Serialize to plain text for context injection."""
        parts = ["<harness-carryover>"]
        if self.user_goals:
            parts.append(f"Goals: {json.dumps(self.user_goals, ensure_ascii=False)}")
        if self.completed_goals:
            parts.append(f"Completed: {json.dumps(self.completed_goals, ensure_ascii=False)}")
        if self.active_artifacts:
            parts.append(f"Artifacts: {json.dumps(self.active_artifacts, ensure_ascii=False)}")
        if self.invoked_tools:
            parts.append(f"Tools used: {json.dumps(self.invoked_tools, ensure_ascii=False)}")
        if self.skills_used:
            parts.append(f"Skills: {json.dumps(self.skills_used, ensure_ascii=False)}")
        if self.verified_work:
            parts.append(f"Verified: {json.dumps(self.verified_work, ensure_ascii=False)}")
        if not any(
            [
                self.user_goals,
                self.completed_goals,
                self.active_artifacts,
                self.invoked_tools,
                self.skills_used,
                self.verified_work,
            ]
        ):
            return ""
        parts.append("</harness-carryover>")
        return "\n".join(parts)

    @classmethod
    def from_context_message(cls, content: str) -> CarryoverState:
        """Deserialize from a previously injected context message."""
        state = cls()
        if "<harness-carryover>" not in content:
            return state

        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("Goals:"):
                state.user_goals = _try_json_parse_list(line.split("Goals:", 1)[1].strip())
            elif line.startswith("Completed:"):
                state.completed_goals = _try_json_parse_list(line.split("Completed:", 1)[1].strip())
            elif line.startswith("Artifacts:"):
                state.active_artifacts = _try_json_parse_dict(line.split("Artifacts:", 1)[1].strip())
            elif line.startswith("Tools used:"):
                state.invoked_tools = _try_json_parse_counts(line.split("Tools used:", 1)[1].strip())
            elif line.startswith("Skills:"):
                state.skills_used = _try_json_parse_list(line.split("Skills:", 1)[1].strip())
            elif line.startswith("Verified:"):
                state.verified_work = _try_json_parse_list(line.split("Verified:", 1)[1].strip())
        return state

    def summary(self) -> dict:
        return {
            "goals": len(self.user_goals),
            "artifacts": len(self.active_artifacts),
            "verified": len(self.verified_work),
            "tool_calls": sum(self.invoked_tools.values()),
            "unique_tools": len(self.invoked_tools),
            "skills": len(self.skills_used),
            "log_entries": len(self.work_log),
        }


def _try_json_parse_list(s: str) -> list[str]:
    try:
        result = json.loads(s)
        if isinstance(result, list):
            return [str(item) for item in result]
    except (json.JSONDecodeError, ValueError):
        pass
    return _parse_list(s)


def _try_json_parse_dict(s: str) -> dict[str, str]:
    try:
        result = json.loads(s)
        if isinstance(result, dict):
            return {str(k): str(v) for k, v in result.items()}
    except (json.JSONDecodeError, ValueError):
        pass
    return _parse_dict(s)


def _try_json_parse_counts(s: str) -> dict[str, int]:
    try:
        result = json.loads(s)
        if isinstance(result, dict):
            return {str(k): int(v) for k, v in result.items()}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return _parse_counts(s)


def _parse_list(s: str) -> list[str]:
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if not s:
        return []
    return [item.strip().strip("'\"") for item in s.split(",") if item.strip()]


def _parse_dict(s: str) -> dict[str, str]:
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if not s:
        return {}
    result = {}
    for item in s.split(","):
        if ":" in item:
            k, v = item.split(":", 1)
            result[k.strip().strip("'\"")] = v.strip().strip("'\"")
    return result


def _parse_counts(s: str) -> dict[str, int]:
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if not s:
        return {}
    result = {}
    for item in s.split(","):
        item = item.strip()
        if ":" in item:
            k, v = item.rsplit(":", 1)
            k = k.strip().strip("'\"")
            v = v.strip().rstrip("x").strip()
            try:
                result[k] = int(v)
            except ValueError:
                pass
    return result
