"""koboi/harness/telemetry.py -- Harness health metrics and before/after comparison.

Collects metrics during agent sessions to measure harness effectiveness:
context efficiency, loop health, permission friction, compaction fidelity, etc.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class IterationRecord:
    iteration: int
    had_tool_calls: bool
    tool_names: list[str]
    tokens_before: int
    duration_seconds: float
    was_productive: bool


@dataclass
class CompactionRecord:
    iteration: int
    messages_before: int
    messages_after: int
    tokens_before: int
    tokens_after: int


@dataclass
class PermissionRecord:
    tool_name: str
    action: str  # "allowed" | "confirmed" | "denied"
    rule_name: str | None


@dataclass
class TelemetrySnapshot:
    session_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    total_iterations: int = 0
    iterations: list[IterationRecord] = field(default_factory=list)
    compactions: list[CompactionRecord] = field(default_factory=list)
    permissions: list[PermissionRecord] = field(default_factory=list)
    doom_loops_detected: int = 0
    total_tool_calls: int = 0
    unique_tools_used: set[str] = field(default_factory=set)
    skills_activated: list[str] = field(default_factory=list)
    tokens_consumed_total: int = 0
    carryover_updates: int = 0
    tools_succeeded: int = 0
    tools_failed: int = 0


_DEFAULT_HEALTH_WEIGHTS = {
    "loop_health": 0.20,
    "tool_success_rate": 0.20,
    "context_efficiency": 0.15,
    "compaction_fidelity": 0.15,
    "permission_friction": 0.15,
    "doom_penalty": 0.15,
}


class TelemetryCollector:
    def __init__(self, session_id: str = "", health_weights: dict[str, float] | None = None):
        self.snapshot = TelemetrySnapshot(session_id=session_id)
        self._health_weights = health_weights or _DEFAULT_HEALTH_WEIGHTS
        self._iteration_start: float = 0.0
        self._tokens_at_iteration_start: int = 0

    def session_start(self) -> None:
        self.snapshot.start_time = time.time()

    def session_end(self) -> None:
        self.snapshot.end_time = time.time()

    def iteration_start(self, tokens_current: int = 0) -> None:
        self._iteration_start = time.time()
        self._tokens_at_iteration_start = tokens_current

    def iteration_end(self, iteration: int, tool_names: list[str] | None = None,
                      tokens_after: int = 0, was_productive: bool = True) -> None:
        duration = time.time() - self._iteration_start if self._iteration_start else 0
        record = IterationRecord(
            iteration=iteration,
            had_tool_calls=bool(tool_names),
            tool_names=tool_names or [],
            tokens_before=self._tokens_at_iteration_start,
            duration_seconds=duration,
            was_productive=was_productive,
        )
        self.snapshot.iterations.append(record)
        self.snapshot.total_iterations += 1
        self.snapshot.tokens_consumed_total += max(0, tokens_after - self._tokens_at_iteration_start)

    def record_compaction(self, iteration: int, messages_before: int,
                          messages_after: int, tokens_before: int,
                          tokens_after: int) -> None:
        self.snapshot.compactions.append(CompactionRecord(
            iteration=iteration,
            messages_before=messages_before,
            messages_after=messages_after,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        ))

    def record_permission(self, tool_name: str, action: str,
                          rule_name: str | None = None) -> None:
        self.snapshot.permissions.append(PermissionRecord(
            tool_name=tool_name,
            action=action,
            rule_name=rule_name,
        ))

    def record_doom_loop(self) -> None:
        self.snapshot.doom_loops_detected += 1

    def record_tool_call(self, tool_name: str) -> None:
        self.snapshot.total_tool_calls += 1
        self.snapshot.unique_tools_used.add(tool_name)

    def record_tool_success(self) -> None:
        self.snapshot.tools_succeeded += 1

    def record_tool_failure(self) -> None:
        self.snapshot.tools_failed += 1

    def record_skill_activation(self, skill_name: str) -> None:
        self.snapshot.skills_activated.append(skill_name)

    def record_carryover_update(self) -> None:
        self.snapshot.carryover_updates += 1

    # --- Computed metrics ---

    def context_efficiency(self) -> float:
        if not self.snapshot.iterations:
            return 1.0
        total_tokens = sum(
            rec.tokens_before for rec in self.snapshot.iterations
        )
        if total_tokens == 0:
            return 1.0
        productive_tokens = sum(
            rec.tokens_before for rec in self.snapshot.iterations if rec.was_productive
        )
        return min(1.0, productive_tokens / total_tokens)

    def tool_utilization(self) -> dict[str, float]:
        if self.snapshot.total_tool_calls == 0:
            return {}
        return {
            tool: count / self.snapshot.total_tool_calls
            for tool, count in {
                tool: sum(1 for rec in self.snapshot.iterations if tool in rec.tool_names)
                for tool in self.snapshot.unique_tools_used
            }.items()
        }

    def loop_health(self) -> float:
        if not self.snapshot.iterations:
            return 1.0
        productive = sum(1 for rec in self.snapshot.iterations if rec.was_productive)
        return productive / len(self.snapshot.iterations)

    def tool_success_rate(self) -> float:
        total = self.snapshot.tools_succeeded + self.snapshot.tools_failed
        if total == 0:
            return 1.0
        return self.snapshot.tools_succeeded / total

    def permission_friction(self) -> float:
        if not self.snapshot.permissions:
            return 1.0
        auto_approved = sum(1 for p in self.snapshot.permissions if p.action == "allowed")
        return auto_approved / len(self.snapshot.permissions)

    def compaction_fidelity(self) -> float:
        if not self.snapshot.compactions:
            return 1.0
        ratios = []
        for rec in self.snapshot.compactions:
            if rec.tokens_before > 0:
                ratios.append(rec.tokens_after / rec.tokens_before)
        return sum(ratios) / len(ratios) if ratios else 1.0

    def health_score(self) -> float:
        if not self.snapshot.iterations:
            return 100.0
        doom_penalty = 1.0
        if self.snapshot.total_iterations > 0:
            doom_penalty = max(0.0, 1.0 - self.snapshot.doom_loops_detected / self.snapshot.total_iterations)
        w = self._health_weights
        score = (
            w.get("loop_health", 0.20) * self.loop_health()
            + w.get("tool_success_rate", 0.20) * self.tool_success_rate()
            + w.get("context_efficiency", 0.15) * self.context_efficiency()
            + w.get("compaction_fidelity", 0.15) * self.compaction_fidelity()
            + w.get("permission_friction", 0.15) * self.permission_friction()
            + w.get("doom_penalty", 0.15) * doom_penalty
        )
        return round(score * 100, 1)

    def report(self) -> dict:
        duration = 0.0
        if self.snapshot.start_time and self.snapshot.end_time:
            duration = self.snapshot.end_time - self.snapshot.start_time
        return {
            "session_id": self.snapshot.session_id,
            "duration_seconds": round(duration, 2),
            "total_iterations": self.snapshot.total_iterations,
            "total_tool_calls": self.snapshot.total_tool_calls,
            "unique_tools": len(self.snapshot.unique_tools_used),
            "doom_loops": self.snapshot.doom_loops_detected,
            "skills_activated": self.snapshot.skills_activated,
            "compaction_events": len(self.snapshot.compactions),
            "carryover_updates": self.snapshot.carryover_updates,
            "metrics": {
                "health_score": self.health_score(),
                "loop_health": round(self.loop_health(), 3),
                "iteration_efficiency": round(self.tool_success_rate(), 3),
                "context_efficiency": round(self.context_efficiency(), 3),
                "compaction_fidelity": round(self.compaction_fidelity(), 3),
                "permission_friction": round(self.permission_friction(), 3),
            },
            "permissions": {
                "total": len(self.snapshot.permissions),
                "allowed": sum(1 for p in self.snapshot.permissions if p.action == "allowed"),
                "confirmed": sum(1 for p in self.snapshot.permissions if p.action == "confirmed"),
                "denied": sum(1 for p in self.snapshot.permissions if p.action == "denied"),
            },
        }

    def summary(self) -> str:
        r = self.report()
        lines = [
            f"=== Harness Telemetry: {r['session_id'] or 'session'} ===",
            f"Duration: {r['duration_seconds']}s | Iterations: {r['total_iterations']}",
            f"Tool calls: {r['total_tool_calls']} ({r['unique_tools']} unique)",
            f"Health Score: {r['metrics']['health_score']}/100",
            f"  Loop Health:     {r['metrics']['loop_health']}",
            f"  Tool Success:    {r['metrics']['iteration_efficiency']}",
            f"  Context Eff:     {r['metrics']['context_efficiency']}",
            f"  Compact Fidelity: {r['metrics']['compaction_fidelity']}",
            f"  Perm Friction:   {r['metrics']['permission_friction']}",
            f"Doom loops: {r['doom_loops']} | Compactions: {r['compaction_events']}",
        ]
        if r["permissions"]["total"] > 0:
            p = r["permissions"]
            lines.append(f"Permissions: {p['allowed']} allow, {p['confirmed']} confirm, {p['denied']} deny")
        return "\n".join(lines)
