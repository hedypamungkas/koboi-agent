"""koboi/orchestration/planner.py -- LLM workflow planner with self-triage.

Given a user instruction, a SINGLE LLM call (via response_format) both decides
whether the request needs a multi-step workflow AND, if so, extracts the workflow
graph (steps + dependencies). Simple/one-shot requests skip the workflow entirely
(``needs_workflow=False``) -- the planner is not applied to all tasks. Resurrects the
dead ``estimate_complexity`` concept as a proper plan-or-skip gate.

Fail-safe: any parse/cycle/empty failure returns ``needs_workflow=False`` so the
caller falls back to a direct answer -- a bad plan never crashes execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from koboi.client import Client

logger = logging.getLogger(__name__)


@dataclass
class PlanStep:
    id: str
    instruction: str
    depends_on: list[str] = field(default_factory=list)


@dataclass
class PlanResult:
    needs_workflow: bool
    reason: str = ""
    steps: list[PlanStep] = field(default_factory=list)

    @property
    def deps(self) -> dict[str, list[str]]:
        return {s.id: list(s.depends_on) for s in self.steps}


# response_format schema: the LLM self-triages (needs_workflow) + plans (steps).
PLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "needs_workflow": {
            "type": "boolean",
            "description": "true if the request genuinely needs a multi-step workflow; "
            "false if it can be answered directly in one step",
        },
        "reason": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "short snake_case step id"},
                    "instruction": {"type": "string", "description": "one-sentence instruction for this step"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "instruction", "depends_on"],
            },
        },
    },
    "required": ["needs_workflow", "reason", "steps"],
}

_PLANNER_PROMPT = """You are a workflow planner. Decide whether the user's request needs a multi-step \
workflow, and if so, decompose it into an ordered graph of steps.

- If the request is simple (one-shot answer, single fact, greeting, calculation, yes/no), \
set needs_workflow=false and leave steps empty.
- If the request is genuinely multi-step (research then transform, multi-part synthesis, \
sequential dependencies, parallel branches), set needs_workflow=true and list the steps. \
Each step needs: a short snake_case id, a one-sentence instruction, and depends_on (the \
ids of steps that must fully complete before this one). Model parallel branches by giving \
them a shared dependency.

Request: {instruction}"""


def _has_cycle(deps: dict[str, list[str]]) -> bool:
    """DFS cycle detection over the deps graph."""
    white, gray, black = 0, 1, 2
    color = {n: white for n in deps}

    def visit(node: str) -> bool:
        color[node] = gray
        for dep in deps.get(node, []):
            if dep not in color:
                continue
            if color[dep] == gray:
                return True
            if color[dep] == white and visit(dep):
                return True
        color[node] = black
        return False

    return any(color[n] == white and visit(n) for n in deps)


async def plan_or_skip(client: Client, instruction: str, max_steps: int = 12) -> PlanResult:
    """Plan a workflow for ``instruction``, or signal it should be answered directly.

    One LLM call (``response_format=PLAN_SCHEMA``). Returns ``needs_workflow=False``
    for simple requests, empty/malformed plans, or cyclic plans -- the caller then
    answers directly. ``max_steps`` caps plan size (truncates, keeps dependency order).
    """
    from koboi.orchestration._utils import extract_json

    prompt = _PLANNER_PROMPT.format(instruction=instruction)
    try:
        resp = await client.complete(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            response_format=PLAN_SCHEMA,
        )
        data = extract_json(resp.content or "")
    except Exception as e:  # noqa: BLE001 - planner is a boundary: any failure -> direct
        logger.warning("planner call failed for '%s': %s", instruction[:50], e)
        return PlanResult(needs_workflow=False, reason=f"planner error: {type(e).__name__}")

    if not isinstance(data, dict):
        return PlanResult(needs_workflow=False, reason="malformed plan (not an object)")

    needs = bool(data.get("needs_workflow", False))
    raw_steps = data.get("steps") or []
    if not needs or not raw_steps:
        return PlanResult(needs_workflow=False, reason=str(data.get("reason", "simple request")))

    steps: list[PlanStep] = []
    for s in raw_steps:
        if not isinstance(s, dict):
            continue
        sid = str(s.get("id", "")).strip()
        if not sid:
            continue
        steps.append(
            PlanStep(
                id=sid,
                instruction=str(s.get("instruction", "")),
                depends_on=[str(d) for d in s.get("depends_on", [])],
            )
        )
        if len(steps) >= max_steps:
            break

    if not steps:
        return PlanResult(needs_workflow=False, reason="plan had no valid steps")

    if _has_cycle({s.id: s.depends_on for s in steps}):
        return PlanResult(needs_workflow=False, reason="cyclic plan; falling back to direct")

    return PlanResult(needs_workflow=True, reason=str(data.get("reason", "")), steps=steps)
