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
    # W2: seed web-search queries a research step should run first (deep-research planner).
    search_queries: list[str] = field(default_factory=list)


@dataclass
class PlanResult:
    needs_workflow: bool
    reason: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    # Deep-research only: the request is ambiguous/under-scoped enough that
    # research would likely target the wrong thing -- ask ONE question before
    # planning instead of guessing. Mutually exclusive with needs_workflow=true.
    needs_clarification: bool = False
    clarifying_question: str = ""

    def __post_init__(self) -> None:
        # Enforce the invariants the JSON schema only *describes* (producers -- and
        # tests/REPL -- can't be trusted to obey them). Matches the RoutingDecision /
        # EvalScore __post_init__ convention in koboi/types.py.
        if self.needs_clarification and not self.clarifying_question.strip():
            raise ValueError("PlanResult: clarifying_question is required when needs_clarification=True")
        if self.needs_clarification and self.needs_workflow:
            raise ValueError("PlanResult: needs_clarification is mutually exclusive with needs_workflow=True")
        if self.clarifying_question and not self.needs_clarification:
            raise ValueError("PlanResult: clarifying_question set but needs_clarification=False")

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


# W2: research planner schema -- same shape as PLAN_SCHEMA plus per-step search_queries.
RESEARCH_PLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "needs_workflow": {
            "type": "boolean",
            "description": "true if the request genuinely needs research; false if answerable directly",
        },
        "needs_clarification": {
            "type": "boolean",
            "description": "true if the request is ambiguous/under-scoped enough that research would "
            "likely target the wrong thing (e.g. missing market/region, budget/price segment, "
            "timeframe, or audience) -- ask ONE clarifying question instead of planning research. "
            "Only when genuinely necessary, not for requests that are already clear enough.",
        },
        "clarifying_question": {
            "type": "string",
            "maxLength": 200,
            "description": "ONE short, specific question to ask before planning research. "
            "Required when needs_clarification=true, otherwise empty. Mutually exclusive "
            "with needs_workflow=true.",
        },
        "reason": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "short snake_case step id"},
                    "instruction": {"type": "string", "description": "one-sentence instruction for this research step"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                    "search_queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "2-4 seed web-search queries this step should run first",
                    },
                },
                "required": ["id", "instruction", "depends_on"],
            },
        },
    },
    "required": ["needs_workflow", "reason", "steps"],
}

_RESEARCH_PLANNER_PROMPT = """You are a research planner. Decompose the request into a graph of \
research sub-questions, each with seed web-search queries.

- If the request is simple (one-shot fact, greeting, calculation, yes/no), set \
needs_workflow=false and leave steps empty.
- If the request is ambiguous or under-scoped in a way that would make research target the \
wrong thing (e.g. missing market/region, budget/price segment, timeframe, or audience), set \
needs_clarification=true, leave needs_workflow=false and steps empty, and give ONE short, \
specific clarifying_question. Only do this when genuinely necessary -- do not ask for \
clarification on requests that are already clear enough to research.
- Otherwise set needs_workflow=true and list the research steps. Each step needs: a short \
snake_case id, a one-sentence instruction (the sub-question to investigate), depends_on (ids \
that must complete first; model parallel branches with a shared dependency), and 2-4 \
search_queries (the web searches this step should run first). Aim for 4-7 focused research \
steps that cover the topic from multiple angles (e.g. scientific, commercial, technical, \
competitive). Each step should investigate a distinct aspect.

Request: {instruction}"""


def _has_cycle(deps: dict[str, list[str]]) -> bool:
    """DFS cycle detection over the deps graph."""
    white, gray, black = 0, 1, 2
    color = dict.fromkeys(deps, white)

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


async def plan_research(
    client: Client, instruction: str, max_steps: int = 12, system_prompt: str | None = None
) -> PlanResult:
    """Plan a research workflow for ``instruction``: sub-questions + per-step seed queries.

    Sibling of ``plan_or_skip`` using ``RESEARCH_PLAN_SCHEMA`` / ``_RESEARCH_PLANNER_PROMPT``.
    Same fail-safe discipline (any failure -> ``needs_workflow=False`` -> caller answers
    directly). Each returned step carries ``search_queries`` (seed web searches).

    ``system_prompt`` (the agent's configured persona/tone, when set) is prepended as a
    leading system message -- mirrors ``Orchestrator._synthesize_research``'s wiring so
    tone/language preferences also shape the planner's clarifying_question, not just the
    final report.
    """
    from koboi.orchestration._utils import extract_json

    prompt = _RESEARCH_PLANNER_PROMPT.format(instruction=instruction)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    try:
        resp = await client.complete(
            messages=messages,
            tools=None,
            response_format=RESEARCH_PLAN_SCHEMA,
        )
        data = extract_json(resp.content or "")
    except Exception as e:  # noqa: BLE001 - planner is a boundary: any failure -> direct
        logger.warning("research planner call failed for '%s': %s", instruction[:50], e)
        return PlanResult(needs_workflow=False, reason=f"planner error: {type(e).__name__}")

    if not isinstance(data, dict):
        return PlanResult(needs_workflow=False, reason="malformed research plan (not an object)")

    needs_clarification = bool(data.get("needs_clarification", False))
    cq_raw = data.get("clarifying_question", "")
    if not isinstance(cq_raw, str):
        # Malformed (null/number/object) -- never ship str(value) (e.g. literal "None")
        # as a user-facing question. Treat as absent and fall through.
        logger.warning(
            "research planner returned non-string clarifying_question=%r for '%s'; ignoring",
            cq_raw,
            instruction[:80],
        )
        clarifying_question = ""
    else:
        clarifying_question = cq_raw.strip()
    if needs_clarification and clarifying_question:
        return PlanResult(
            needs_workflow=False,
            reason=str(data.get("reason", "ambiguous request")),
            needs_clarification=True,
            clarifying_question=clarifying_question,
        )
    if needs_clarification:
        # The planner asked to clarify but gave no usable question -- don't silently
        # ship an empty/garbage turn (or answer the query it just flagged ambiguous).
        # Fall through to normal planning; the warn gives operators a trace.
        logger.warning(
            "research planner set needs_clarification=true but gave no usable "
            "clarifying_question for '%s'; falling through to normal planning",
            instruction[:80],
        )

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
                search_queries=[str(q) for q in s.get("search_queries", [])],
            )
        )
        if len(steps) >= max_steps:
            break

    if not steps:
        return PlanResult(needs_workflow=False, reason="research plan had no valid steps")

    if _has_cycle({s.id: s.depends_on for s in steps}):
        return PlanResult(needs_workflow=False, reason="cyclic research plan; falling back to direct")

    return PlanResult(needs_workflow=True, reason=str(data.get("reason", "")), steps=steps)
