"""koboi/eval/scorers/skill_scorer.py -- Skill-specific evaluation scorers.

SkillTriggerAccuracyScorer: did the expected skill activate during the eval run?

(The SkillRoutingAccuracyScorer and SkillTokenOverheadScorer were removed -- they
had no population path in the t eval surface: routing_accuracy needs the
orchestration router and token_overhead needs SkillRegistry budget introspection,
neither of which the t context surfaces.)
"""

from __future__ import annotations

from koboi.eval.scorers.base import BaseScorer
from koboi.types import EvalCase, EvalScore


class SkillTriggerAccuracyScorer(BaseScorer):
    """Check if the expected skill was activated during the eval run.

    Expects context["skills_activated"] to be a list of skill names
    (populated by TestContext._build_context from telemetry.snapshot.skills_activated,
    which AgentCore._activate_skill records).
    Score: 1.0 if expected skill activated, 0.0 otherwise.
    """

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        expected_skill = case.metadata.get("expected_skill")
        if not expected_skill:
            return EvalScore("skill_trigger_accuracy", 1.0, "No expected skill specified")

        activated = context.get("skills_activated", [])
        if expected_skill in activated:
            return EvalScore("skill_trigger_accuracy", 1.0, f"Expected skill '{expected_skill}' activated")
        elif activated:
            return EvalScore(
                "skill_trigger_accuracy",
                0.0,
                f"Wrong skill activated: {activated}, expected: {expected_skill}",
            )
        else:
            return EvalScore("skill_trigger_accuracy", 0.0, f"No skills activated, expected: {expected_skill}")
