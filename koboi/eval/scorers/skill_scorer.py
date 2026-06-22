"""koboi/eval/scorers/skill_scorer.py -- Skill-specific evaluation scorers.

Scorers for measuring skill system effectiveness:
- SkillTriggerAccuracyScorer: Did the expected skill activate?
- SkillRoutingAccuracyScorer: Did routing return the expected skill?
- SkillTokenOverheadScorer: Is token overhead within budget?
"""

from __future__ import annotations

from koboi.eval.scorers.base import BaseScorer
from koboi.types import EvalCase, EvalScore


class SkillTriggerAccuracyScorer(BaseScorer):
    """Check if the expected skill was activated during the eval run.

    Expects context["skills_activated"] to be a list of skill names.
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


class SkillRoutingAccuracyScorer(BaseScorer):
    """Check if routing returned the expected skill in top-k results.

    Expects context["routed_skills"] to be a list of skill names (ordered by relevance).
    Score: 1.0 if in top-1, 0.5 if in top-3, 0.0 if missing.
    """

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        expected_skill = case.metadata.get("expected_skill")
        if not expected_skill:
            return EvalScore("skill_routing_accuracy", 1.0, "No expected skill specified")

        routed = context.get("routed_skills", [])
        if not routed:
            return EvalScore("skill_routing_accuracy", 0.0, "No skills routed")

        if routed[0] == expected_skill:
            return EvalScore("skill_routing_accuracy", 1.0, f"'{expected_skill}' in top-1")
        elif expected_skill in routed[:3]:
            return EvalScore("skill_routing_accuracy", 0.5, f"'{expected_skill}' in top-3 (not top-1)")
        elif expected_skill in routed:
            pos = routed.index(expected_skill) + 1
            return EvalScore("skill_routing_accuracy", 0.3, f"'{expected_skill}' at position {pos}")
        else:
            return EvalScore(
                "skill_routing_accuracy",
                0.0,
                f"'{expected_skill}' not in routed skills: {routed}",
            )


class SkillTokenOverheadScorer(BaseScorer):
    """Measure token overhead of skill descriptions in context.

    Expects context["skill_description_chars"] to be the char count.
    Score: 1.0 if under budget, scales down linearly.
    """

    def __init__(self, budget_chars: int = 8000):
        self.budget_chars = budget_chars

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore:
        chars = context.get("skill_description_chars", 0)
        if chars == 0:
            return EvalScore("skill_token_overhead", 1.0, "No skill descriptions in context")

        ratio = chars / self.budget_chars
        if ratio <= 1.0:
            score = 1.0
            reason = f"{chars} chars within budget ({self.budget_chars})"
        else:
            score = max(0.0, 1.0 - (ratio - 1.0))
            reason = f"{chars} chars exceeds budget ({self.budget_chars}) by {ratio:.1%}"

        return EvalScore("skill_token_overhead", round(score, 3), reason)
