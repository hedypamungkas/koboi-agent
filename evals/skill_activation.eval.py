"""Sample `t` eval: skill activation via the [ACTIVATE_SKILL] marker (R3).

Drives a real SkillRegistry through the mock-with-CONFIG seam. The scripted LLM
emits an ``[ACTIVATE_SKILL: code-review]`` marker; ``AgentCore._activate_skill``
detects it, activates the skill, and records to ``telemetry.snapshot.skills_activated``
(R3 wiring). ``t.activatedSkill`` asserts the activation deterministically -- no
API key. Note the skill name is the hyphenated registry name (``code-review``),
not the directory name (``code_review``).

Run:  koboi eval-test evals/skill_activation.eval.py --mock --strict
"""

from koboi.eval.t import scripted_response

CONFIG = {
    "agent": {
        "name": "skill-activation-eval",
        "description": "Eval probe for skill activation via [ACTIVATE_SKILL] marker",
        "system_prompt": "You are a helpful assistant.",
        "max_iterations": 6,
    },
    "llm": {
        "provider": "openai",
        "model": "gpt-4o-mini",  # required by KoboiConfig even in mock (never contacted)
        "api_key": "dummy",
    },
    "skills": {"search_paths": ["./skills"]},
}

# Attempt 1: the LLM emits the activation marker (_activate_skill detects it,
# activates code-review, records to telemetry, then continues the loop).
# Attempt 2: a clean reply that completes the turn.
MOCK_RESPONSES = [
    scripted_response("[ACTIVATE_SKILL: code-review]\nI'll review the code now."),
    scripted_response("Code review complete. Looks good."),
]
TAGS = ["smoke", "skills"]


async def test_activates_code_review_skill(t):
    """A response containing the marker must activate the named skill (R3).

    ``t.activatedSkill`` reads ``telemetry.snapshot.skills_activated`` (populated by
    ``AgentCore._activate_skill`` via the TelemetryHook).
    """
    await t.send("Please review this code.")
    t.activatedSkill("code-review")
    t.completed()
