"""Example 29: Skills Enhanced Features.

Demonstrates all new skills enhancements:
1. Character budget control (truncation when over budget)
2. Invocation control (user-only vs model-invocable skills)
3. Dynamic context injection (!`command` preprocessing)
4. Skill persistence after compaction
5. Skill evaluation scorers

Run:
    python examples/29_skills_enhanced.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from koboi.skills.registry import SkillRegistry, build_discovery_prompt
from koboi.types import SkillDefinition
from koboi.eval.scorers.skill_scorer import SkillTriggerAccuracyScorer


def demo_budget_control():
    """Demo 1: Character budget for discovery prompt."""
    print("=" * 60)
    print("DEMO 1: Character Budget Control")
    print("=" * 60)

    # Create skills with known descriptions
    skills = []
    for i in range(20):
        skills.append(
            SkillDefinition(
                name=f"skill-{i:02d}",
                description=f"This is skill number {i} with a description that explains what it does in detail",
                skill_dir=f"/tmp/skill-{i}",
            )
        )

    # No budget (original behavior)
    prompt_unlimited = build_discovery_prompt(skills, budget_chars=None)
    print(f"\n  No budget: {len(prompt_unlimited)} chars, {len(skills)} skills")

    # With budget (new behavior)
    prompt_budgeted = build_discovery_prompt(skills, budget_chars=1000)
    print(f"  1000-char budget: {len(prompt_budgeted)} chars")
    if "..." in prompt_budgeted:
        print("  → Skills truncated (budget exceeded)")
    else:
        print("  → All skills fit within budget")

    # With very small budget
    prompt_small = build_discovery_prompt(skills, budget_chars=500)
    print(f"  500-char budget: {len(prompt_small)} chars")

    # Show the actual output
    print("\n  Full output (1000-char budget):")
    for line in prompt_budgeted.split("\n"):
        print(f"    {line}")


def demo_invocation_control():
    """Demo 2: Invocation control frontmatter."""
    print("\n" + "=" * 60)
    print("DEMO 2: Invocation Control")
    print("=" * 60)

    # Create skills with different invocation settings
    normal = SkillDefinition(
        name="code-review",
        description="Review code for bugs and quality",
        skill_dir="/tmp/normal",
    )

    user_only = SkillDefinition(
        name="deploy-production",
        description="Deploy to production environment",
        skill_dir="/tmp/deploy",
        disable_model_invocation=True,
    )

    print("\n  Skills:")
    print(f"    {normal.name}: model_invoke={not normal.disable_model_invocation}, user_invoke={normal.user_invocable}")
    print(
        f"    {user_only.name}: model_invoke={not user_only.disable_model_invocation}, user_invoke={user_only.user_invocable}"
    )

    # Build discovery prompt
    prompt = build_discovery_prompt([normal, user_only])
    print("\n  Discovery prompt:")
    for line in prompt.split("\n"):
        if line.strip():
            print(f"    {line}")

    # Routing behavior
    registry = SkillRegistry()
    registry._skills[normal.name] = normal
    registry._skills[user_only.name] = user_only

    # Default routing (excludes model-disabled skills)
    results = registry.route("deploy to production", top_k=5)
    print(f"\n  Route 'deploy to production' (default): {[s.name for s in results]}")

    # Include model-disabled skills
    results = registry.route("deploy to production", top_k=5, include_model_disabled=True)
    print(f"  Route 'deploy to production' (include_model_disabled): {[s.name for s in results]}")


def demo_dynamic_context():
    """Demo 3: Dynamic context injection."""
    print("\n" + "=" * 60)
    print("DEMO 3: Dynamic Context Injection")
    print("=" * 60)

    # Show how `!`command`` works
    print("\n  Template: !`echo Current date: $(date +%Y-%m-%d)`")
    print("  → Gets replaced with command output at activation time")

    # Demonstrate with a real command
    from koboi.skills.registry import _preprocess_shell_commands

    body = "System status: !`echo all-systems-go`"
    result = _preprocess_shell_commands(body)
    print(f"\n  Input:  {body}")
    print(f"  Output: {result}")

    # Show failure handling
    body_fail = "Result: !`this_command_does_not_exist`"
    result_fail = _preprocess_shell_commands(body_fail)
    print(f"\n  Input:  {body_fail}")
    print(f"  Output: {result_fail}")


def demo_scorers():
    """Demo 4: Skill evaluation scorers."""
    print("\n" + "=" * 60)
    print("DEMO 4: Skill Evaluation Scorers")
    print("=" * 60)

    from koboi.types import EvalCase

    # Trigger accuracy
    scorer = SkillTriggerAccuracyScorer()
    case = EvalCase(name="test", user_message="review code", metadata={"expected_skill": "code-review"})

    import asyncio

    # Correct activation
    score = asyncio.run(scorer.score(case, "output", {"skills_activated": ["code-review"]}))
    print(f"\n  Trigger accuracy (correct):   {score.value:.1f} — {score.reason}")

    # Wrong activation
    score = asyncio.run(scorer.score(case, "output", {"skills_activated": ["search"]}))
    print(f"  Trigger accuracy (wrong):     {score.value:.1f} — {score.reason}")

    # (Routing-accuracy and token-overhead demos removed in R3: those scorers were
    # deleted -- no population path in the t eval surface.)


def demo_persistence_hook():
    """Demo 5: Skill persistence after compaction."""
    print("\n" + "=" * 60)
    print("DEMO 5: Skill Persistence After Compaction")
    print("=" * 60)

    print("\n  SkillPersistenceHook:")
    print("    - Listens for POST_COMPACT events")
    print("    - Re-injects activated skill bodies into context")
    print("    - Truncates to 5000 chars per skill")
    print("    - Priority 45 (runs after security hooks)")
    print("\n  This ensures skills survive context truncation/summarization")
    print("  in long conversations.")


def main():
    print("Skills Enhanced Features Demo")
    print("=" * 60)
    print("This demo showcases all 5 enhancements from the skills")
    print("architecture research recommendations.\n")

    demo_budget_control()
    demo_invocation_control()
    demo_dynamic_context()
    demo_scorers()
    demo_persistence_hook()

    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
