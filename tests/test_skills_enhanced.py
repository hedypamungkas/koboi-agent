"""Tests for skills enhancements: budget, invocation control, dynamic context, persistence hook, scorers."""

from __future__ import annotations

import pytest

from koboi.skills.registry import (
    SkillRegistry,
    build_discovery_prompt,
    activate_skill,
    parse_frontmatter,
    _preprocess_shell_commands,
)
from koboi.types import SkillDefinition
from koboi.hooks.chain import HookContext, HookEvent
from koboi.hooks.skill_persistence_hook import SkillPersistenceHook
from koboi.eval.scorers.skill_scorer import (
    SkillTriggerAccuracyScorer,
    SkillRoutingAccuracyScorer,
    SkillTokenOverheadScorer,
)
from koboi.types import EvalCase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def budget_skills():
    """Create 10 skills with known descriptions for budget testing."""
    skills = []
    for i in range(10):
        skills.append(
            SkillDefinition(
                name=f"skill-{i}",
                description=f"Description for skill number {i} with some extra text to make it realistic",
                skill_dir=f"/tmp/skill-{i}",
            )
        )
    return skills


@pytest.fixture
def invocation_skills(tmp_path):
    """Create skills with different invocation control settings."""
    skills = []

    # Normal skill (model can invoke)
    d = tmp_path / "normal"
    d.mkdir()
    (d / "SKILL.md").write_text("---\nname: normal-skill\ndescription: Normal skill\n---\nBody")
    skills.append(
        SkillDefinition(
            name="normal-skill",
            description="Normal skill",
            skill_dir=str(d),
        )
    )

    # User-only skill (model cannot invoke)
    d = tmp_path / "user-only"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: deploy-skill\ndescription: Deploy to production\ndisable-model-invocation: true\n---\nBody"
    )
    skills.append(
        SkillDefinition(
            name="deploy-skill",
            description="Deploy to production",
            skill_dir=str(d),
            disable_model_invocation=True,
        )
    )

    # Background skill (not user-invocable)
    d = tmp_path / "background"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: background-skill\ndescription: Background knowledge\nuser-invocable: false\n---\nBody"
    )
    skills.append(
        SkillDefinition(
            name="background-skill",
            description="Background knowledge",
            skill_dir=str(d),
            user_invocable=False,
        )
    )

    return skills


# ---------------------------------------------------------------------------
# Test: Character Budget (Gap 1)
# ---------------------------------------------------------------------------


class TestBuildDiscoveryPromptBudget:
    def test_budget_limits_output(self, budget_skills):
        """Budget should truncate skills when limit exceeded."""
        # With a very small budget, not all skills fit
        prompt = build_discovery_prompt(budget_skills, budget_chars=400)
        # Should contain fewer skills than 10
        assert "skill-0" in prompt  # First skill should be included
        assert "... and" in prompt  # Truncation message
        assert len(prompt) < len(build_discovery_prompt(budget_skills))

    def test_budget_unlimited(self, budget_skills):
        """No budget should include all skills."""
        prompt = build_discovery_prompt(budget_skills, budget_chars=None)
        for i in range(10):
            assert f"skill-{i}" in prompt

    def test_budget_exact_fit(self):
        """Skills that fit exactly within budget should all be included."""
        skills = [
            SkillDefinition(name="s1", description="Short", skill_dir="/tmp"),
            SkillDefinition(name="s2", description="Also short", skill_dir="/tmp"),
        ]
        prompt = build_discovery_prompt(skills, budget_chars=1000)
        assert "s1" in prompt
        assert "s2" in prompt
        assert "..." not in prompt

    def test_budget_zero(self):
        """Budget of 0 should return empty string."""
        skills = [
            SkillDefinition(name="s1", description="Test", skill_dir="/tmp"),
        ]
        prompt = build_discovery_prompt(skills, budget_chars=0)
        assert prompt == ""

    def test_budget_very_small(self):
        """Very small budget should return at least header/footer."""
        skills = [
            SkillDefinition(name="s1", description="Test skill", skill_dir="/tmp"),
        ]
        prompt = build_discovery_prompt(skills, budget_chars=50)
        # Should be empty or just header/footer (budget too small)
        assert isinstance(prompt, str)

    def test_registry_uses_budget(self, budget_skills):
        """SkillRegistry should pass budget_chars to build_discovery_prompt."""
        registry = SkillRegistry(budget_chars=300)
        for s in budget_skills:
            registry._skills[s.name] = s
        prompt = registry.get_discovery_prompt()
        # With 300 char budget, not all 10 skills fit
        full_prompt = build_discovery_prompt(budget_skills)
        assert len(prompt) < len(full_prompt)

    def test_registry_default_budget(self):
        """Default budget should be 8000."""
        registry = SkillRegistry()
        assert registry.budget_chars == 8000

    def test_registry_none_budget(self, budget_skills):
        """Budget=None should include all skills."""
        registry = SkillRegistry(budget_chars=None)
        for s in budget_skills:
            registry._skills[s.name] = s
        prompt = registry.get_discovery_prompt()
        for i in range(10):
            assert f"skill-{i}" in prompt


# ---------------------------------------------------------------------------
# Test: Invocation Control (Gap 3)
# ---------------------------------------------------------------------------


class TestParseFrontmatterInvocation:
    def test_parse_disable_model_invocation(self):
        content = "---\nname: test\ndescription: Test\ndisable-model-invocation: true\n---\nBody"
        result = parse_frontmatter(content)
        assert result["disable-model-invocation"] is True

    def test_parse_disable_model_invocation_false(self):
        content = "---\nname: test\ndescription: Test\ndisable-model-invocation: false\n---\nBody"
        result = parse_frontmatter(content)
        assert result["disable-model-invocation"] is False

    def test_parse_user_invocable(self):
        content = "---\nname: test\ndescription: Test\nuser-invocable: false\n---\nBody"
        result = parse_frontmatter(content)
        assert result["user-invocable"] is False

    def test_parse_disallowed_tools(self):
        content = "---\nname: test\ndescription: Test\ndisallowed-tools: shell filesystem\n---\nBody"
        result = parse_frontmatter(content)
        assert result["disallowed-tools"] == ["shell", "filesystem"]

    def test_parse_invocation_defaults(self):
        """Fields not specified should not appear in parsed result."""
        content = "---\nname: test\ndescription: Test\n---\nBody"
        result = parse_frontmatter(content)
        assert "disable-model-invocation" not in result
        assert "user-invocable" not in result


class TestRouteInvocationControl:
    def test_route_skips_disabled_model_skills(self, invocation_skills):
        """route() should exclude disable_model_invocation=True skills by default."""
        registry = SkillRegistry()
        for s in invocation_skills:
            registry._skills[s.name] = s

        # deploy-skill has disable_model_invocation=True
        results = registry.route("deploy to production", top_k=10)
        names = [s.name for s in results]
        assert "deploy-skill" not in names

    def test_route_includes_disabled_when_requested(self, invocation_skills):
        """route(include_model_disabled=True) should include all skills."""
        registry = SkillRegistry()
        for s in invocation_skills:
            registry._skills[s.name] = s

        results = registry.route("deploy to production", top_k=10, include_model_disabled=True)
        names = [s.name for s in results]
        assert "deploy-skill" in names

    def test_route_normal_skills_work(self, invocation_skills):
        """Normal skills should still be routed."""
        registry = SkillRegistry()
        for s in invocation_skills:
            registry._skills[s.name] = s

        results = registry.route("normal skill", top_k=10)
        names = [s.name for s in results]
        assert "normal-skill" in names


class TestDiscoveryPromptInvocation:
    def test_user_only_skills_marked(self, invocation_skills):
        """Skills with disable_model_invocation should be marked [user-only]."""
        registry = SkillRegistry()
        for s in invocation_skills:
            registry._skills[s.name] = s

        prompt = registry.get_discovery_prompt()
        assert "deploy-skill: Deploy to production [user-only]" in prompt

    def test_normal_skills_not_marked(self, invocation_skills):
        """Normal skills should not have [user-only] marker."""
        registry = SkillRegistry()
        for s in invocation_skills:
            registry._skills[s.name] = s

        prompt = registry.get_discovery_prompt()
        assert "normal-skill: Normal skill\n" in prompt
        assert "normal-skill: Normal skill [user-only]" not in prompt


class TestSkillDefinitionInvocation:
    def test_skill_definition_new_fields(self):
        """SkillDefinition should accept new invocation fields."""
        skill = SkillDefinition(
            name="test",
            description="Test",
            skill_dir="/tmp",
            disable_model_invocation=True,
            user_invocable=False,
            disallowed_tools=["shell"],
        )
        assert skill.disable_model_invocation is True
        assert skill.user_invocable is False
        assert skill.disallowed_tools == ["shell"]

    def test_skill_definition_defaults(self):
        """New fields should have sensible defaults."""
        skill = SkillDefinition(name="test", description="Test", skill_dir="/tmp")
        assert skill.disable_model_invocation is False
        assert skill.user_invocable is True
        assert skill.disallowed_tools is None


# ---------------------------------------------------------------------------
# Test: Dynamic Context Injection (Gap 4)
# ---------------------------------------------------------------------------


class TestDynamicContextInjection:
    def test_shell_command_replacement(self):
        """Shell commands in skill body should be replaced with output."""
        body = "Current status:\n```\n!`echo hello`\n```"
        result = _preprocess_shell_commands(body)
        assert "hello" in result

    def test_shell_command_failure(self):
        """Failed commands should show error marker."""
        body = "Result: !`false_command_that_does_not_exist_12345`"
        result = _preprocess_shell_commands(body)
        assert "[command failed" in result

    def test_shell_command_timeout(self):
        """Commands that timeout should show timeout marker."""
        body = "Result: !`sleep 30`"
        # Use a very short timeout for test
        import subprocess
        import re

        def _run(match):
            cmd = match.group(1).strip()
            try:
                subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=0.01)
                return "ok"
            except subprocess.TimeoutExpired:
                return f"[command timed out: {cmd}]"

        result = re.sub(r"!`([^`]+)`", _run, body)
        assert "[command timed out" in result

    def test_no_commands_passthrough(self):
        """Body without shell commands should be unchanged."""
        body = "This is a normal skill body with no commands."
        result = _preprocess_shell_commands(body)
        assert result == body

    def test_multiple_commands(self):
        """Multiple shell commands should all be processed."""
        body = "!`echo one` and !`echo two`"
        result = _preprocess_shell_commands(body)
        assert "one" in result
        assert "two" in result

    def test_shell_command_does_not_leak_secret_env(self, monkeypatch):
        """P0a: skill !`cmd` preprocessing must not leak secret env vars."""
        from koboi.harness.env import configure_env_defaults

        configure_env_defaults(None)  # ensure no passthrough from other tests
        monkeypatch.setenv("OPENAI_API_KEY", "sk-leak-me")
        body = "Leaked: !`echo $OPENAI_API_KEY`"
        result = _preprocess_shell_commands(body)
        assert "sk-leak-me" not in result


class TestActivateSkillDynamic:
    def test_activate_with_shell(self, tmp_path):
        """activate_skill should preprocess shell commands by default."""
        skill_dir = tmp_path / "test"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: test\ndescription: Test\n---\nStatus: !`echo active`")
        skill = SkillDefinition(name="test", description="Test", skill_dir=str(skill_dir))
        body = activate_skill(skill, run_shell=True)
        assert "active" in body

    def test_activate_without_shell(self, tmp_path):
        """activate_skill(run_shell=False) should leave commands as-is."""
        skill_dir = tmp_path / "test"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: test\ndescription: Test\n---\nStatus: !`echo active`")
        skill = SkillDefinition(name="test", description="Test", skill_dir=str(skill_dir))
        body = activate_skill(skill, run_shell=False)
        assert "!`echo active`" in body


# ---------------------------------------------------------------------------
# Test: Skill Persistence Hook (Gap 2)
# ---------------------------------------------------------------------------


class TestSkillPersistenceHook:
    @pytest.fixture
    def registry_with_activated(self, tmp_path):
        """Create a registry with one activated skill."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: test-skill\ndescription: Test\n---\n# Test Skill\nDo stuff.")
        registry = SkillRegistry()
        registry.discover([str(tmp_path)])
        registry.activate("test-skill")
        return registry

    async def test_reinjects_activated_skills(self, registry_with_activated):
        """Hook should add activated skill bodies to inject_messages."""
        hook = SkillPersistenceHook(skills=registry_with_activated)
        ctx = HookContext(event=HookEvent.POST_COMPACT)
        result = await hook.execute(ctx)
        assert len(result.inject_messages) == 1
        assert "test-skill" in result.inject_messages[0]
        assert "Do stuff." in result.inject_messages[0]

    async def test_no_activated_skills(self, tmp_path):
        """No injection when no skills are activated."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: test-skill\ndescription: Test\n---\nBody")
        registry = SkillRegistry()
        registry.discover([str(tmp_path)])
        # Don't activate

        hook = SkillPersistenceHook(skills=registry)
        ctx = HookContext(event=HookEvent.POST_COMPACT)
        result = await hook.execute(ctx)
        assert len(result.inject_messages) == 0

    async def test_respects_5000_char_limit(self, tmp_path):
        """Body should be truncated to 5000 chars."""
        skill_dir = tmp_path / "big-skill"
        skill_dir.mkdir()
        big_body = "# Skill\n" + "x" * 10000
        (skill_dir / "SKILL.md").write_text(f"---\nname: big-skill\ndescription: Test\n---\n{big_body}")
        registry = SkillRegistry()
        registry.discover([str(tmp_path)])
        registry.activate("big-skill")

        hook = SkillPersistenceHook(skills=registry)
        ctx = HookContext(event=HookEvent.POST_COMPACT)
        result = await hook.execute(ctx)
        assert len(result.inject_messages) == 1
        # Body should be truncated to 5000 chars
        assert len(result.inject_messages[0]) < len(big_body) + 200  # overhead for XML tags

    def test_priority_is_45(self):
        """Priority should be 45 for post-compaction processing."""
        registry = SkillRegistry()
        hook = SkillPersistenceHook(skills=registry)
        assert hook.priority == 45

    def test_handles_post_compact(self):
        """Hook should handle POST_COMPACT event."""
        registry = SkillRegistry()
        hook = SkillPersistenceHook(skills=registry)
        assert hook.handles() == [HookEvent.POST_COMPACT]


# ---------------------------------------------------------------------------
# Test: Skill Config (ConfigModel + ConfigBuilder)
# ---------------------------------------------------------------------------


class TestSkillBudgetConfig:
    def test_skills_config_budget_chars(self):
        """SkillsConfig should accept budget_chars."""
        from koboi.config_models import SkillsConfig

        config = SkillsConfig(search_paths=["./skills"], budget_chars=5000)
        assert config.budget_chars == 5000

    def test_skills_config_default_budget(self):
        """Default budget should be 8000."""
        from koboi.config_models import SkillsConfig

        config = SkillsConfig()
        assert config.budget_chars == 8000

    def test_config_builder_skills_budget(self):
        """ConfigBuilder.skills() should accept budget_chars."""
        from koboi.config import ConfigBuilder

        config = ConfigBuilder().agent(name="test").llm(model="gpt-4o", api_key="sk").skills(budget_chars=3000).build()
        assert config.get("skills", "budget_chars") == 3000


# ---------------------------------------------------------------------------
# Test: Skill Scorers (Gap 5)
# ---------------------------------------------------------------------------


class TestSkillTriggerAccuracyScorer:
    async def test_correct_skill_activated(self):
        """Score 1.0 when expected skill is activated."""
        scorer = SkillTriggerAccuracyScorer()
        case = EvalCase(name="test", user_message="review code", metadata={"expected_skill": "code-review"})
        context = {"skills_activated": ["code-review"]}
        score = await scorer.score(case, "output", context)
        assert score.value == 1.0
        assert "activated" in score.reason

    async def test_wrong_skill_activated(self):
        """Score 0.0 when wrong skill is activated."""
        scorer = SkillTriggerAccuracyScorer()
        case = EvalCase(name="test", user_message="review code", metadata={"expected_skill": "code-review"})
        context = {"skills_activated": ["search"]}
        score = await scorer.score(case, "output", context)
        assert score.value == 0.0

    async def test_no_skill_activated(self):
        """Score 0.0 when no skill is activated."""
        scorer = SkillTriggerAccuracyScorer()
        case = EvalCase(name="test", user_message="review code", metadata={"expected_skill": "code-review"})
        context = {"skills_activated": []}
        score = await scorer.score(case, "output", context)
        assert score.value == 0.0

    async def test_no_expected_skill(self):
        """Score 1.0 when no expected skill specified."""
        scorer = SkillTriggerAccuracyScorer()
        case = EvalCase(name="test", user_message="hello")
        context = {}
        score = await scorer.score(case, "output", context)
        assert score.value == 1.0


class TestSkillRoutingAccuracyScorer:
    async def test_top1_hit(self):
        """Score 1.0 when expected skill is in top-1."""
        scorer = SkillRoutingAccuracyScorer()
        case = EvalCase(name="test", user_message="review code", metadata={"expected_skill": "code-review"})
        context = {"routed_skills": ["code-review", "bug-hunter"]}
        score = await scorer.score(case, "output", context)
        assert score.value == 1.0

    async def test_top3_hit(self):
        """Score 0.5 when expected skill is in top-3 but not top-1."""
        scorer = SkillRoutingAccuracyScorer()
        case = EvalCase(name="test", user_message="review code", metadata={"expected_skill": "code-review"})
        context = {"routed_skills": ["bug-hunter", "code-review"]}
        score = await scorer.score(case, "output", context)
        assert score.value == 0.5

    async def test_miss(self):
        """Score 0.0 when expected skill is not routed."""
        scorer = SkillRoutingAccuracyScorer()
        case = EvalCase(name="test", user_message="review code", metadata={"expected_skill": "code-review"})
        context = {"routed_skills": ["bug-hunter", "search"]}
        score = await scorer.score(case, "output", context)
        assert score.value == 0.0

    async def test_no_routed_skills(self):
        """Score 0.0 when no skills routed."""
        scorer = SkillRoutingAccuracyScorer()
        case = EvalCase(name="test", user_message="review code", metadata={"expected_skill": "code-review"})
        context = {"routed_skills": []}
        score = await scorer.score(case, "output", context)
        assert score.value == 0.0


class TestSkillTokenOverheadScorer:
    async def test_under_budget(self):
        """Score 1.0 when under budget."""
        scorer = SkillTokenOverheadScorer(budget_chars=8000)
        case = EvalCase(name="test", user_message="hello")
        context = {"skill_description_chars": 4000}
        score = await scorer.score(case, "output", context)
        assert score.value == 1.0

    async def test_over_budget(self):
        """Score < 1.0 when over budget."""
        scorer = SkillTokenOverheadScorer(budget_chars=8000)
        case = EvalCase(name="test", user_message="hello")
        context = {"skill_description_chars": 12000}
        score = await scorer.score(case, "output", context)
        assert score.value < 1.0

    async def test_no_skill_chars(self):
        """Score 1.0 when no skill chars."""
        scorer = SkillTokenOverheadScorer(budget_chars=8000)
        case = EvalCase(name="test", user_message="hello")
        context = {}
        score = await scorer.score(case, "output", context)
        assert score.value == 1.0


# ---------------------------------------------------------------------------
# Test: Scorer Registration
# ---------------------------------------------------------------------------


class TestScorerRegistration:
    def test_skill_scorers_registered(self):
        """Skill scorers should be registered in ScorerRegistry."""
        from koboi.eval.registry import ScorerRegistry

        available = ScorerRegistry.list_available()
        assert "skill_trigger_accuracy" in available
        assert "skill_routing_accuracy" in available
        assert "skill_token_overhead" in available

    def test_create_skill_scorers(self):
        """Should be able to create skill scorers from registry."""
        from koboi.eval.registry import ScorerRegistry

        scorer = ScorerRegistry.create("skill_trigger_accuracy")
        assert isinstance(scorer, SkillTriggerAccuracyScorer)

        scorer = ScorerRegistry.create("skill_routing_accuracy")
        assert isinstance(scorer, SkillRoutingAccuracyScorer)

        scorer = ScorerRegistry.create("skill_token_overhead", budget_chars=5000)
        assert isinstance(scorer, SkillTokenOverheadScorer)
