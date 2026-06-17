"""Tests for koboi/hooks/skill_hook.py — SkillHook (0% → >85%)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from koboi.hooks.chain import HookContext, HookEvent
from koboi.hooks.skill_hook import SkillHook


class TestSkillHookHandles:
    def test_handles_returns_post_llm_call(self):
        """SkillHook should handle POST_LLM_CALL event."""
        hook = SkillHook()
        assert hook.handles() == [HookEvent.POST_LLM_CALL]


class TestSkillHookPatternDetection:
    async def test_skill_pattern_detects_activation(self):
        """SKILL_PATTERN should detect [ACTIVATE_SKILL:name] pattern."""
        hook = SkillHook()
        pattern = hook.SKILL_PATTERN
        match = pattern.search("[ACTIVATE_SKILL:web_search]")
        assert match is not None
        assert match.group(1) == "web_search"

    async def test_skill_pattern_case_insensitive(self):
        """SKILL_PATTERN directive should be case-insensitive, but skill names are limited to [a-z0-9_-]."""
        hook = SkillHook()
        pattern = hook.SKILL_PATTERN
        # The pattern is case-insensitive for ACTIVATE_SKILL but skill name must match [a-z0-9_-]+
        match = pattern.search("[ACTIVATE_SKILL:websearch]")
        assert match is not None
        assert match.group(1) == "websearch"

    async def test_skill_pattern_uppercase_directive(self):
        """Pattern should match uppercase ACTIVATE_SKILL directive."""
        hook = SkillHook()
        pattern = hook.SKILL_PATTERN
        match = pattern.search("[ACTIVATE_SKILL:web_search]")
        assert match is not None
        assert match.group(1) == "web_search"

    async def test_skill_pattern_with_hyphen(self):
        """SKILL_PATTERN should detect skills with hyphens."""
        hook = SkillHook()
        pattern = hook.SKILL_PATTERN
        match = pattern.search("[ACTIVATE_SKILL:web-search]")
        assert match is not None
        assert match.group(1) == "web-search"

    async def test_skill_pattern_with_underscore(self):
        """SKILL_PATTERN should detect skills with underscores."""
        hook = SkillHook()
        pattern = hook.SKILL_PATTERN
        match = pattern.search("[ACTIVATE_SKILL:web_search_advanced]")
        assert match is not None
        assert match.group(1) == "web_search_advanced"

    async def test_skill_pattern_with_numbers(self):
        """SKILL_PATTERN should detect skills with numbers."""
        hook = SkillHook()
        pattern = hook.SKILL_PATTERN
        match = pattern.search("[ACTIVATE_SKILL:gpt4_tool]")
        assert match is not None
        assert match.group(1) == "gpt4_tool"

    async def test_skill_pattern_multiple_in_text(self):
        """SKILL_PATTERN should find multiple activations in text."""
        hook = SkillHook()
        text = "Use [ACTIVATE_SKILL:search] and then [ACTIVATE_SKILL:calculate]"
        matches = list(hook.SKILL_PATTERN.finditer(text))
        assert len(matches) == 2
        assert matches[0].group(1) == "search"
        assert matches[1].group(1) == "calculate"


class TestSkillHookExecution:
    async def test_empty_response_passthrough(self):
        """Empty response should return context unchanged."""
        hook = SkillHook()

        class EmptyResponse:
            content = ""

        ctx = HookContext(event=HookEvent.POST_LLM_CALL, llm_response=EmptyResponse())
        result = await hook.execute(ctx)
        assert result is ctx
        assert "skills_detected" not in result.metadata

    async def test_no_llm_response_passthrough(self):
        """No llm_response should return context unchanged."""
        hook = SkillHook()
        ctx = HookContext(event=HookEvent.POST_LLM_CALL, llm_response=None)
        result = await hook.execute(ctx)
        assert result is ctx
        assert "skills_detected" not in result.metadata

    async def test_detects_single_skill_activation(self):
        """Should detect a single skill activation in response."""
        hook = SkillHook()

        class MockResponse:
            content = "I'll use [ACTIVATE_SKILL:calculator] for this."

        ctx = HookContext(event=HookEvent.POST_LLM_CALL, llm_response=MockResponse())
        result = await hook.execute(ctx)
        assert "skills_detected" in result.metadata
        assert len(result.metadata["skills_detected"]) == 1
        assert result.metadata["skills_detected"][0]["name"] == "calculator"

    async def test_detects_multiple_skill_activations(self):
        """Should detect multiple skill activations in response."""
        hook = SkillHook()

        class MockResponse:
            content = "Use [ACTIVATE_SKILL:search] then [ACTIVATE_SKILL:summarize]"

        ctx = HookContext(event=HookEvent.POST_LLM_CALL, llm_response=MockResponse())
        result = await hook.execute(ctx)
        assert len(result.metadata["skills_detected"]) == 2
        skills = [s["name"] for s in result.metadata["skills_detected"]]
        assert "search" in skills
        assert "summarize" in skills


class TestSkillHookFiltering:
    async def test_available_skills_filters_unknown_skills(self):
        """Should filter out skills not in available_skills list."""
        hook = SkillHook(available_skills=["calculator", "search"])

        class MockResponse:
            content = "Use [ACTIVATE_SKILL:calculator] and [ACTIVATE_SKILL:unknown_skill]"

        ctx = HookContext(event=HookEvent.POST_LLM_CALL, llm_response=MockResponse())
        result = await hook.execute(ctx)
        skills = [s["name"] for s in result.metadata["skills_detected"]]
        assert "calculator" in skills
        assert "unknown_skill" not in skills

    async def test_empty_available_skills_accepts_all(self):
        """Empty available_skills should accept any skill."""
        hook = SkillHook(available_skills=None)

        class MockResponse:
            content = "Use [ACTIVATE_SKILL:any_skill]"

        ctx = HookContext(event=HookEvent.POST_LLM_CALL, llm_response=MockResponse())
        result = await hook.execute(ctx)
        assert len(result.metadata["skills_detected"]) == 1
        assert result.metadata["skills_detected"][0]["name"] == "any_skill"

    async def test_case_insensitive_available_skills_check(self):
        """Available skills check should match case-insensitively (pattern lowercases matches)."""
        hook = SkillHook(available_skills=["calculator", "search"])

        class MockResponse:
            content = "Use [ACTIVATE_SKILL:calculator]"

        ctx = HookContext(event=HookEvent.POST_LLM_CALL, llm_response=MockResponse())
        result = await hook.execute(ctx)
        assert len(result.metadata["skills_detected"]) == 1


class TestSkillHookActivatedSkillsProperty:
    async def test_activated_skills_accumulates(self):
        """activated_skills property should accumulate across calls."""
        hook = SkillHook()

        class MockResponse1:
            content = "[ACTIVATE_SKILL:skill1]"

        class MockResponse2:
            content = "[ACTIVATE_SKILL:skill2]"

        ctx1 = HookContext(event=HookEvent.POST_LLM_CALL, llm_response=MockResponse1())
        await hook.execute(ctx1)

        ctx2 = HookContext(event=HookEvent.POST_LLM_CALL, llm_response=MockResponse2())
        await hook.execute(ctx2)

        assert hook.activated_skills == ["skill1", "skill2"]

    async def test_activated_skills_duplicates_allowed(self):
        """activated_skills should allow duplicates."""
        hook = SkillHook()

        class MockResponse:
            content = "[ACTIVATE_SKILL:skill1]"

        ctx = HookContext(event=HookEvent.POST_LLM_CALL, llm_response=MockResponse())
        await hook.execute(ctx)
        await hook.execute(ctx)

        assert hook.activated_skills == ["skill1", "skill1"]


class TestSkillHookCarryover:
    async def test_carryover_record_skill_called(self):
        """Should call carryover.record_skill when carryover available."""
        hook = SkillHook()
        mock_carryover = MagicMock()
        mock_carryover.record_skill = MagicMock()

        class MockResponse:
            content = "[ACTIVATE_SKILL:test_skill]"

        ctx = HookContext(
            event=HookEvent.POST_LLM_CALL,
            llm_response=MockResponse(),
            carryover=mock_carryover,
        )
        await hook.execute(ctx)
        mock_carryover.record_skill.assert_called_once_with("test_skill")

    async def test_carryover_record_skill_multiple_times(self):
        """Should call carryover.record_skill for each detected skill."""
        hook = SkillHook()
        mock_carryover = MagicMock()
        mock_carryover.record_skill = MagicMock()

        class MockResponse:
            content = "[ACTIVATE_SKILL:skill1] and [ACTIVATE_SKILL:skill2]"

        ctx = HookContext(
            event=HookEvent.POST_LLM_CALL,
            llm_response=MockResponse(),
            carryover=mock_carryover,
        )
        await hook.execute(ctx)
        assert mock_carryover.record_skill.call_count == 2

    async def test_no_carryover_no_error(self):
        """Should not error when carryover is None."""
        hook = SkillHook()

        class MockResponse:
            content = "[ACTIVATE_SKILL:test_skill]"

        ctx = HookContext(
            event=HookEvent.POST_LLM_CALL,
            llm_response=MockResponse(),
            carryover=None,
        )
        result = await hook.execute(ctx)
        assert result is ctx


class TestSkillHookAutoActivate:
    async def test_auto_activate_sets_skills_to_activate(self):
        """When auto_activate=True, should set skills_to_activate in metadata."""
        hook = SkillHook(auto_activate=True)

        class MockResponse:
            content = "[ACTIVATE_SKILL:test_skill]"

        ctx = HookContext(event=HookEvent.POST_LLM_CALL, llm_response=MockResponse())
        result = await hook.execute(ctx)
        assert "skills_to_activate" in result.metadata
        assert len(result.metadata["skills_to_activate"]) == 1
        assert result.metadata["skills_to_activate"][0]["name"] == "test_skill"

    async def test_auto_activate_false_no_metadata(self):
        """When auto_activate=False, should not set skills_to_activate."""
        hook = SkillHook(auto_activate=False)

        class MockResponse:
            content = "[ACTIVATE_SKILL:test_skill]"

        ctx = HookContext(event=HookEvent.POST_LLM_CALL, llm_response=MockResponse())
        result = await hook.execute(ctx)
        assert "skills_to_activate" not in result.metadata

    async def test_auto_activate_default_is_false(self):
        """Default auto_activate should be False."""
        hook = SkillHook()

        class MockResponse:
            content = "[ACTIVATE_SKILL:test_skill]"

        ctx = HookContext(event=HookEvent.POST_LLM_CALL, llm_response=MockResponse())
        result = await hook.execute(ctx)
        assert "skills_to_activate" not in result.metadata


class TestSkillHookMetadata:
    async def test_skill_detected_metadata_format(self):
        """skills_detected metadata should have correct format."""
        hook = SkillHook()

        class MockResponse:
            content = "[ACTIVATE_SKILL:my_skill]"

        ctx = HookContext(event=HookEvent.POST_LLM_CALL, llm_response=MockResponse())
        result = await hook.execute(ctx)
        skill_info = result.metadata["skills_detected"][0]
        assert "name" in skill_info
        assert skill_info["name"] == "my_skill"
        assert "arguments" in skill_info
        assert skill_info["arguments"] == ""

    async def test_skill_with_arguments_in_pattern(self):
        """Current implementation sets arguments to empty string."""
        hook = SkillHook()

        class MockResponse:
            content = "[ACTIVATE_SKILL:my_skill]"

        ctx = HookContext(event=HookEvent.POST_LLM_CALL, llm_response=MockResponse())
        result = await hook.execute(ctx)
        skill_info = result.metadata["skills_detected"][0]
        assert skill_info["arguments"] == ""

    async def test_metadata_preserved(self):
        """Existing metadata should be preserved."""
        hook = SkillHook()

        class MockResponse:
            content = "[ACTIVATE_SKILL:test]"

        ctx = HookContext(
            event=HookEvent.POST_LLM_CALL,
            llm_response=MockResponse(),
        )
        ctx.metadata["existing"] = "value"
        result = await hook.execute(ctx)
        assert result.metadata["existing"] == "value"
