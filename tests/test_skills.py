"""Tests for koboi.skills module."""
from __future__ import annotations

import os
from pathlib import Path

from koboi.skills.registry import SkillRegistry, discover_skills
from koboi.types import SkillDefinition


class TestSkillDiscovery:
    def test_discover_skill_md(self, tmp_path):
        skill_dir = tmp_path / "my_skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            "---\nname: my-skill\ndescription: A test skill\n---\n\n# My Skill\n\nDo stuff.\n"
        )
        skills = discover_skills([str(tmp_path)])
        assert len(skills) == 1
        assert skills[0].name == "my-skill"
        assert skills[0].description == "A test skill"

    def test_no_skills(self, tmp_path):
        skills = discover_skills([str(tmp_path)])
        assert len(skills) == 0


class TestSkillRegistry:
    def test_register_and_route(self):
        registry = SkillRegistry()
        skill = SkillDefinition(
            name="search-and-summarize",
            description="Research and summarize topics from web sources",
            skill_dir="/tmp",
        )
        registry._skills[skill.name] = skill
        results = registry.route("search for information about Python", top_k=1)
        assert len(results) == 1
        assert results[0].name == "search-and-summarize"

    def test_activate(self, tmp_path):
        registry = SkillRegistry()
        skill_dir = tmp_path / "test_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: Test\n---\n\n# Body content here\n"
        )
        skill = SkillDefinition(name="test-skill", description="Test", skill_dir=str(skill_dir))
        registry._skills[skill.name] = skill
        body = registry.activate("test-skill")
        assert body is not None
        assert "Body content" in body

    def test_get(self):
        registry = SkillRegistry()
        skill = SkillDefinition(name="x", description="X", skill_dir="/tmp")
        registry._skills["x"] = skill
        assert registry.get("x").name == "x"
        assert registry.get("missing") is None

    def test_is_activated(self, tmp_path):
        registry = SkillRegistry()
        skill_dir = tmp_path / "act_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: act-skill\ndescription: Test\n---\n\nBody\n"
        )
        skill = SkillDefinition(name="act-skill", description="Test", skill_dir=str(skill_dir))
        registry._skills[skill.name] = skill
        assert registry.is_activated("act-skill") is False
        registry.activate("act-skill")
        assert registry.is_activated("act-skill") is True
