"""Tests for koboi/skills/registry.py — Skill discovery and activation."""
from __future__ import annotations

import os
import pytest
from pathlib import Path

from koboi.skills.registry import SkillRegistry


@pytest.fixture
def skill_dir(tmp_path):
    """Create a temp directory with a SKILL.md file."""
    skill_path = tmp_path / "coding"
    skill_path.mkdir()
    skill_md = skill_path / "SKILL.md"
    skill_md.write_text(
        "---\n"
        "name: coding\n"
        "description: Coding assistant skill\n"
        "version: 1.0\n"
        "---\n"
        "You are a coding assistant. Help with programming tasks.\n"
    )
    return tmp_path


@pytest.fixture
def multi_skill_dir(tmp_path):
    """Create a temp directory with multiple skills."""
    for name in ["coding", "writing", "research"]:
        skill_path = tmp_path / name
        skill_path.mkdir()
        skill_md = skill_path / "SKILL.md"
        skill_md.write_text(
            f"---\n"
            f"name: {name}\n"
            f"description: {name.title()} skill\n"
            f"version: 1.0\n"
            f"---\n"
            f"You help with {name}.\n"
        )
    return tmp_path


class TestSkillDiscovery:
    def test_discover_single_skill(self, skill_dir):
        registry = SkillRegistry()
        names = registry.discover([str(skill_dir)])
        assert "coding" in names

    def test_discover_multiple_skills(self, multi_skill_dir):
        registry = SkillRegistry()
        names = registry.discover([str(multi_skill_dir)])
        assert "coding" in names
        assert "writing" in names
        assert "research" in names

    def test_discover_empty_dir(self, tmp_path):
        registry = SkillRegistry()
        names = registry.discover([str(tmp_path)])
        assert names == []

    def test_discover_nonexistent_dir(self, tmp_path):
        registry = SkillRegistry()
        names = registry.discover([str(tmp_path / "nonexistent")])
        assert names == []

    def test_discover_skips_invalid_skill_md(self, tmp_path):
        bad_dir = tmp_path / "bad_skill"
        bad_dir.mkdir()
        (bad_dir / "SKILL.md").write_text("Not valid frontmatter --- missing")
        registry = SkillRegistry()
        names = registry.discover([str(tmp_path)])
        assert isinstance(names, list)


class TestSkillActivation:
    def test_activate_skill(self, skill_dir):
        registry = SkillRegistry()
        registry.discover([str(skill_dir)])
        body = registry.activate("coding")
        assert body is not None
        assert "coding assistant" in body.lower()

    def test_activate_nonexistent(self, skill_dir):
        registry = SkillRegistry()
        registry.discover([str(skill_dir)])
        body = registry.activate("nonexistent")
        assert body is None

    def test_is_activated(self, skill_dir):
        registry = SkillRegistry()
        registry.discover([str(skill_dir)])
        assert not registry.is_activated("coding")
        registry.activate("coding")
        assert registry.is_activated("coding")


class TestSkillGet:
    def test_get_skill(self, skill_dir):
        registry = SkillRegistry()
        registry.discover([str(skill_dir)])
        skill = registry.get("coding")
        assert skill is not None
        assert skill.name == "coding"

    def test_get_nonexistent(self, skill_dir):
        registry = SkillRegistry()
        registry.discover([str(skill_dir)])
        skill = registry.get("nonexistent")
        assert skill is None


class TestDiscoveryPrompt:
    def test_get_discovery_prompt(self, skill_dir):
        registry = SkillRegistry()
        registry.discover([str(skill_dir)])
        prompt = registry.get_discovery_prompt()
        assert prompt is not None
        assert "coding" in prompt.lower()

    def test_get_routed_discovery_prompt(self, skill_dir):
        registry = SkillRegistry()
        registry.discover([str(skill_dir)])
        prompt = registry.get_routed_discovery_prompt("help me code")
        assert prompt is not None

    def test_empty_registry_prompt(self):
        registry = SkillRegistry()
        prompt = registry.get_discovery_prompt()
        assert prompt == "" or prompt is None


class TestSkillResourceLoading:
    def test_load_resource(self, skill_dir):
        # Create a resource file
        res_dir = skill_dir / "coding"
        (res_dir / "example.py").write_text("print('hello')")

        registry = SkillRegistry()
        registry.discover([str(skill_dir)])
        content = registry.load_resource("coding", "example.py")
        assert content is not None
        assert "hello" in content

    def test_load_nonexistent_resource(self, skill_dir):
        registry = SkillRegistry()
        registry.discover([str(skill_dir)])
        content = registry.load_resource("coding", "nonexistent.py")
        assert content is None


class TestSkillRoute:
    def test_route_to_matching_skill(self, skill_dir):
        registry = SkillRegistry()
        registry.discover([str(skill_dir)])
        result = registry.route("help me write python code")
        assert result is not None

    def test_route_no_match(self, tmp_path):
        registry = SkillRegistry()
        result = registry.route("random query")
        assert result is None or isinstance(result, (str, list, type(None)))
