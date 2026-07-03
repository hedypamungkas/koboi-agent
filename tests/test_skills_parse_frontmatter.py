"""Tests for parse_frontmatter and skills edge cases."""

from __future__ import annotations


from koboi.skills.registry import (
    parse_frontmatter,
    discover_skills,
    activate_skill,
    load_resource,
    build_discovery_prompt,
    SkillRegistry,
)
from koboi.types import SkillDefinition


class TestParseFrontmatter:
    def test_basic_frontmatter(self):
        content = "---\nname: my-skill\ndescription: A test skill\n---\nBody here"
        result = parse_frontmatter(content)
        assert result["name"] == "my-skill"
        assert result["description"] == "A test skill"

    def test_no_frontmatter(self):
        result = parse_frontmatter("No frontmatter here")
        assert result == {}

    def test_quoted_values(self):
        content = "---\nname: \"quoted name\"\ndescription: 'quoted desc'\n---\n"
        result = parse_frontmatter(content)
        assert result["name"] == "quoted name"
        assert result["description"] == "quoted desc"

    def test_metadata_section(self):
        content = "---\nname: test\nmetadata:\n  author: me\n  version: 1.0\n---\n"
        result = parse_frontmatter(content)
        assert result["metadata"]["author"] == "me"
        assert result["metadata"]["version"] == "1.0"

    def test_allowed_tools(self):
        content = "---\nname: test\nallowed-tools: calculator search web\n---\n"
        result = parse_frontmatter(content)
        assert result["allowed-tools"] == ["calculator", "search", "web"]

    def test_block_scalar(self):
        content = "---\nname: test\ndescription: >\n  A long description\n  that spans lines\n---\n"
        result = parse_frontmatter(content)
        assert "long description" in result["description"]

    def test_comments_ignored(self):
        content = "---\n# This is a comment\nname: test\n---\n"
        result = parse_frontmatter(content)
        assert result.get("name") == "test"

    def test_empty_frontmatter(self):
        content = "---\n\n---\nBody"
        result = parse_frontmatter(content)
        assert result == {}


class TestDiscoverSkills:
    def test_discover_in_directory(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\ndescription: Test\n---\nBody")
        skills = discover_skills([str(tmp_path)])
        assert len(skills) == 1
        assert skills[0].name == "my-skill"

    def test_discover_nonexistent_path(self):
        skills = discover_skills(["/nonexistent/path"])
        assert skills == []

    def test_discover_dedup(self, tmp_path):
        for i in range(2):
            skill_dir = tmp_path / f"skill-{i}"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("---\nname: same-name\ndescription: Same\n---\nBody")
        skills = discover_skills([str(tmp_path)])
        assert len(skills) == 1

    def test_discover_skip_files(self, tmp_path):
        (tmp_path / "not-a-dir.txt").write_text("not a skill")
        skills = discover_skills([str(tmp_path)])
        assert skills == []

    def test_discover_skip_no_name(self, tmp_path):
        skill_dir = tmp_path / "bad-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\ndescription: No name\n---\nBody")
        skills = discover_skills([str(tmp_path)])
        assert skills == []

    def test_discover_recursive(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "skill"
        nested.mkdir(parents=True)
        (nested / "SKILL.md").write_text("---\nname: deep-skill\ndescription: Deep\n---\nBody")
        skills = discover_skills([str(tmp_path)], recursive=True)
        assert len(skills) == 1

    def test_discover_recursive_skip_node_modules(self, tmp_path):
        nm = tmp_path / "node_modules" / "pkg" / "skill"
        nm.mkdir(parents=True)
        (nm / "SKILL.md").write_text("---\nname: nm-skill\ndescription: NM\n---\nBody")
        skills = discover_skills([str(tmp_path)], recursive=True)
        assert len(skills) == 0


class TestActivateSkill:
    def test_activate(self, tmp_path):
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: test\ndescription: Test\n---\nSkill body here")
        skill = SkillDefinition(name="test", description="Test", skill_dir=str(skill_dir))
        body = activate_skill(skill)
        assert "Skill body here" in body
        assert skill.body == body


class TestLoadResource:
    def test_load_existing(self, tmp_path):
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        (skill_dir / "data.txt").write_text("resource content")
        skill = SkillDefinition(name="test", description="Test", skill_dir=str(skill_dir))
        result = load_resource(skill, "data.txt")
        assert result == "resource content"

    def test_load_nonexistent(self, tmp_path):
        skill = SkillDefinition(name="test", description="Test", skill_dir=str(tmp_path))
        result = load_resource(skill, "missing.txt")
        assert result is None

    def test_path_traversal_blocked(self, tmp_path):
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        skill = SkillDefinition(name="test", description="Test", skill_dir=str(skill_dir))
        result = load_resource(skill, "../../../etc/passwd")
        assert result is None


class TestBuildDiscoveryPrompt:
    def test_with_skills(self):
        skills = [
            SkillDefinition(name="s1", description="Skill 1", skill_dir="/tmp"),
            SkillDefinition(name="s2", description="Skill 2", skill_dir="/tmp"),
        ]
        prompt = build_discovery_prompt(skills)
        assert "s1" in prompt
        assert "s2" in prompt

    def test_empty(self):
        assert build_discovery_prompt([]) == ""


class TestSkillRegistryEdge:
    def test_discover_all(self, tmp_path, monkeypatch):
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: test\ndescription: Test\n---\nBody")
        registry = SkillRegistry()
        monkeypatch.setattr(SkillRegistry, "PROJECT_SKILLS", [str(tmp_path)])
        monkeypatch.setattr(SkillRegistry, "USER_SKILLS", ["/nonexistent"])
        monkeypatch.setattr(SkillRegistry, "PLUGIN_SKILLS", ["/nonexistent"])
        names = registry.discover_all()
        assert "test" in names

    def test_get_routed_discovery_prompt(self, tmp_path):
        skill_dir = tmp_path / "code-review"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: code-review\ndescription: Review code for bugs\n---\nBody")
        registry = SkillRegistry()
        registry.discover([str(tmp_path)])
        prompt = registry.get_routed_discovery_prompt("review my code")
        assert "code-review" in prompt

    def test_route_stopwords_only(self):
        registry = SkillRegistry()
        result = registry.route("the and or")
        assert result == []

    def test_activate_nonexistent(self):
        registry = SkillRegistry()
        assert registry.activate("nonexistent") is None

    def test_load_resource_nonexistent(self):
        registry = SkillRegistry()
        assert registry.load_resource("nonexistent", "file.txt") is None
