"""Tests for TUI widgets -- slash_suggester, file_suggester (no Textual instantiation)."""
from __future__ import annotations

import pytest

from koboi.tui.widgets.file_suggester import FileSuggester, CompositeSuggester
from koboi.tui.widgets.slash_suggester import SlashSuggester


class TestSlashSuggester:
    @pytest.mark.asyncio
    async def test_suggest_slash(self):
        s = SlashSuggester(["/help", "/history", "/reset"])
        result = await s.get_suggestion("/he")
        assert result == "/help"

    @pytest.mark.asyncio
    async def test_no_suggestion_no_slash(self):
        s = SlashSuggester(["/help"])
        result = await s.get_suggestion("hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_exact_match_no_suggestion(self):
        s = SlashSuggester(["/help"])
        result = await s.get_suggestion("/help")
        assert result is None

    @pytest.mark.asyncio
    async def test_no_match(self):
        s = SlashSuggester(["/help"])
        result = await s.get_suggestion("/zzz")
        assert result is None


class TestFileSuggester:
    @pytest.mark.asyncio
    async def test_no_at_symbol(self, tmp_path):
        s = FileSuggester(str(tmp_path))
        result = await s.get_suggestion("hello world")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_after_at(self, tmp_path):
        s = FileSuggester(str(tmp_path))
        result = await s.get_suggestion("hello @")
        assert result is None

    @pytest.mark.asyncio
    async def test_suggest_file(self, tmp_path):
        (tmp_path / "test_file.py").write_text("")
        s = FileSuggester(str(tmp_path))
        result = await s.get_suggestion("@test")
        assert result is not None
        assert "test_file.py" in result

    @pytest.mark.asyncio
    async def test_suggest_directory(self, tmp_path):
        (tmp_path / "subdir").mkdir()
        s = FileSuggester(str(tmp_path))
        result = await s.get_suggestion("@sub")
        assert result is not None
        assert "subdir/" in result

    @pytest.mark.asyncio
    async def test_nonexistent_dir(self, tmp_path):
        s = FileSuggester(str(tmp_path))
        result = await s.get_suggestion("@nonexistent_dir/file")
        assert result is None


class TestCompositeSuggester:
    @pytest.mark.asyncio
    async def test_delegates_to_slash(self):
        slash = SlashSuggester(["/help", "/history"])
        file_s = FileSuggester(".")
        composite = CompositeSuggester(slash, file_s)
        result = await composite.get_suggestion("/he")
        assert result == "/help"

    @pytest.mark.asyncio
    async def test_delegates_to_file(self, tmp_path):
        (tmp_path / "test.txt").write_text("")
        slash = SlashSuggester(["/help"])
        file_s = FileSuggester(str(tmp_path))
        composite = CompositeSuggester(slash, file_s)
        result = await composite.get_suggestion(f"@{tmp_path}/test")

    @pytest.mark.asyncio
    async def test_no_match(self):
        slash = SlashSuggester(["/help"])
        file_s = FileSuggester(".")
        composite = CompositeSuggester(slash, file_s)
        result = await composite.get_suggestion("just plain text")
        assert result is None
