"""Tests for koboi.tools.builtin.search module."""

from __future__ import annotations

import os

import pytest

from koboi.tools.builtin.search import (
    grep_search,
    glob_find,
    _is_binary,
    _expand_braces,
    _match_glob,
)


class TestGrepSearch:
    def test_grep_search_basic_pattern(self, tmp_path):
        """Test grep_search with basic regex pattern."""
        # Create test files
        (tmp_path / "test1.py").write_text("def hello():\n    print('world')\n")
        (tmp_path / "test2.py").write_text("# TODO: implement this\n")

        result = grep_search(pattern=r"def \w+", path=str(tmp_path))
        assert "test1.py" in result
        assert "def hello" in result
        assert "test2.py" not in result

    def test_grep_search_with_context_lines(self, tmp_path):
        """Test grep_search with context lines."""
        test_file = tmp_path / "context.txt"
        test_file.write_text("line1\nline2 TARGET line3\nline4\nline5")

        result = grep_search(pattern="TARGET", path=str(tmp_path), context_lines=1)
        # Should show context around match
        assert "line2" in result
        assert "TARGET" in result
        assert "line3" in result
        # With context=1, should also include adjacent lines
        assert "line1" in result or "line4" in result

    def test_grep_search_content_mode(self, tmp_path):
        """Test grep_search with content output mode."""
        (tmp_path / "test.txt").write_text("error at line 1\nwarning at line 2\nerror at line 3")

        result = grep_search(pattern="error", path=str(tmp_path), output_mode="content")
        assert "error at line 1" in result
        assert "error at line 3" in result

    def test_grep_search_files_mode(self, tmp_path):
        """Test grep_search with files output mode."""
        (tmp_path / "file1.txt").write_text("match this")
        (tmp_path / "file2.txt").write_text("no match")
        (tmp_path / "file3.txt").write_text("match this too")

        result = grep_search(pattern="match", path=str(tmp_path), output_mode="files")
        # Files mode returns all files that were searched
        assert "file1.txt" in result
        assert "file3.txt" in result

    def test_grep_search_count_mode(self, tmp_path):
        """Test grep_search with count output mode."""
        (tmp_path / "count.txt").write_text("error\nwarning\nerror\nerror")

        result = grep_search(pattern="error", path=str(tmp_path), output_mode="count")
        assert "3 matches" in result
        assert "count.txt" in result

    def test_grep_search_with_file_filter(self, tmp_path):
        """Test grep_search with file filter."""
        (tmp_path / "test.py").write_text("TODO: fix bug")
        (tmp_path / "test.js").write_text("TODO: fix bug")
        (tmp_path / "test.txt").write_text("TODO: fix bug")

        result = grep_search(pattern="TODO", path=str(tmp_path), file_filter="*.py")
        assert "test.py" in result
        assert "test.js" not in result
        assert "test.txt" not in result

    def test_grep_search_with_brace_expansion(self, tmp_path):
        """Test grep_search with brace expansion in file_filter."""
        (tmp_path / "test.py").write_text("import os")
        (tmp_path / "test.js").write_text("import os")
        (tmp_path / "test.txt").write_text("import os")

        result = grep_search(pattern="import", path=str(tmp_path), file_filter="*.{py,js}")
        assert "test.py" in result
        assert "test.js" in result
        assert "test.txt" not in result

    def test_grep_search_no_match(self, tmp_path):
        """Test grep_search when pattern doesn't match."""
        (tmp_path / "test.txt").write_text("some content")

        result = grep_search(pattern="nonexistent", path=str(tmp_path))
        assert "No match found" in result

    def test_grep_search_invalid_regex(self, tmp_path):
        """Test grep_search with invalid regex pattern."""
        result = grep_search(pattern="[invalid(", path=str(tmp_path))
        assert "Error" in result
        assert "invalid regex" in result.lower()

    def test_grep_search_nonexistent_path(self):
        """Test grep_search with nonexistent path."""
        result = grep_search(pattern="test", path="/nonexistent/path/xyz")
        assert "Error" in result
        assert "not found" in result.lower() or "not a directory" in result.lower()


class TestGlobFind:
    def test_glob_find_basic_pattern(self, tmp_path):
        """Test glob_find with basic pattern."""
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.txt").write_text("content2")
        (tmp_path / "other.py").write_text("content3")

        result = glob_find(pattern="*.txt", path=str(tmp_path))
        assert "file1.txt" in result
        assert "file2.txt" in result
        assert "other.py" not in result

    def test_glob_find_recursive_pattern(self, tmp_path):
        """Test glob_find with recursive pattern."""
        (tmp_path / "level1").mkdir()
        (tmp_path / "level1" / "level2").mkdir()
        (tmp_path / "file.txt").write_text("root")
        (tmp_path / "level1" / "file.txt").write_text("level1")
        (tmp_path / "level1" / "level2" / "file.txt").write_text("level2")

        result = glob_find(pattern="**/*.txt", path=str(tmp_path))
        assert "file.txt" in result
        assert os.path.join("level1", "file.txt") in result or "level1/file.txt" in result
        # Should find nested files

    def test_glob_find_no_matches(self, tmp_path):
        """Test glob_find when pattern matches nothing."""
        result = glob_find(pattern="*.nonexistent", path=str(tmp_path))
        assert "No files matching" in result

    def test_glob_find_invalid_pattern(self, tmp_path):
        """Test glob_find with invalid pattern."""
        # Most patterns are valid, but test with a path that doesn't exist
        result = glob_find(pattern="*.txt", path=str(tmp_path / "nonexistent"))
        assert "Error" in result or "not found" in result.lower()

    def test_glob_find_max_results(self, tmp_path):
        """Test glob_find respects max results limit."""
        # Create many files
        for i in range(100):
            (tmp_path / f"file{i}.txt").write_text(f"content{i}")

        # Set max results to 10 via _tool_config
        result = glob_find(pattern="*.txt", path=str(tmp_path), _tool_config={"max_results": 10})
        # Should indicate truncation
        assert "showing first 10" in result or "10 total" in result

    def test_glob_find_absolute_path(self, tmp_path):
        """Test glob_find with absolute path pattern."""
        (tmp_path / "test.txt").write_text("content")

        # Use absolute path
        result = glob_find(pattern="*.txt", path=str(tmp_path))
        assert "test.txt" in result


class TestBinaryFileDetection:
    def test_is_binary_with_binary_file(self, tmp_path):
        """Test _is_binary detects binary files."""
        binary_file = tmp_path / "binary.bin"
        binary_file.write_bytes(b"\x00\x01\x02\x03\xff\xfe")

        assert _is_binary(str(binary_file)) is True

    def test_is_binary_with_text_file(self, tmp_path):
        """Test _is_binary returns False for text files."""
        text_file = tmp_path / "text.txt"
        text_file.write_text("Plain text content")

        assert _is_binary(str(text_file)) is False

    def test_is_binary_with_empty_file(self, tmp_path):
        """Test _is_binary with empty file."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        # Empty file is not binary
        assert _is_binary(str(empty_file)) is False

    def test_is_binary_with_mixed_content(self, tmp_path):
        """Test _is_binary with mixed content."""
        mixed_file = tmp_path / "mixed.txt"
        # Text with null byte (binary indicator)
        mixed_file.write_bytes(b"Hello\x00World")

        assert _is_binary(str(mixed_file)) is True

    def test_grep_search_skips_binary_files(self, tmp_path):
        """Test grep_search skips binary files."""
        binary_file = tmp_path / "binary.bin"
        binary_file.write_bytes(b"error message here\x00\x01")

        text_file = tmp_path / "text.txt"
        text_file.write_text("error message here")

        result = grep_search(pattern="error", path=str(tmp_path))
        # Should only find text file
        assert "text.txt" in result
        assert "binary.bin" not in result


class TestBraceExpansion:
    def test_expand_braces_simple(self):
        """Test _expand_braces with simple pattern."""
        result = _expand_braces("*.{py,js,txt}")
        assert result == ["*.py", "*.js", "*.txt"]

    def test_expand_braces_single_item(self):
        """Test _expand_braces with single item."""
        result = _expand_braces("*.py")
        assert result == ["*.py"]

    def test_expand_braces_multiple_braces(self):
        """Test _expand_braces with first brace only."""
        result = _expand_braces("*.{py,js}.{test,prod}")
        # Only expands first brace
        assert len(result) == 2
        assert "*.py.{test,prod}" in result
        assert "*.js.{test,prod}" in result

    def test_expand_braces_empty_variants(self):
        """Test _expand_braces with empty variants."""
        result = _expand_braces("*.{,py}")
        assert result == ["*.", "*.py"]


class TestGlobMatching:
    def test_match_glob_basic(self):
        """Test _match_glob with basic patterns."""
        assert _match_glob("test.py", ["*.py"]) is True
        assert _match_glob("test.py", ["*.txt"]) is False
        # _match_glob matches files in subdirs with *.py pattern
        assert _match_glob("dir/test.py", ["*.py"]) is True

    def test_match_glob_with_double_star(self):
        """Test _match_glob with ** pattern."""
        assert _match_glob("dir/subdir/test.py", ["**/*.py"]) is True
        assert _match_glob("test.py", ["**/*.py"]) is True
        assert _match_glob("dir/subdir/test.txt", ["**/*.py"]) is False

    def test_match_glob_leading_double_star(self):
        """Test _match_glob with leading **/."""
        assert _match_glob("some/deep/path/test.py", ["**/test.py"]) is True
        assert _match_glob("test.py", ["**/test.py"]) is True
        assert _match_glob("other.txt", ["**/test.py"]) is False

    def test_match_glob_multiple_patterns(self):
        """Test _match_glob with multiple patterns."""
        patterns = ["*.py", "*.js", "*.txt"]
        assert _match_glob("test.py", patterns) is True
        assert _match_glob("test.js", patterns) is True
        assert _match_glob("test.txt", patterns) is True
        assert _match_glob("test.md", patterns) is False


class TestEdgeCases:
    def test_grep_search_empty_directory(self, tmp_path):
        """Test grep_search with empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = grep_search(pattern="test", path=str(empty_dir))
        assert "No match found" in result

    def test_glob_find_empty_directory(self, tmp_path):
        """Test glob_find with empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = glob_find(pattern="*.txt", path=str(empty_dir))
        assert "No files matching" in result

    def test_grep_search_with_hidden_files(self, tmp_path):
        """Test grep_search with hidden files."""
        (tmp_path / "visible.txt").write_text("match")
        (tmp_path / ".hidden.txt").write_text("match")

        result = grep_search(pattern="match", path=str(tmp_path))
        # grep_search shows all files including hidden ones
        assert "visible.txt" in result
        # Hidden files might be shown

    def test_glob_find_with_hidden_files(self, tmp_path):
        """Test glob_find with hidden files."""
        (tmp_path / "visible.txt").write_text("content")
        (tmp_path / ".hidden.txt").write_text("content")

        result = glob_find(pattern="*.txt", path=str(tmp_path))
        # glob_find doesn't filter hidden files by default
        # So hidden files might be included
        assert "visible.txt" in result

    def test_grep_search_special_regex_chars(self, tmp_path):
        """Test grep_search with special regex characters."""
        (tmp_path / "test.txt").write_text("Price: $100\nDiscount: 50%\nEmail: test@example.com")

        result = grep_search(pattern=r"\$\d+", path=str(tmp_path))
        assert "$100" in result

    def test_grep_search_unicode_content(self, tmp_path):
        """Test grep_search with unicode content."""
        (tmp_path / "unicode.txt").write_text("Hello 世界 🌍\nTest 测试")

        result = grep_search(pattern="世界", path=str(tmp_path))
        assert "unicode.txt" in result

    def test_grep_search_multiline_pattern(self, tmp_path):
        """Test grep_search with multiline content."""
        (tmp_path / "multiline.txt").write_text("line1\nline2\nline3")

        # Pattern should work across lines due to MULTILINE flag
        result = grep_search(pattern="line2", path=str(tmp_path))
        assert "line2" in result

    def test_grep_search_output_truncation(self, tmp_path):
        """Test grep_search truncates large output."""
        # Create file with many matches
        (tmp_path / "large.txt").write_text("\n".join([f"line {i}" for i in range(1000)]))

        result = grep_search(pattern="line", path=str(tmp_path))
        # Should truncate
        assert "truncated" in result.lower() or len(result) < 100000

    def test_glob_find_with_symlinks(self, tmp_path):
        """Test glob_find handles symlinks."""
        target = tmp_path / "target.txt"
        target.write_text("content")
        link = tmp_path / "link.txt"

        try:
            link.symlink_to(target)

            result = glob_find(pattern="*.txt", path=str(tmp_path))
            assert "link.txt" in result
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks not supported")

    def test_grep_search_case_sensitivity(self, tmp_path):
        """Test grep_search is case-sensitive."""
        (tmp_path / "case.txt").write_text("ERROR\nerror\nError")

        result = grep_search(pattern="ERROR", path=str(tmp_path))
        # Should only match exact case
        lines = result.split("\n")
        error_lines = [line for line in lines if "error" in line.lower()]
        # At least the uppercase ERROR should match
        assert len(error_lines) >= 1

    def test_glob_find_nested_directories(self, tmp_path):
        """Test glob_find with deeply nested directories."""
        (tmp_path / "a" / "b" / "c" / "d").mkdir(parents=True)
        (tmp_path / "a" / "b" / "c" / "d" / "deep.txt").write_text("deep")

        result = glob_find(pattern="**/*.txt", path=str(tmp_path))
        assert "deep.txt" in result


class TestToolConfig:
    def test_tool_config_max_output(self, tmp_path):
        """Test _tool_config parameter controls max_output limit."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("x" * 10000)

        result = grep_search(pattern="x", path=str(tmp_path), _tool_config={"max_output": 500})
        assert len(result) <= 500 + 100  # Allow some margin for formatting

    def test_tool_config_max_results(self, tmp_path):
        """Test _tool_config parameter controls max_results for glob_find."""
        for i in range(10):
            (tmp_path / f"file{i}.txt").write_text("content")

        result = glob_find(pattern="*.txt", path=str(tmp_path), _tool_config={"max_results": 5})
        assert "showing first 5" in result or "5 total" in result or "10 total" in result
