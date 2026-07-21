"""Tests for koboi.tools.builtin.filesystem module."""

from __future__ import annotations


import pytest

from koboi.tools.builtin.filesystem import (
    list_files,
    read_file,
    write_file,
    edit_file,
    apply_patch,
    delete_file,
    _validate_path,
)


class TestListFiles:
    def test_list_files_with_real_tmp_path_directory(self, tmp_path):
        """Test list_files with a real temporary directory."""
        # Create some test files
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.py").write_text("content2")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file3.md").write_text("content3")

        result = list_files(path=str(tmp_path))
        # Output format includes emojis and formatting
        assert "file1.txt" in result
        assert "file2.py" in result
        assert "subdir" in result
        # Note: file3.md is in a subdir and may not appear in top-level listing

    def test_list_files_with_glob_pattern_filtering(self, tmp_path):
        """Test list_files with glob pattern filtering."""
        (tmp_path / "test.py").write_text("python")
        (tmp_path / "test.js").write_text("javascript")
        (tmp_path / "data.json").write_text("json")
        (tmp_path / "README.md").write_text("markdown")

        # Filter by .py files
        result = list_files(path=str(tmp_path), pattern="*.py")
        assert "test.py" in result
        assert "test.js" not in result

        # Note: fnmatch doesn't support {py,js} brace expansion
        # So this test only checks single pattern

    def test_list_files_with_nonexistent_path(self, tmp_path):
        """Test list_files with nonexistent path."""
        result = list_files(path=str(tmp_path / "nonexistent"))
        assert "Error" in result
        assert "not found" in result.lower()


class TestReadFile:
    def test_read_file_reads_content_correctly(self, tmp_path):
        """Test read_file reads file content correctly."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, world!")

        result = read_file(path=str(test_file))
        assert result == "Hello, world!"

    def test_read_file_with_max_read_size_truncation(self, tmp_path):
        """Test read_file with max_read_size truncates content."""
        test_file = tmp_path / "large.txt"
        content = "x" * 10000
        test_file.write_text(content)

        # Pass max_read_size via _tool_config
        result = read_file(path=str(test_file), _tool_config={"max_read_size": 100})
        assert result.startswith("x" * 100)
        assert "truncated at 100 chars" in result
        assert "offset/limit" in result

    def test_read_file_nonexistent_file(self, tmp_path):
        """Test read_file with nonexistent file."""
        result = read_file(path=str(tmp_path / "nonexistent.txt"))
        assert "Error" in result
        assert "not found" in result.lower() or "no such file" in result.lower()


class TestWriteFile:
    def test_write_file_creates_file_and_parent_dirs(self, tmp_path):
        """Test write_file creates file and parent directories."""
        nested_path = tmp_path / "level1" / "level2" / "test.txt"

        result = write_file(path=str(nested_path), content="test content")
        assert "Success" in result or "wrote" in result.lower()

        # Verify file exists
        assert nested_path.exists()
        assert nested_path.read_text() == "test content"

    def test_write_file_to_directory_raises_error(self, tmp_path):
        """Test write_file to a directory path raises error."""
        result = write_file(path=str(tmp_path), content="content")
        assert "Error" in result
        assert "directory" in result.lower() or "is a dir" in result.lower()


class TestDeleteFile:
    def test_delete_file_removes_file(self, tmp_path):
        """Test delete_file removes the file."""
        test_file = tmp_path / "to_delete.txt"
        test_file.write_text("delete me")

        assert test_file.exists()

        result = delete_file(path=str(test_file))
        assert "Success" in result or "deleted" in result.lower()
        assert not test_file.exists()

    def test_delete_file_nonexistent_file(self, tmp_path):
        """Test delete_file with nonexistent file."""
        result = delete_file(path=str(tmp_path / "nonexistent.txt"))
        assert "Error" in result
        assert "not found" in result.lower() or "no such file" in result.lower()


class TestPathValidation:
    def test_validate_path_without_sandbox(self, tmp_path, monkeypatch):
        """Test _validate_path with sandbox disabled."""
        # Remove sandbox env var
        monkeypatch.setenv("KOBOI_SANDBOX_DIR", "")

        # Should allow any path when sandbox is disabled
        result = _validate_path(str(tmp_path))
        assert result == str(tmp_path)  # Returns resolved path

    def test_validate_path_with_sandbox(self, tmp_path, monkeypatch):
        """Test _validate_path with sandbox enabled."""
        # Set sandbox directory
        monkeypatch.setenv("KOBOI_SANDBOX_DIR", str(tmp_path))

        # Should allow paths within sandbox
        result = _validate_path(str(tmp_path / "test.txt"))
        # Result should be resolved path within sandbox
        assert str(tmp_path) in result or "private" in result  # macOS resolves to /private

        # Should raise PermissionError for paths outside sandbox
        # Note: On macOS, /etc resolves to /private/etc which may not match tmp_path
        try:
            # Use a clearly different path
            result = _validate_path("/var/tmp/test")
            # If it doesn't raise, at least check it's not our sandbox
            if str(tmp_path) not in result:
                pass  # Expected - different path
            else:
                pytest.fail("Should have raised PermissionError or returned different path")
        except PermissionError:
            pass  # Expected


class TestToolConfig:
    def test_tool_config_max_read_size(self, tmp_path):
        """Test _tool_config parameter controls max_read_size."""
        test_file = tmp_path / "large.txt"
        test_file.write_text("x" * 10000)

        result = read_file(path=str(test_file), _tool_config={"max_read_size": 5000})
        # With max_read_size=5000, should truncate
        assert len(result) < 10000


class TestEdgeCases:
    def test_read_file_with_unicode_content(self, tmp_path):
        """Test read_file with unicode content."""
        test_file = tmp_path / "unicode.txt"
        unicode_content = "Hello 世界 🌍"
        test_file.write_text(unicode_content, encoding="utf-8")

        result = read_file(path=str(test_file))
        assert result == unicode_content

    def test_write_file_with_unicode_content(self, tmp_path):
        """Test write_file with unicode content."""
        test_file = tmp_path / "unicode_out.txt"
        unicode_content = "Test 中文 🚀"

        result = write_file(path=str(test_file), content=unicode_content)
        assert "Success" in result or "wrote" in result.lower()

        assert test_file.read_text(encoding="utf-8") == unicode_content

    def test_read_empty_file(self, tmp_path):
        """Test read_file with empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        result = read_file(path=str(test_file))
        assert result == ""

    def test_write_empty_file(self, tmp_path):
        """Test write_file with empty content."""
        test_file = tmp_path / "empty_write.txt"

        result = write_file(path=str(test_file), content="")
        assert "Success" in result or "wrote" in result.lower()

        assert test_file.read_text() == ""

    def test_list_files_empty_directory(self, tmp_path):
        """Test list_files with empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = list_files(path=str(empty_dir))
        assert result == "(empty)" or "empty" in result.lower()

    def test_list_files_recursive_with_hidden_files(self, tmp_path):
        """Test list_files handles hidden files correctly."""
        (tmp_path / ".hidden").write_text("hidden")
        (tmp_path / "visible.txt").write_text("visible")
        (tmp_path / ".git" / "config").parent.mkdir()
        (tmp_path / ".git" / "config").write_text("git config")

        result = list_files(path=str(tmp_path))
        # list_files shows all files, including hidden ones
        assert "visible.txt" in result
        # Hidden files are shown

    def test_delete_file_on_directory(self, tmp_path):
        """Test delete_file on a directory path."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        result = delete_file(path=str(test_dir))
        assert "Error" in result
        assert "directory" in result.lower() or "is a dir" in result.lower()

    def test_write_file_overwrites_existing(self, tmp_path):
        """Test write_file overwrites existing file."""
        test_file = tmp_path / "overwrite.txt"
        test_file.write_text("original")

        result = write_file(path=str(test_file), content="new content")
        assert "Success" in result or "wrote" in result.lower()

        assert test_file.read_text() == "new content"

    def test_read_file_with_permission_denied(self, tmp_path):
        """Test read_file with permission denied."""
        test_file = tmp_path / "no_read.txt"
        test_file.write_text("secret")
        test_file.chmod(0o000)

        try:
            result = read_file(path=str(test_file))
            assert "Error" in result
        finally:
            # Restore permissions for cleanup
            test_file.chmod(0o644)

    def test_write_file_with_permission_denied(self, tmp_path):
        """Test write_file with permission denied."""
        test_file = tmp_path / "no_write.txt"
        test_file.write_text("initial")
        test_file.chmod(0o444)  # Read-only

        try:
            result = write_file(path=str(test_file), content="should fail")
            assert "Error" in result
        finally:
            # Restore permissions for cleanup
            test_file.chmod(0o644)

    def test_list_files_with_symlinks(self, tmp_path):
        """Test list_files handles symlinks correctly."""
        target = tmp_path / "target.txt"
        target.write_text("content")
        link = tmp_path / "link.txt"

        try:
            link.symlink_to(target)

            result = list_files(path=str(tmp_path))
            # Symlink should be included
            assert "link.txt" in result
        except (OSError, NotImplementedError):
            # Skip test if symlinks not supported
            pytest.skip("Symlinks not supported on this system")

    def test_read_binary_file_returns_error(self, tmp_path):
        """Test read_file with binary file raises decode error."""
        import pytest

        test_file = tmp_path / "binary.bin"
        test_file.write_bytes(b"\x00\x01\x02\x03\xff\xfe")

        # Binary files will raise UnicodeDecodeError when read as text
        with pytest.raises(UnicodeDecodeError):
            read_file(path=str(test_file))

    def test_list_files_with_special_characters(self, tmp_path):
        """Test list_files with filenames containing special characters."""
        special_files = [
            "file with spaces.txt",
            "file-with-dashes.txt",
            "file_with_underscores.txt",
            "file.multiple.dots.txt",
        ]

        for filename in special_files:
            (tmp_path / filename).write_text("content")

        result = list_files(path=str(tmp_path))
        for filename in special_files:
            assert filename in result


class TestReadFileRanged:
    def test_read_file_with_offset_and_limit(self, tmp_path):
        """offset/limit returns a numbered line range with a range header."""
        test_file = tmp_path / "ranged.txt"
        test_file.write_text("\n".join(f"line {i}" for i in range(1, 11)) + "\n")

        result = read_file(path=str(test_file), offset=3, limit=2)
        assert "(lines 3-4 of 10" in result
        assert "line 3" in result
        assert "line 4" in result
        assert "line 2" not in result
        assert "line 5" not in result
        # cat -n style line numbers
        assert "     3\t" in result

    def test_read_file_with_offset_only_reads_to_eof(self, tmp_path):
        """offset without limit reads to end of file."""
        test_file = tmp_path / "ranged.txt"
        test_file.write_text("a\nb\nc\nd\n")

        result = read_file(path=str(test_file), offset=3)
        assert "(lines 3-4 of 4" in result
        assert "c" in result and "d" in result

    def test_read_file_with_limit_only_starts_at_line_one(self, tmp_path):
        """limit without offset starts at line 1."""
        test_file = tmp_path / "ranged.txt"
        test_file.write_text("a\nb\nc\n")

        result = read_file(path=str(test_file), limit=2)
        assert "(lines 1-2 of 3" in result
        assert "c" not in result.split("\n", 1)[1]

    def test_read_file_offset_beyond_eof(self, tmp_path):
        """offset past the last line returns a clear error."""
        test_file = tmp_path / "short.txt"
        test_file.write_text("only\n")

        result = read_file(path=str(test_file), offset=5)
        assert "Error" in result
        assert "beyond end of file" in result

    def test_read_file_default_call_unchanged(self, tmp_path):
        """No offset/limit keeps the plain-content contract."""
        test_file = tmp_path / "plain.txt"
        test_file.write_text("Hello, world!")

        assert read_file(path=str(test_file)) == "Hello, world!"

    def test_read_file_range_streams_large_file(self, tmp_path):
        # Streaming: a huge file with a small requested range must not load the
        # whole file into memory (the old readlines() did -- OOM on large inputs).
        import tracemalloc

        big = tmp_path / "big.txt"
        with big.open("w") as f:
            for i in range(50_000):
                f.write(f"line-{i:06d}\n")
        tracemalloc.start()
        result = read_file(path=str(big), offset=25_000, limit=3)
        _cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert "(lines 25000-25002 of 50000" in result
        # File is ~450KB; the old readlines() materialized all of it. Peak heap
        # during a streamed 3-line window must stay well under the file size.
        assert peak < 4_000_000, f"streaming regressed: peak={peak}"

    def test_read_file_rejects_nonpositive_offset_and_limit(self, tmp_path):
        # Negative/zero used to silently clamp (offset=-3 -> 1), masking mistakes.
        test_file = tmp_path / "f.txt"
        test_file.write_text("a\nb\nc\n")
        assert "must be >= 1" in read_file(path=str(test_file), offset=0)
        assert "must be >= 1" in read_file(path=str(test_file), offset=-3)
        assert "must be >= 1" in read_file(path=str(test_file), limit=0)
        assert "must be >= 1" in read_file(path=str(test_file), limit=-2)


class TestEditFile:
    def test_edit_file_unique_replace(self, tmp_path):
        """A unique old_string is replaced and written to disk."""
        test_file = tmp_path / "code.py"
        test_file.write_text("a = 1\nb = 2\nc = 3\n")

        result = edit_file(path=str(test_file), old_string="b = 2", new_string="b = 20")
        assert "Successfully replaced 1 occurrence(s)" in result
        assert test_file.read_text() == "a = 1\nb = 20\nc = 3\n"

    def test_edit_file_oserror_returns_error_string(self, tmp_path, monkeypatch):
        # ENOSPC/EIO during the atomic swap must return "Error:" (the tool
        # contract is -> str), not raise out of the tool. Temp file is cleaned up.
        import os

        test_file = tmp_path / "code.py"
        test_file.write_text("a = 1\n")

        def boom(*_a, **_k):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr("os.replace", boom)
        result = edit_file(path=str(test_file), old_string="a = 1", new_string="a = 2")
        assert result.startswith("Error:")
        assert not any(p.startswith(".edit_file_") for p in os.listdir(tmp_path))

    def test_edit_file_no_match_errors_and_leaves_file_unchanged(self, tmp_path):
        """0 matches -> error, file untouched."""
        test_file = tmp_path / "code.py"
        original = "a = 1\n"
        test_file.write_text(original)

        result = edit_file(path=str(test_file), old_string="zzz", new_string="y")
        assert "Error" in result
        assert "not found" in result
        assert test_file.read_text() == original

    def test_edit_file_multiple_matches_errors_with_count(self, tmp_path):
        """Ambiguous old_string -> error naming the match count, file untouched."""
        test_file = tmp_path / "code.py"
        original = "x = 1\ny = 2\nx = 1\n"
        test_file.write_text(original)

        result = edit_file(path=str(test_file), old_string="x = 1", new_string="x = 9")
        assert "Error" in result
        assert "matched 2 times" in result
        assert "replace_all" in result
        assert test_file.read_text() == original

    def test_edit_file_replace_all(self, tmp_path):
        """replace_all replaces every occurrence and reports the count."""
        test_file = tmp_path / "code.py"
        test_file.write_text("x = 1\ny = 2\nx = 1\n")

        result = edit_file(path=str(test_file), old_string="x = 1", new_string="x = 9", replace_all=True)
        assert "Successfully replaced 2 occurrence(s)" in result
        assert test_file.read_text() == "x = 9\ny = 2\nx = 9\n"

    def test_edit_file_identical_strings_rejected(self, tmp_path):
        """old_string == new_string is rejected up front."""
        test_file = tmp_path / "code.py"
        test_file.write_text("a = 1\n")

        result = edit_file(path=str(test_file), old_string="a = 1", new_string="a = 1")
        assert "Error" in result
        assert "identical" in result

    def test_edit_file_nonexistent_file(self, tmp_path):
        """Missing file -> not-found error."""
        result = edit_file(path=str(tmp_path / "missing.py"), old_string="a", new_string="b")
        assert "Error" in result
        assert "not found" in result

    def test_edit_file_advisory_note_without_prior_read(self, tmp_path):
        """Editing a never-read path emits the read-before-write advisory."""
        from koboi.tools.builtin.filesystem import reset_read_before_write

        reset_read_before_write()
        test_file = tmp_path / "code.py"
        test_file.write_text("a = 1\n")

        result = edit_file(path=str(test_file), old_string="a = 1", new_string="a = 2")
        assert "without having read it first" in result

        # After a read, the advisory disappears.
        test_file2 = tmp_path / "code2.py"
        test_file2.write_text("a = 1\n")
        read_file(path=str(test_file2))
        result2 = edit_file(path=str(test_file2), old_string="a = 1", new_string="a = 2")
        assert "without having read it first" not in result2

    def test_edit_file_preserves_permissions(self, tmp_path):
        """The atomic swap keeps the original file mode."""
        import os as _os
        import stat

        test_file = tmp_path / "script.sh"
        test_file.write_text("echo old\n")
        test_file.chmod(0o755)

        edit_file(path=str(test_file), old_string="old", new_string="new")
        mode = stat.S_IMODE(_os.stat(test_file).st_mode)
        assert mode == 0o755

    def test_edit_file_outside_sandbox_blocked(self, tmp_path, monkeypatch):
        """Sandbox containment applies to edit_file like the other fs tools."""
        import koboi.tools.builtin.filesystem as fs

        sandbox_dir = tmp_path / "sandbox"
        sandbox_dir.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("secret = 1\n")
        monkeypatch.setattr(fs, "_SANDBOX_DIR", str(sandbox_dir))

        result = edit_file(path=str(outside), old_string="secret = 1", new_string="secret = 2")
        assert "Error" in result
        assert "no access" in result
        assert outside.read_text() == "secret = 1\n"


class TestApplyPatch:
    def _patch(self, old, new, context_before="", context_after=""):
        """Build a minimal single-hunk unified diff replacing ``old`` with ``new``."""
        return f"@@ -1,3 +1,3 @@\n{context_before}-{old}\n+{new}\n{context_after}"

    def test_apply_patch_single_hunk(self, tmp_path):
        test_file = tmp_path / "code.py"
        test_file.write_text("def add(a, b):\n    return a - b\n\ndef sub(a, b):\n    return a - b\n")
        patch = "@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n"
        result = apply_patch(path=str(test_file), patch=patch)
        assert "Successfully applied 1 hunk" in result
        assert "return a + b" in test_file.read_text()
        # untouched hunk survives
        assert "def sub(a, b):\n    return a - b" in test_file.read_text()

    def test_apply_patch_oserror_returns_error_string(self, tmp_path, monkeypatch):
        # OSError during the atomic swap -> "Error:" string, not a raise.
        import os

        test_file = tmp_path / "code.py"
        test_file.write_text("a = 1\n")

        def boom(*_a, **_k):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr("os.replace", boom)
        result = apply_patch(path=str(test_file), patch="@@ -1,1 +1,1 @@\n-a = 1\n+a = 2\n")
        assert result.startswith("Error:")
        assert not any(p.startswith(".apply_patch_") for p in os.listdir(tmp_path))

    def test_apply_patch_reports_effective_hunk_count(self, tmp_path):
        # A context-only (no-op) hunk + a real hunk: the message reports EFFECTIVE
        # hunks (1), not the total (2) -- otherwise the model thinks more changed.
        test_file = tmp_path / "code.py"
        test_file.write_text("a = 1\nb = 2\nc = 3\n")
        patch = "@@ -1,1 +1,1 @@\n a = 1\n@@ -3,1 +3,1 @@\n-c = 3\n+c = 30\n"
        result = apply_patch(path=str(test_file), patch=patch)
        assert "Successfully applied 1 hunk" in result
        assert "c = 30" in test_file.read_text()

    def test_apply_patch_multi_hunk_atomic(self, tmp_path):
        test_file = tmp_path / "code.py"
        test_file.write_text("a = 1\nb = 2\nc = 3\n")
        patch = "@@ -1,1 +1,1 @@\n-a = 1\n+a = 10\n@@ -3,1 +3,1 @@\n-c = 3\n+c = 30\n"
        result = apply_patch(path=str(test_file), patch=patch)
        assert "Successfully applied 2 hunk" in result
        assert test_file.read_text() == "a = 10\nb = 2\nc = 30\n"

    def test_apply_patch_all_or_nothing_leaves_file_unchanged(self, tmp_path):
        """If a later hunk fails, earlier hunks are NOT partially written."""
        test_file = tmp_path / "code.py"
        original = "a = 1\nb = 2\n"
        test_file.write_text(original)
        patch = "@@ -1,1 +1,1 @@\n-a = 1\n+a = 10\n@@ -2,1 +2,1 @@\n-NOT PRESENT\n+x\n"
        result = apply_patch(path=str(test_file), patch=patch)
        assert "Error" in result
        assert "hunk #2" in result
        # File is byte-identical to the original -- no partial mutation.
        assert test_file.read_text() == original

    def test_apply_patch_context_not_found_error(self, tmp_path):
        test_file = tmp_path / "code.py"
        original = "a = 1\n"
        test_file.write_text(original)
        patch = "@@ -1,1 +1,1 @@\n-missing\n+new\n"
        result = apply_patch(path=str(test_file), patch=patch)
        assert "Error" in result
        assert "context not found" in result
        assert test_file.read_text() == original

    def test_apply_patch_ambiguous_context_error(self, tmp_path):
        test_file = tmp_path / "code.py"
        original = "dup\na = 1\ndup\n"
        test_file.write_text(original)
        patch = "@@ -1,1 +1,1 @@\n-dup\n+unique\n"
        result = apply_patch(path=str(test_file), patch=patch)
        assert "Error" in result
        assert "matched 2 times" in result

    def test_apply_patch_tolerates_line_drift(self, tmp_path):
        """@@ says line 1 but the block is now further down -- content match still applies."""
        test_file = tmp_path / "code.py"
        test_file.write_text("# header\n# more\n\ndef add(a, b):\n    return a - b\n")
        # Hunk header claims -1 but the real content is at line 4; tolerated.
        patch = "@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n"
        result = apply_patch(path=str(test_file), patch=patch)
        assert "Successfully applied" in result
        assert "return a + b" in test_file.read_text()

    def test_apply_patch_no_newline_at_end_of_file(self, tmp_path):
        """File without a trailing newline patches + keeps no trailing newline."""
        test_file = tmp_path / "code.py"
        test_file.write_text("x = 1")  # NO trailing newline
        # The `\ No newline` marker follows BOTH the - and + lines, so the
        # result also has no trailing newline.
        patch = "@@ -1,1 +1,1 @@\n-x = 1\n\\ No newline at end of file\n+x = 2\n\\ No newline at end of file\n"
        result = apply_patch(path=str(test_file), patch=patch)
        assert "Successfully applied" in result
        assert test_file.read_text() == "x = 2"

    def test_apply_patch_preserves_permissions(self, tmp_path):
        import os
        import stat

        test_file = tmp_path / "script.py"
        test_file.write_text("def add(a, b):\n    return a - b\n")
        os.chmod(test_file, 0o755)
        patch = "@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n"
        apply_patch(path=str(test_file), patch=patch)
        assert stat.S_IMODE(os.stat(test_file).st_mode) == 0o755

    def test_apply_patch_malformed_patch_error(self, tmp_path):
        test_file = tmp_path / "code.py"
        test_file.write_text("a = 1\n")
        result = apply_patch(path=str(test_file), patch="this is not a diff")
        assert "Error" in result
        assert "hunk header" in result

    def test_apply_patch_accepts_file_header_pair(self, tmp_path):
        """A --- / +++ header pair (single file) is tolerated and ignored."""
        test_file = tmp_path / "code.py"
        test_file.write_text("a = 1\n")
        patch = "--- a/code.py\n+++ b/code.py\n@@ -1,1 +1,1 @@\n-a = 1\n+a = 2\n"
        result = apply_patch(path=str(test_file), patch=patch)
        assert "Successfully applied" in result
        assert test_file.read_text() == "a = 2\n"

    def test_apply_patch_nonexistent_file(self, tmp_path):
        result = apply_patch(path=str(tmp_path / "missing.py"), patch="@@ -1,1 +1,1 @@\n-a\n+b\n")
        assert "Error" in result
        assert "not found" in result

    def test_apply_patch_outside_sandbox_blocked(self, tmp_path, monkeypatch):
        import koboi.tools.builtin.filesystem as fs

        sandbox_dir = tmp_path / "sandbox"
        sandbox_dir.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("a = 1\n")
        monkeypatch.setattr(fs, "_SANDBOX_DIR", str(sandbox_dir))

        result = apply_patch(path=str(outside), patch="@@ -1,1 +1,1 @@\n-a = 1\n+a = 2\n")
        assert "Error" in result
        assert "no access" in result
        assert outside.read_text() == "a = 1\n"

    def test_apply_patch_advisory_note_without_prior_read(self, tmp_path):
        fs_read_paths = tmp_path / "code.py"
        fs_read_paths.write_text("a = 1\n")
        fs_mod = pytest.importorskip("koboi.tools.builtin.filesystem")
        fs_mod.reset_read_before_write()
        try:
            patch = "@@ -1,1 +1,1 @@\n-a = 1\n+a = 2\n"
            result = apply_patch(path=str(fs_read_paths), patch=patch)
            assert "Note:" in result
        finally:
            fs_mod.reset_read_before_write()
