"""Tests for koboi.tools.builtin.repo_map (Wave 4)."""

from __future__ import annotations

from koboi.modes import is_read_only_tool
from koboi.tools.builtin.repo_map import repo_map


def _write(path, content=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


class TestTreeRendering:
    def test_nested_dirs_rendered(self, tmp_path):
        _write(tmp_path / "a" / "b" / "c.txt", "hi")
        result = repo_map(str(tmp_path), max_depth=5)
        assert "a/" in result
        assert "b/" in result
        assert "c.txt" in result

    def test_junk_dirs_skipped(self, tmp_path):
        _write(tmp_path / "node_modules" / "pkg.js", "x")
        _write(tmp_path / "__pycache__" / "x.pyc", "x")
        _write(tmp_path / ".git" / "HEAD", "ref")
        _write(tmp_path / "real.py", "def f(): pass")
        result = repo_map(str(tmp_path))
        assert "node_modules" not in result
        assert "__pycache__" not in result
        assert ".git" not in result
        assert "real.py" in result

    def test_empty_dir_returns_header_only(self, tmp_path):
        result = repo_map(str(tmp_path))
        assert result.strip().endswith("/")

    def test_nonexistent_path_errors(self, tmp_path):
        result = repo_map(str(tmp_path / "nope"))
        assert result.startswith("Error:")


class TestSymbolExtraction:
    def test_python_function_and_class(self, tmp_path):
        _write(
            tmp_path / "mod.py",
            "def foo(a, b=1):\n    pass\n\n\nclass Bar:\n    def method(self):\n        pass\n",
        )
        result = repo_map(str(tmp_path))
        assert "symbols:" in result
        assert "foo(a, b=...)" in result
        assert "Bar" in result
        # Nested methods are not top-level -- must not appear as a bare symbol.
        assert "method" not in result.split("symbols:")[1].split("\n")[0]

    def test_async_function_captured(self, tmp_path):
        _write(tmp_path / "mod.py", "async def fetch(url):\n    pass\n")
        result = repo_map(str(tmp_path))
        assert "fetch(url)" in result

    def test_malformed_python_does_not_crash(self, tmp_path):
        _write(tmp_path / "broken.py", "def f(:\n    this is not valid python !!!\n")
        result = repo_map(str(tmp_path))
        assert "broken.py" in result  # file still listed
        assert "Error" not in result

    def test_non_python_regex_fallback(self, tmp_path):
        _write(tmp_path / "app.js", "export function handler(req, res) {\n  return 1;\n}\n")
        result = repo_map(str(tmp_path))
        assert "handler" in result

    def test_include_symbols_false_omits_outline(self, tmp_path):
        _write(tmp_path / "mod.py", "def foo():\n    pass\n")
        result = repo_map(str(tmp_path), include_symbols=False)
        assert "symbols:" not in result


class TestBounds:
    def test_max_depth_truncates_descent(self, tmp_path):
        _write(tmp_path / "a" / "b" / "c" / "d" / "deep.txt", "x")
        result = repo_map(str(tmp_path), max_depth=1)
        assert "a/" in result
        assert "deep.txt" not in result

    def test_max_entries_truncation_noted(self, tmp_path):
        for i in range(20):
            _write(tmp_path / f"file{i}.txt", "x")
        result = repo_map(str(tmp_path), max_entries=5)
        assert "truncated at 5 entries" in result


class TestSandboxContainment:
    def test_path_escape_rejected(self, tmp_path):
        class DenyingSandbox:
            def validate_path(self, path):
                raise PermissionError("escape")

        result = repo_map(str(tmp_path), _deps={"sandbox": DenyingSandbox()})
        assert result.startswith("Error: no access")


class TestModeAllowlist:
    def test_repo_map_is_read_only(self):
        assert is_read_only_tool("repo_map") is True
