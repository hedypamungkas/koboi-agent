"""Tests for koboi.eval.t.loader -- .eval.py discovery."""

from __future__ import annotations

from koboi.eval.t.loader import PythonTestLoader, discover


class TestDiscovery:
    async def test_discover_single_file(self, tmp_path):
        (tmp_path / "calc.eval.py").write_text("CONFIG='x'\nasync def test_a(t):\n    pass\n")
        tests = discover(tmp_path)
        assert len(tests) == 1
        assert tests[0].func_name == "test_a"
        assert tests[0].config == "x"
        assert tests[0].case_name == "calc.eval::test_a"

    async def test_discover_multiple_funcs_and_module_attrs(self, tmp_path):
        (tmp_path / "multi.eval.py").write_text(
            "CONFIG='cfg.yaml'\n"
            "MOCK_RESPONSES=[1, 2]\n"
            "TAGS=['smoke']\n"
            "TIMEOUT=5.0\n"
            "async def test_one(t):\n    pass\n"
            "async def test_two(t):\n    pass\n"
            "async def helper(t):\n    pass\n"  # not collected
        )
        tests = discover(tmp_path)
        assert [t.func_name for t in tests] == ["test_one", "test_two"]
        first = tests[0]
        assert first.use_mock is True  # MOCK_RESPONSES present
        assert first.tags == ["smoke"]
        assert first.timeout == 5.0
        assert first.mock_responses == [1, 2]

    async def test_skip_non_test_coroutines(self, tmp_path):
        (tmp_path / "x.eval.py").write_text(
            "def test_sync(t):\n    pass\nasync def not_test(t):\n    pass\nasync def test_ok(t):\n    pass\n"
        )
        assert [t.func_name for t in discover(tmp_path)] == ["test_ok"]

    async def test_use_mock_flag_without_responses(self, tmp_path):
        (tmp_path / "x.eval.py").write_text("USE_MOCK=True\nasync def test_a(t):\n    pass\n")
        test = discover(tmp_path)[0]
        assert test.use_mock is True
        assert test.mock_responses is None

    async def test_discover_recursive_glob(self, tmp_path):
        (tmp_path / "a.eval.py").write_text("async def test_a(t):\n    pass\n")
        (tmp_path / "nested").mkdir()
        (tmp_path / "nested" / "b.eval.py").write_text("async def test_b(t):\n    pass\n")
        (tmp_path / "ignore.txt").write_text("nope")
        assert sorted(t.func_name for t in discover(tmp_path)) == ["test_a", "test_b"]

    async def test_discover_empty_dir(self, tmp_path):
        assert discover(tmp_path) == []

    async def test_function_without_param_is_skipped(self, tmp_path):
        (tmp_path / "x.eval.py").write_text("async def test_no_param():\n    pass\nasync def test_ok(t):\n    pass\n")
        assert [t.func_name for t in discover(tmp_path)] == ["test_ok"]

    async def test_loader_custom_glob(self, tmp_path):
        (tmp_path / "t.py").write_text("async def test_a(t):\n    pass\n")
        loader = PythonTestLoader(glob="*.py")
        assert len(loader.discover(tmp_path)) == 1
        assert discover(tmp_path) == []  # default glob (**/*.eval.py) does not match t.py
