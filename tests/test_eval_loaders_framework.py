"""Tests for framework-specific eval loaders: BFCL, GAIA, SWE-bench."""
from __future__ import annotations

import json

import pytest

from koboi.eval.loaders.bfcl_loader import BFCLLoader, _parse_python_call
from koboi.eval.loaders.gaia_loader import GAIALoader
from koboi.eval.loaders.swe_bench_loader import SWEBenchLoader


class TestBFCLLoader:
    def test_framework_name(self):
        assert BFCLLoader().framework_name() == "bfcl"

    @pytest.mark.asyncio
    async def test_load_jsonl(self, tmp_path):
        entry = {
            "question": ["What is the weather?"],
            "function": [{"name": "get_weather", "description": "Get weather", "parameters": {}}],
            "ground_truth": [{"name": "get_weather", "arguments": {"city": "NYC"}}],
        }
        path = tmp_path / "test.jsonl"
        path.write_text(json.dumps(entry) + "\n")
        loader = BFCLLoader()
        cases = await loader.load(str(path))
        assert len(cases) == 1
        assert cases[0].user_message == "What is the weather?"
        assert "bfcl" in cases[0].tags

    @pytest.mark.asyncio
    async def test_load_directory(self, tmp_path):
        entry = {
            "question": ["query"],
            "function": [],
            "ground_truth": [],
        }
        (tmp_path / "simple.jsonl").write_text(json.dumps(entry) + "\n")
        loader = BFCLLoader()
        cases = await loader.load(str(tmp_path))
        assert len(cases) == 1

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, tmp_path):
        loader = BFCLLoader()
        cases = await loader.load(str(tmp_path / "nonexistent"))
        assert len(cases) == 0

    @pytest.mark.asyncio
    async def test_max_cases(self, tmp_path):
        for i in range(5):
            entry = {"question": [f"q{i}"], "function": [], "ground_truth": []}
            (tmp_path / f"t{i}.jsonl").write_text(json.dumps(entry) + "\n")
        loader = BFCLLoader()
        cases = await loader.load(str(tmp_path), max_cases=3)
        assert len(cases) == 3

    def test_sanitize_name(self):
        assert BFCLLoader._sanitize_name("org.func") == "org_func"

    def test_parse_function_def_openai_format(self):
        fn = {"type": "function", "function": {"name": "test", "parameters": {"type": "object"}}}
        result = BFCLLoader._parse_function_def(fn)
        assert result["type"] == "function"

    def test_parse_function_def_bfcl_format(self):
        fn = {"name": "test.fn", "description": "desc", "parameters": {"type": "dict"}}
        result = BFCLLoader._parse_function_def(fn)
        assert result["function"]["name"] == "test_fn"
        assert result["function"]["parameters"]["type"] == "object"

    def test_parse_ground_truth_simple(self):
        gt = [{"name": "func", "arguments": {"a": 1}}]
        result = BFCLLoader._parse_ground_truth(gt)
        assert len(result) == 1
        assert result[0]["name"] == "func"

    def test_parse_ground_truth_nested(self):
        gt = [[{"name": "f1", "arguments": {}}, {"name": "f2", "arguments": {}}]]
        result = BFCLLoader._parse_ground_truth(gt)
        assert len(result) == 2

    def test_parse_ground_truth_string(self):
        gt = ["func(arg1='hello')"]
        result = BFCLLoader._parse_ground_truth(gt)
        assert len(result) == 1
        assert result[0]["name"] == "func"

    def test_parse_ground_truth_empty(self):
        assert BFCLLoader._parse_ground_truth([]) == []

    def test_parse_ground_truth_dict(self):
        gt = {"name": "func", "arguments": {"a": 1}}
        result = BFCLLoader._parse_ground_truth(gt)
        assert len(result) == 1

    def test_extract_category(self):
        assert BFCLLoader._extract_category("simple.jsonl", None) == "simple"
        assert BFCLLoader._extract_category("unknown_file.jsonl", None) == "unknown"

    def test_extract_v4_category(self):
        assert BFCLLoader._extract_v4_category("questions_BFCL_v4_simple_python.json") == "simple_python"

    def test_parse_python_call(self):
        result = _parse_python_call("func(a=1, b='hello')")
        assert result["name"] == "func"
        assert result["arguments"]["a"] == 1

    def test_parse_python_call_no_args(self):
        result = _parse_python_call("func()")
        assert result["name"] == "func"
        assert result["arguments"] == {}

    def test_parse_python_call_invalid(self):
        assert _parse_python_call("not a call") is None

    def test_parse_v4_ground_truth(self):
        gt = [{"get_weather": {"city": ["NYC", "New York"]}}]
        result = BFCLLoader._parse_v4_ground_truth(gt)
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["arguments"]["city"] == "NYC"


# --- GAIA Loader ---


class TestGAIALoader:
    def test_framework_name(self):
        assert GAIALoader().framework_name() == "gaia"

    def test_row_to_eval_case(self):
        loader = GAIALoader()
        row = {"Question": "What is 2+2?", "Final answer": "4", "Level": 1, "task_id": "t1"}
        case = loader._row_to_eval_case(row, 0, 1)
        assert case is not None
        assert case.user_message == "What is 2+2?"
        assert case.expected_answer == "4"
        assert "gaia" in case.tags
        assert "level-1" in case.tags

    def test_row_to_eval_case_no_question(self):
        loader = GAIALoader()
        row = {"Final answer": "4"}
        case = loader._row_to_eval_case(row, 0, 1)
        assert case is None

    def test_row_to_eval_case_alt_fields(self):
        loader = GAIALoader()
        row = {"question": "Q?", "answer": "A.", "Level": 2}
        case = loader._row_to_eval_case(row, 5, 2)
        assert case is not None
        assert case.user_message == "Q?"
        assert case.name == "gaia_5"

    @pytest.mark.asyncio
    async def test_load_local_nonexistent(self):
        loader = GAIALoader()
        cases = await loader.load("/nonexistent/path", local_only=True)
        assert len(cases) == 0

    @pytest.mark.asyncio
    async def test_load_local_dir_empty(self, tmp_path):
        loader = GAIALoader()
        cases = await loader.load(str(tmp_path), local_only=True)
        assert len(cases) == 0


# --- SWE-bench Loader ---


class TestSWEBenchLoader:
    def test_framework_name(self):
        assert SWEBenchLoader().framework_name() == "swe-bench"

    @pytest.mark.asyncio
    async def test_load_nonexistent(self):
        loader = SWEBenchLoader()
        cases = await loader.load("/nonexistent/path", local_only=True)
        assert len(cases) == 0

    @pytest.mark.asyncio
    async def test_load_empty_dir(self, tmp_path):
        loader = SWEBenchLoader()
        cases = await loader.load(str(tmp_path), local_only=True)
        assert len(cases) == 0

    def test_row_to_eval_case(self):
        loader = SWEBenchLoader()
        row = {
            "instance_id": "test-123",
            "repo": "org/repo",
            "problem_statement": "Fix the bug",
            "base_commit": "abc123",
            "patch": "diff --git a/f.py",
        }
        case = loader._row_to_eval_case(row, 0)
        assert case is not None
        assert case.user_message == "Fix the bug"
        assert case.expected_answer == "diff --git a/f.py"
        assert "swe-bench" in case.tags

    def test_row_to_eval_case_no_statement(self):
        loader = SWEBenchLoader()
        row = {"instance_id": "test-123"}
        case = loader._row_to_eval_case(row, 0)
        assert case is None
