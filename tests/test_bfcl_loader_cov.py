"""koboi/eval/loaders/bfcl_loader.py -- branch coverage for BFCLLoader.

Targets the BFCL v4 directory loader, ground-truth/function-def parsers, and the
Python-style call parser that the existing loader tests don't reach.
"""

from __future__ import annotations

import json

import pytest

from koboi.eval.loaders.bfcl_loader import BFCLLoader, _parse_python_call


@pytest.fixture
def loader() -> BFCLLoader:
    return BFCLLoader()


class TestParsePythonCall:
    def test_no_match(self):
        assert _parse_python_call("not a call") is None

    def test_no_args(self):
        assert _parse_python_call("do_thing()") == {"name": "do_thing", "arguments": {}}

    def test_kwargs_json_values(self):
        result = _parse_python_call('f(a=1, b="x", c=true)')
        assert result["name"] == "f"
        assert result["arguments"]["a"] == 1
        assert result["arguments"]["b"] == "x"
        assert result["arguments"]["c"] is True

    def test_kwargs_non_json_fallback(self):
        result = _parse_python_call("f(city=Jakarta)")
        # Not valid JSON -> kept as raw string
        assert result["arguments"]["city"] == "Jakarta"


class TestExtractCategory:
    def test_known_category(self):
        assert BFCLLoader._extract_category("BFCL_v4_simple_python.json", None) == "simple"

    def test_multiple_category(self):
        assert BFCLLoader._extract_category("multiple_2.json", None) == "multiple"

    def test_unknown_category(self):
        assert BFCLLoader._extract_category("weird.json", None) == "unknown"

    def test_extract_v4_category_simple(self):
        assert BFCLLoader._extract_v4_category("questions_BFCL_v4_simple_python.json") == "simple_python"

    def test_extract_v4_category_answers_prefix(self):
        assert BFCLLoader._extract_v4_category("answers_BFCL_v4_sql.json") == "sql"

    def test_extract_v4_category_unknown(self):
        assert BFCLLoader._extract_v4_category("random.json") == "unknown"


class TestParseGroundTruth:
    def test_empty(self):
        assert BFCLLoader._parse_ground_truth([]) == []

    def test_dict_wrapped(self):
        gt = {"name": "foo", "arguments": {"x": 1}}
        out = BFCLLoader._parse_ground_truth(gt)
        assert out == [{"name": "foo", "arguments": {"x": 1}}]

    def test_nested_parallel_unwraps(self):
        gt = [[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]]
        out = BFCLLoader._parse_ground_truth(gt)
        assert {c["name"] for c in out} == {"a", "b"}

    def test_string_python_call(self):
        out = BFCLLoader._parse_ground_truth(['get_weather(city="Jakarta")'])
        assert out == [{"name": "get_weather", "arguments": {"city": "Jakarta"}}]

    def test_string_unparseable_skipped(self):
        assert BFCLLoader._parse_ground_truth(["???"]) == []

    def test_name_sanitization(self):
        out = BFCLLoader._parse_ground_truth([{"name": "ns.tool", "arguments": {}}])
        assert out[0]["name"] == "ns_tool"


class TestParseFunctionDef:
    def test_openai_format_passes_through_with_sanitized_name(self):
        fn = {"type": "function", "function": {"name": "a.b", "parameters": {"type": "object"}}}
        out = BFCLLoader._parse_function_def(fn)
        assert out["function"]["name"] == "a_b"

    def test_bfcl_direct_format_converted(self):
        fn = {"name": "calc", "description": "math", "parameters": {"type": "dict"}}
        out = BFCLLoader._parse_function_def(fn)
        assert out["type"] == "function"
        assert out["function"]["name"] == "calc"
        assert out["function"]["description"] == "math"
        # "dict" normalized to "object"
        assert out["function"]["parameters"]["type"] == "object"

    def test_no_name_key(self):
        out = BFCLLoader._parse_function_def({"parameters": {}})
        assert out["function"]["name"] == ""


class TestParseV4GroundTruth:
    def test_non_dict_items_skipped(self):
        out = BFCLLoader._parse_v4_ground_truth(["x", 5, {"f": {"a": [1, 2]}}])
        assert out == [{"name": "f", "arguments": {"a": 1}}]

    def test_args_not_dict(self):
        # Non-dict args leave parsed_args empty (only dict args are expanded)
        out = BFCLLoader._parse_v4_ground_truth([{"f": "raw"}])
        assert out == [{"name": "f", "arguments": {}}]

    def test_empty_list_val(self):
        out = BFCLLoader._parse_v4_ground_truth([{"f": {"a": []}}])
        assert out == [{"name": "f", "arguments": {"a": []}}]


class TestParseV4Entry:
    def test_nested_question_list(self, loader: BFCLLoader):
        entry = {
            "question": [[{"role": "user", "content": "hello"}, {"role": "user", "content": "world"}]],
            "function": {"name": "f", "parameters": {}},
        }
        case = loader._parse_v4_entry(entry, [], "id1", "simple")
        assert case is not None
        assert case.user_message == "hello world"

    def test_flat_question_list(self, loader: BFCLLoader):
        case = loader._parse_v4_entry({"question": ["a", "b"]}, [], "id1", "simple")
        assert case.user_message == "a b"

    def test_string_question(self, loader: BFCLLoader):
        case = loader._parse_v4_entry({"question": "hi"}, [], "id1", "simple")
        assert case.user_message == "hi"

    def test_functions_dict_promoted(self, loader: BFCLLoader):
        case = loader._parse_v4_entry({"question": "q", "function": {"name": "f"}}, [], "id1", "sql")
        assert len(case.tool_definitions) == 1

    def test_metadata_and_tags(self, loader: BFCLLoader):
        case = loader._parse_v4_entry({"question": "q"}, [], "id1", "java")
        assert case.metadata["source"] == "bfcl_v4"
        assert "java" in case.tags


class TestLoadPaths:
    async def test_nonexistent_source_returns_empty(self, loader: BFCLLoader):
        assert await loader.load("/no/such/bfcl/path") == []

    async def test_single_file_load(self, loader: BFCLLoader, tmp_path):
        f = tmp_path / "simple_test.jsonl"
        f.write_text(
            json.dumps({"question": ["q"], "function": {"name": "f"}, "ground_truth": [{"name": "f", "arguments": {}}]})
            + "\n"
        )
        cases = await loader.load(str(f))
        assert len(cases) == 1
        assert cases[0].expected_tool_calls[0]["name"] == "f"

    async def test_single_file_invalid_json_skipped(self, loader: BFCLLoader, tmp_path):
        f = tmp_path / "broken.jsonl"
        f.write_text("{not json}\n" + json.dumps({"question": ["q"]}) + "\n\n")
        cases = await loader.load(str(f))
        assert len(cases) == 1  # only the valid line; blank line skipped

    async def test_legacy_dir_with_category_filter(self, loader: BFCLLoader, tmp_path):
        (tmp_path / "simple_a.jsonl").write_text(json.dumps({"question": ["q1"]}) + "\n")
        (tmp_path / "sql_b.jsonl").write_text(json.dumps({"question": ["q2"]}) + "\n")
        cases = await loader.load(str(tmp_path), categories=["sql"])
        assert len(cases) == 1
        assert cases[0].metadata["category"] == "sql"

    async def test_max_cases_truncation(self, loader: BFCLLoader, tmp_path):
        f = tmp_path / "simple_c.jsonl"
        f.write_text("".join(json.dumps({"question": [f"q{i}"]}) + "\n" for i in range(5)))
        cases = await loader.load(str(f), max_cases=2)
        assert len(cases) == 2


class TestLoadV4Dir:
    async def test_v4_directory_merge(self, loader: BFCLLoader, tmp_path):
        q = [{"id": "x1", "question": [[{"content": "hello"}]], "function": {"name": "f"}}]
        a = [{"id": "x1", "ground_truth": [{"f": {"a": [1, 2]}}]}]
        qf = tmp_path / "questions_BFCL_v4_simple_python.json"
        af = tmp_path / "answers_BFCL_v4_simple_python.json"
        qf.write_text("".join(json.dumps(e) + "\n" for e in q))
        af.write_text("\n" + "".join(json.dumps(e) + "\n" for e in a))  # leading blank line

        cases = await loader.load(str(tmp_path))
        assert len(cases) == 1
        assert cases[0].name == "x1"
        assert cases[0].expected_tool_calls[0]["arguments"] == {"a": 1}

    async def test_v4_missing_answer_file_skipped(self, loader: BFCLLoader, tmp_path):
        qf = tmp_path / "questions_BFCL_v4_parallel.json"
        qf.write_text(json.dumps({"id": "y1", "question": ["q"]}) + "\n")
        cases = await loader.load(str(tmp_path))
        assert cases == []

    async def test_v4_entry_id_fallback(self, loader: BFCLLoader, tmp_path):
        # question without id -> synthesized "{cat}_{i}"; answer present for that id
        q = [{"question": ["q"]}]  # no id
        a = [{"id": "simple_0", "ground_truth": []}]
        (tmp_path / "questions_BFCL_v4_simple.json").write_text(json.dumps(q[0]) + "\n")
        (tmp_path / "answers_BFCL_v4_simple.json").write_text(json.dumps(a[0]) + "\n")
        cases = await loader.load(str(tmp_path))
        assert cases[0].name == "simple_0"

    async def test_v4_category_filter(self, loader: BFCLLoader, tmp_path):
        for cat, qid in [("simple", "s1"), ("sql", "z1")]:
            (tmp_path / f"questions_BFCL_v4_{cat}.json").write_text(json.dumps({"id": qid, "question": ["q"]}) + "\n")
            (tmp_path / f"answers_BFCL_v4_{cat}.json").write_text(json.dumps({"id": qid, "ground_truth": []}) + "\n")
        cases = await loader.load(str(tmp_path), categories=["sql"])
        assert [c.name for c in cases] == ["z1"]
