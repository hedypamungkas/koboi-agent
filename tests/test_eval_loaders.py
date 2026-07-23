"""Tests for koboi/eval/loaders/ -- Dataset loaders."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from koboi.eval.loaders import YAMLLoader, LoaderRegistry, register_default_loaders
from koboi.eval.loaders.ragas_generator import RAGASDataGenerator


@pytest.fixture
def yaml_file(tmp_path):
    data = {
        "cases": [
            {
                "name": "case1",
                "user_message": "hello",
                "expected_tools": ["read"],
                "expected_keywords": ["greeting"],
                "max_iterations": 5,
                "tags": ["fast"],
            },
            {
                "name": "case2",
                "user_message": "help",
            },
        ]
    }
    path = tmp_path / "test_cases.yaml"
    path.write_text(yaml.dump(data))
    return str(path)


@pytest.fixture
def yaml_dir(tmp_path):
    for i in range(2):
        data = [{"name": f"dir_case_{i}", "user_message": f"msg_{i}"}]
        (tmp_path / f"suite_{i}.yaml").write_text(yaml.dump(data))
    return str(tmp_path)


class TestYAMLLoader:
    async def test_load_file(self, yaml_file):
        loader = YAMLLoader()
        cases = await loader.load(yaml_file)
        assert len(cases) == 2
        assert cases[0].name == "case1"
        assert cases[0].expected_tools == ["read"]
        assert cases[1].name == "case2"

    async def test_load_directory(self, yaml_dir):
        loader = YAMLLoader()
        cases = await loader.load(yaml_dir)
        assert len(cases) == 2

    async def test_coding_harness_fields_round_trip(self, tmp_path):
        """Wave 1: repo/base_commit/setup_commands/test_command survive YAML loading."""
        data = {
            "cases": [
                {
                    "name": "coding",
                    "user_message": "fix the bug",
                    "repo": "/path/to/fixture",
                    "base_commit": "abc123",
                    "setup_commands": ["pip install -e ."],
                    "test_command": "python3 -m unittest discover -q",
                },
                {"name": "plain", "user_message": "hello"},
            ]
        }
        path = tmp_path / "coding_cases.yaml"
        path.write_text(yaml.dump(data))
        cases = await YAMLLoader().load(str(path))
        assert cases[0].repo == "/path/to/fixture"
        assert cases[0].base_commit == "abc123"
        assert cases[0].setup_commands == ["pip install -e ."]
        assert cases[0].test_command == "python3 -m unittest discover -q"
        # plain cases default to inert values
        assert cases[1].repo is None
        assert cases[1].setup_commands == []
        assert cases[1].test_command is None

    async def test_load_single_dict(self, tmp_path):
        data = {"name": "single", "user_message": "msg"}
        path = tmp_path / "single.yaml"
        path.write_text(yaml.dump(data))
        loader = YAMLLoader()
        cases = await loader.load(str(path))
        assert len(cases) == 1

    async def test_load_list_format(self, tmp_path):
        data = [{"name": "a", "user_message": "m1"}, {"name": "b", "user_message": "m2"}]
        path = tmp_path / "list.yaml"
        path.write_text(yaml.dump(data))
        loader = YAMLLoader()
        cases = await loader.load(str(path))
        assert len(cases) == 2

    async def test_load_invalid_yaml(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("- just a string")
        loader = YAMLLoader()
        cases = await loader.load(str(path))
        assert len(cases) == 0

    def test_framework_name(self):
        assert YAMLLoader().framework_name() == "yaml"


class TestLoaderRegistry:
    def test_register_and_get(self):
        register_default_loaders()
        loader = LoaderRegistry.get("yaml")
        assert isinstance(loader, YAMLLoader)

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown loader"):
            LoaderRegistry.get("nonexistent_loader")

    def test_list_available(self):
        register_default_loaders()
        available = LoaderRegistry.list_available()
        assert "yaml" in available

    async def test_load_via_registry(self, yaml_file):
        register_default_loaders()
        cases = await LoaderRegistry.load("yaml", yaml_file)
        assert len(cases) == 2


class TestRAGASDataGenerator:
    def test_init(self):
        client = MagicMock()
        gen = RAGASDataGenerator(client, chunk_size=500)
        assert gen.chunk_size == 500

    def test_parse_qa_response_valid(self):
        text = '[{"question": "What?", "answer": "This."}]'
        result = RAGASDataGenerator._parse_qa_response(text)
        assert len(result) == 1
        assert result[0]["question"] == "What?"

    def test_parse_qa_response_with_markdown(self):
        text = '```json\n[{"question": "Q?", "answer": "A."}]\n```'
        result = RAGASDataGenerator._parse_qa_response(text)
        assert len(result) == 1

    def test_parse_qa_response_empty(self):
        assert RAGASDataGenerator._parse_qa_response("[]") == []

    def test_parse_qa_response_invalid(self):
        assert RAGASDataGenerator._parse_qa_response("not json") == []

    def test_parse_qa_response_filters_invalid(self):
        text = '[{"question": "Q?", "answer": "A."}, {"no_question": true}]'
        result = RAGASDataGenerator._parse_qa_response(text)
        assert len(result) == 1

    def test_chunk_text(self):
        client = MagicMock()
        gen = RAGASDataGenerator(client, chunk_size=100)
        chunks = gen._chunk_text("Para one.\n\nPara two.\n\nPara three.")
        assert len(chunks) >= 1

    def test_chunk_text_empty(self):
        client = MagicMock()
        gen = RAGASDataGenerator(client, chunk_size=100)
        chunks = gen._chunk_text("")
        assert len(chunks) == 1

    async def test_generate_from_docs_nonexistent(self, tmp_path):
        client = MagicMock()
        gen = RAGASDataGenerator(client)
        cases = await gen.generate_from_docs([str(tmp_path / "nonexistent.md")])
        assert len(cases) == 0

    async def test_generate_from_docs_success(self, tmp_path):
        doc = tmp_path / "test.md"
        doc.write_text("This is test content about AI agents.")

        resp = MagicMock()
        resp.content = '[{"question": "What is this about?", "answer": "AI agents."}]'
        client = MagicMock()
        client.complete = AsyncMock(return_value=resp)
        gen = RAGASDataGenerator(client, chunk_size=200)

        cases = await gen.generate_from_docs([str(doc)], num_questions_per_doc=1)
        assert len(cases) == 1
        assert cases[0].user_message == "What is this about?"
        assert "ragas" in cases[0].tags

    async def test_generate_llm_failure(self, tmp_path):
        doc = tmp_path / "test.md"
        doc.write_text("Content here.")

        client = MagicMock()
        client.complete = AsyncMock(side_effect=Exception("LLM down"))
        gen = RAGASDataGenerator(client, chunk_size=200)

        cases = await gen.generate_from_docs([str(doc)])
        assert len(cases) == 0
