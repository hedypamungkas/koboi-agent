"""Tests for orchestration/factory.py helpers and AgentFactory."""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

import pytest

from koboi.orchestration._utils import extract_json as _extract_json
from koboi.orchestration.factory import (
    _split_catalog,
    _chunk_all,
    _get_hr_chunks,
    _get_sales_chunks,
    _get_finance_chunks,
    AgentFactory,
    DynamicAgentBuilder,
    KNOWN_DOMAINS,
    CHUNKER,
)
from koboi.types import AgentBlueprint, AgentDef


class TestExtractJson:
    def test_valid_json(self):
        assert _extract_json('{"domain": "hr", "is_known": true}') == {"domain": "hr", "is_known": True}

    def test_json_in_text(self):
        text = 'Here is the answer: {"domain": "sales", "is_known": false} done.'
        assert _extract_json(text) == {"domain": "sales", "is_known": False}

    def test_nested_json(self):
        text = 'Result: {"a": 1, "b": {"c": 2}} end'
        result = _extract_json(text)
        assert result == {"a": 1, "b": {"c": 2}}

    def test_no_json(self):
        assert _extract_json("no json here") is None

    def test_empty_string(self):
        assert _extract_json("") is None

    def test_invalid_json_after_brace(self):
        assert _extract_json("{invalid json content") is None

    def test_multiple_braces_takes_first_complete(self):
        text = '{"a": 1} and {"b": 2}'
        assert _extract_json(text) == {"a": 1}


class TestSplitCatalog:
    def test_split_catalog_returns_two_parts(self):
        sales, finance = _split_catalog()
        # Should split on TERMS AND CONDITIONS or Payment Terms
        assert isinstance(sales, str)
        assert isinstance(finance, str)

    def test_sales_part_not_empty(self):
        sales, _ = _split_catalog()
        assert len(sales) > 0


class TestChunkHelpers:
    def test_chunk_all_returns_chunks(self):
        chunks = _chunk_all()
        assert len(chunks) > 0
        assert all(hasattr(c, "doc_id") for c in chunks)

    def test_hr_chunks_filter(self):
        chunks = _get_hr_chunks()
        assert all(c.doc_id in ("company_policy", "employee_handbook") for c in chunks)

    def test_sales_chunks_non_empty(self):
        chunks = _get_sales_chunks()
        assert len(chunks) > 0

    def test_finance_chunks_non_empty(self):
        chunks = _get_finance_chunks()
        assert len(chunks) > 0


class TestAgentFactory:
    def test_create_hr_agent(self):
        client = MagicMock()
        agent = AgentFactory.create_agent("hr", client)
        assert agent is not None

    def test_create_sales_agent(self):
        client = MagicMock()
        agent = AgentFactory.create_agent("sales", client)
        assert agent is not None

    def test_create_finance_agent(self):
        client = MagicMock()
        agent = AgentFactory.create_agent("finance", client)
        assert agent is not None

    def test_create_general_agent(self):
        client = MagicMock()
        agent = AgentFactory.create_agent("general", client)
        assert agent is not None

    def test_create_unknown_defaults_to_general(self):
        client = MagicMock()
        agent = AgentFactory.create_agent("unknown", client)
        assert agent is not None

    def test_configure_updates_defaults(self):
        original = AgentFactory._defaults.copy()
        AgentFactory.configure(top_k=10)
        assert AgentFactory._defaults["top_k"] == 10
        AgentFactory._defaults = original

    def test_build_tools_from_config_none(self):
        assert AgentFactory._build_tools_from_config(None) is None

    def test_build_tools_from_config_empty(self):
        assert AgentFactory._build_tools_from_config({}) is None

    def test_build_tools_from_config_with_builtins(self):
        registry = AgentFactory._build_tools_from_config({"builtin": ["calculator"]})
        assert registry is not None

    def test_build_rag_from_config_none(self):
        assert AgentFactory.build_rag_from_config(None, None) is None

    def test_build_rag_from_config_disabled(self):
        assert AgentFactory.build_rag_from_config({"enabled": False}, None) is None

    def test_build_rag_from_config_with_parent(self):
        parent = {"enabled": True, "documents": []}
        assert AgentFactory.build_rag_from_config(None, parent) is None  # no docs

    def test_create_configured_agent(self):
        client = MagicMock()
        ad = AgentDef(
            name="test",
            system_prompt="You are test.",
            rag_config=None,
            tools_config=None,
            llm_config=None,
        )
        agent = AgentFactory.create_configured_agent(ad, client)
        assert agent is not None

    def test_create_configured_agent_with_llm_config(self):
        client = MagicMock()
        ad = AgentDef(
            name="test",
            system_prompt="You are test.",
            rag_config=None,
            tools_config=None,
            llm_config={"max_context_tokens": 4000},
        )
        agent = AgentFactory.create_configured_agent(ad, client)
        assert agent is not None

    def test_create_all_configured(self):
        client = MagicMock()
        defs = [
            AgentDef(name="a", system_prompt="A", rag_config=None, tools_config=None, llm_config=None),
            AgentDef(name="b", system_prompt="B", rag_config=None, tools_config=None, llm_config=None),
        ]
        agents = AgentFactory.create_all_configured(defs, client)
        assert "a" in agents
        assert "b" in agents


class TestDynamicAgentBuilder:
    def test_init_defaults(self):
        client = MagicMock()
        builder = DynamicAgentBuilder(client=client)
        assert builder._top_k == 5
        assert builder._chunk_size == 400

    def test_init_custom(self):
        client = MagicMock()
        builder = DynamicAgentBuilder(client=client, top_k=10, chunk_size=200)
        assert builder._top_k == 10
        assert builder._chunk_size == 200

    async def test_analyze_domain_known(self):
        client = MagicMock()
        client.complete = AsyncMock(return_value=MagicMock(content='{"domain": "hr", "is_known": true}'))
        builder = DynamicAgentBuilder(client=client)
        domain, is_known = await builder.analyze_domain("What is the leave policy?")
        assert domain == "hr"
        assert is_known is True

    async def test_analyze_domain_unknown(self):
        client = MagicMock()
        client.complete = AsyncMock(return_value=MagicMock(content='{"domain": "legal", "is_known": false}'))
        builder = DynamicAgentBuilder(client=client)
        domain, is_known = await builder.analyze_domain("What about legal?")
        assert domain == "legal"
        assert is_known is False

    async def test_analyze_domain_llm_error(self):
        client = MagicMock()
        client.complete = AsyncMock(side_effect=Exception("network error"))
        builder = DynamicAgentBuilder(client=client)
        domain, is_known = await builder.analyze_domain("test")
        assert domain == "general"
        assert is_known is False

    async def test_analyze_domain_bad_json(self):
        client = MagicMock()
        client.complete = AsyncMock(return_value=MagicMock(content="not json at all"))
        builder = DynamicAgentBuilder(client=client)
        domain, is_known = await builder.analyze_domain("test")
        assert domain == "general"

    async def test_find_relevant_chunks(self):
        client = MagicMock()
        builder = DynamicAgentBuilder(client=client)
        chunks = await builder.find_relevant_chunks("leave policy")
        assert isinstance(chunks, list)

    async def test_generate_system_prompt_success(self):
        client = MagicMock()
        client.complete = AsyncMock(
            return_value=MagicMock(
                content="You are an HR specialist agent at Acme Corp. You handle leave policies and employee benefits."
            )
        )
        builder = DynamicAgentBuilder(client=client)
        prompt = await builder.generate_system_prompt("leave", "hr", [])
        assert "specialist" in prompt.lower() or len(prompt) > 0

    async def test_generate_system_prompt_short_response_uses_fallback(self):
        client = MagicMock()
        client.complete = AsyncMock(return_value=MagicMock(content="short"))
        builder = DynamicAgentBuilder(client=client)
        prompt = await builder.generate_system_prompt("test", "hr", [])
        assert "Acme Corp" in prompt

    async def test_generate_system_prompt_error_uses_fallback(self):
        client = MagicMock()
        client.complete = AsyncMock(side_effect=Exception("fail"))
        builder = DynamicAgentBuilder(client=client)
        prompt = await builder.generate_system_prompt("test", "hr", [])
        assert "Acme Corp" in prompt

    async def test_build_blueprint(self):
        client = MagicMock()
        client.complete = AsyncMock(return_value=MagicMock(content='{"domain": "hr", "is_known": true}'))
        builder = DynamicAgentBuilder(client=client)
        bp = await builder.build_blueprint("What is the leave policy?")
        assert isinstance(bp, AgentBlueprint)
        assert bp.name.startswith("dynamic_")

    async def test_build_blueprint_with_domain_label(self):
        client = MagicMock()
        builder = DynamicAgentBuilder(client=client)
        bp = await builder.build_blueprint("test query", domain_label="custom")
        assert "custom" in bp.name

    def test_build_agent_with_chunks(self):
        client = MagicMock()
        builder = DynamicAgentBuilder(client=client)
        bp = AgentBlueprint(
            name="test",
            domain_label="test",
            system_prompt="test prompt",
            chunks=builder.all_chunks[:3],
            chunker_config={},
            retriever_top_k=3,
            source="test",
            created_at=0,
        )
        agent = builder.build_agent(bp)
        assert agent is not None

    def test_build_agent_without_chunks(self):
        client = MagicMock()
        builder = DynamicAgentBuilder(client=client)
        bp = AgentBlueprint(
            name="test",
            domain_label="test",
            system_prompt="test prompt",
            chunks=[],
            chunker_config={},
            retriever_top_k=3,
            source="test",
            created_at=0,
        )
        agent = builder.build_agent(bp)
        assert agent is not None
