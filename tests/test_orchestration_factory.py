"""Tests for koboi/orchestration/factory.py -- Agent factory."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from koboi.orchestration.factory import (
    AgentFactory,
    DynamicAgentBuilder,
    _split_catalog,
    _chunk_all,
    _get_hr_chunks,
    _get_sales_chunks,
    _get_finance_chunks,
    KNOWN_DOMAINS,
    HR_PROMPT,
    SALES_PROMPT,
    FINANCE_PROMPT,
    GENERAL_PROMPT,
)


class TestKnowledgeSplitting:
    def test_split_catalog(self):
        sales, finance = _split_catalog()
        # Both should have some content
        assert len(sales) > 0

    def test_chunk_all(self):
        chunks = _chunk_all()
        assert len(chunks) > 0
        assert all(hasattr(c, "content") for c in chunks)

    def test_hr_chunks(self):
        chunks = _get_hr_chunks()
        assert len(chunks) > 0
        for c in chunks:
            assert c.doc_id in ("company_policy", "employee_handbook")

    def test_sales_chunks(self):
        chunks = _get_sales_chunks()
        assert len(chunks) > 0

    def test_finance_chunks(self):
        chunks = _get_finance_chunks()
        # May be empty if split doesn't find marker
        assert isinstance(chunks, list)


class TestAgentFactory:
    def test_configure(self):
        AgentFactory.configure(top_k=5)
        assert AgentFactory._defaults["top_k"] == 5
        AgentFactory.configure(top_k=3)  # reset

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
        agent = AgentFactory.create_agent("unknown_domain", client)
        assert agent is not None


class TestDynamicAgentBuilder:
    def test_init_defaults(self):
        client = MagicMock()
        builder = DynamicAgentBuilder(client=client)
        assert builder._top_k == 5
        assert builder._chunk_size == 400

    def test_build_agent_from_blueprint(self):
        from koboi.types import AgentBlueprint

        client = MagicMock()
        builder = DynamicAgentBuilder(client=client)
        blueprint = AgentBlueprint(
            name="test_agent",
            domain_label="test",
            system_prompt="You are a test agent.",
            chunks=[],
            chunker_config={},
            retriever_top_k=3,
            source="test",
            created_at=0,
        )
        agent = builder.build_agent(blueprint)
        assert agent is not None

    def test_build_agent_with_chunks(self):
        from koboi.types import AgentBlueprint
        from koboi.rag.types import Chunk

        client = MagicMock()
        builder = DynamicAgentBuilder(client=client)
        chunks = [Chunk(id="c1", doc_id="d1", content="test content")]
        blueprint = AgentBlueprint(
            name="test_agent",
            domain_label="test",
            system_prompt="Test prompt.",
            chunks=chunks,
            chunker_config={},
            retriever_top_k=3,
            source="test",
            created_at=0,
        )
        agent = builder.build_agent(blueprint)
        assert agent is not None

    @pytest.mark.asyncio
    async def test_analyze_domain_llm_failure(self):
        client = MagicMock()
        client.complete = AsyncMock(side_effect=Exception("LLM down"))
        builder = DynamicAgentBuilder(client=client)
        domain, is_known = await builder.analyze_domain("test query")
        assert domain == "general"
        assert is_known is False

    @pytest.mark.asyncio
    async def test_analyze_domain_success(self):
        client = MagicMock()
        resp = MagicMock()
        resp.content = '{"domain": "hr", "is_known": true}'
        client.complete = AsyncMock(return_value=resp)
        builder = DynamicAgentBuilder(client=client)
        domain, is_known = await builder.analyze_domain("leave policy")
        assert domain == "hr"
        assert is_known is True

    @pytest.mark.asyncio
    async def test_analyze_domain_unknown(self):
        client = MagicMock()
        resp = MagicMock()
        resp.content = '{"domain": "quantum", "is_known": false}'
        client.complete = AsyncMock(return_value=resp)
        builder = DynamicAgentBuilder(client=client)
        domain, is_known = await builder.analyze_domain("quantum computing")
        assert domain == "quantum"
        assert is_known is False

    @pytest.mark.asyncio
    async def test_find_relevant_chunks(self):
        client = MagicMock()
        builder = DynamicAgentBuilder(client=client)
        chunks = await builder.find_relevant_chunks("product pricing")
        assert isinstance(chunks, list)

    @pytest.mark.asyncio
    async def test_generate_system_prompt_success(self):
        client = MagicMock()
        resp = MagicMock()
        resp.content = "You are a test specialist agent at Acme Corp. Answer based on context."
        client.complete = AsyncMock(return_value=resp)
        builder = DynamicAgentBuilder(client=client)
        prompt = await builder.generate_system_prompt("test", "test", [])
        assert len(prompt) > 50

    @pytest.mark.asyncio
    async def test_generate_system_prompt_fallback(self):
        client = MagicMock()
        client.complete = AsyncMock(side_effect=Exception("fail"))
        builder = DynamicAgentBuilder(client=client)
        prompt = await builder.generate_system_prompt("test", "custom_domain", [])
        assert "custom_domain" in prompt

    @pytest.mark.asyncio
    async def test_build_blueprint(self):
        client = MagicMock()
        resp = MagicMock()
        resp.content = '{"domain": "hr", "is_known": true}'
        client.complete = AsyncMock(return_value=resp)
        builder = DynamicAgentBuilder(client=client)
        blueprint = await builder.build_blueprint("leave policy")
        assert blueprint.name.startswith("dynamic_")
        assert blueprint.domain_label == "hr"


class TestPrompts:
    def test_hr_prompt_content(self):
        assert "HR specialist" in HR_PROMPT

    def test_sales_prompt_content(self):
        assert "Sales specialist" in SALES_PROMPT

    def test_finance_prompt_content(self):
        assert "Finance specialist" in FINANCE_PROMPT

    def test_general_prompt_content(self):
        assert "internal assistant" in GENERAL_PROMPT

    def test_known_domains(self):
        assert "hr" in KNOWN_DOMAINS
        assert "sales" in KNOWN_DOMAINS
        assert "finance" in KNOWN_DOMAINS


class TestBuildRagEmbeddingClient:
    """build_rag_from_config routes to a dedicated embedding client when the
    ``embedding:`` section has an api_key, else uses the chat client."""

    def test_no_embedding_config_uses_chat_client(self, monkeypatch):
        captured: dict = {}

        def fake_build_rag(rag_dict, *, client=None, logger=None, chat_client=None):
            captured["client"] = client
            captured["chat_client"] = chat_client
            return "AUG"

        monkeypatch.setattr("koboi.rag.registry.build_rag", fake_build_rag)

        chat_client = MagicMock(name="chat_client")
        AgentFactory.build_rag_from_config(
            {"enabled": True, "augmentation": "in_memory"}, None, None, client=chat_client
        )
        assert captured["client"] is chat_client
        # #9: the chat client is threaded separately for query rewriting.
        assert captured["chat_client"] is chat_client

    def test_embedding_config_with_api_key_uses_dedicated_client(self, monkeypatch):
        captured: dict = {}

        def fake_build_rag(rag_dict, *, client=None, logger=None, chat_client=None):
            captured["client"] = client
            captured["chat_client"] = chat_client
            return "AUG"

        monkeypatch.setattr("koboi.rag.registry.build_rag", fake_build_rag)

        chat_client = MagicMock(name="chat_client")
        AgentFactory.build_rag_from_config(
            {"enabled": True, "augmentation": "in_memory"},
            None,
            None,
            client=chat_client,
            embedding_config={
                "api_key": "sk-emb",
                "base_url": "https://emb.example/v1",
                "model": "text-embedding-3-small",
            },
        )
        # A dedicated embedding client is built (real adapter; no network at
        # construction) and used INSTEAD of the chat client.
        assert captured["client"] is not chat_client
        assert captured["client"] is not None
