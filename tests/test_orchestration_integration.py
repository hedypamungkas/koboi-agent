"""Integration tests for config-driven orchestration."""

from __future__ import annotations

import json

import yaml

from koboi.config import Config
from koboi.types import AgentDef
from tests.conftest import MockClient, make_mock_response


def _write_config(tmp_path, config_data: dict) -> str:
    path = tmp_path / "test_config.yaml"
    with open(path, "w") as f:
        yaml.dump(config_data, f)
    return str(path)


# -- AgentDef parsing ---------------------------------------------------------


class TestParseAgentDefs:
    def test_parses_agent_list_from_config(self, tmp_path):
        from koboi.facade import _parse_agent_defs

        config_data = {
            "agent": {"name": "orchestrator"},
            "llm": {"model": "gpt-4"},
            "orchestration": {
                "enabled": True,
                "agents": [
                    {"name": "sales", "description": "Sales agent", "keywords": ["price", "buy"]},
                    {"name": "hr", "system_prompt": "You are HR."},
                ],
            },
        }
        path = _write_config(tmp_path, config_data)
        config = Config.from_yaml(path)
        defs = _parse_agent_defs(config)

        assert len(defs) == 2
        assert defs[0].name == "sales"
        assert defs[0].description == "Sales agent"
        assert defs[0].keywords == ["price", "buy"]
        assert defs[1].name == "hr"
        assert defs[1].system_prompt == "You are HR."

    def test_raises_when_no_agents(self, tmp_path):
        from koboi.facade import _parse_agent_defs

        config_data = {"agent": {"name": "test"}, "llm": {"model": "gpt-4"}, "orchestration": {"enabled": True}}
        path = _write_config(tmp_path, config_data)
        config = Config.from_yaml(path)

        import pytest

        with pytest.raises(ValueError, match="orchestration.agents must have at least one agent"):
            _parse_agent_defs(config)

    def test_parses_rag_config(self, tmp_path):
        from koboi.facade import _parse_agent_defs

        config_data = {
            "agent": {"name": "rag-orchestrator"},
            "llm": {"model": "gpt-4"},
            "orchestration": {
                "agents": [
                    {
                        "name": "doc_agent",
                        "rag": {
                            "documents": [{"path": "./data/sample/company_policy.md"}],
                        },
                    },
                ],
            },
        }
        path = _write_config(tmp_path, config_data)
        config = Config.from_yaml(path)
        defs = _parse_agent_defs(config)

        assert defs[0].rag_config is not None
        assert defs[0].rag_config["documents"][0]["path"] == "./data/sample/company_policy.md"


# -- Config-driven routers ----------------------------------------------------


class TestConfigDrivenRouters:
    async def test_keyword_router_with_agent_defs(self):
        from koboi.orchestration.router import KeywordRouter

        agent_defs = [
            AgentDef(name="sales", keywords=["price", "buy", "product"]),
            AgentDef(name="hr", keywords=["leave", "policy", "benefit"]),
        ]
        router = KeywordRouter(agent_defs=agent_defs)

        result = await router.route("What is the price?")
        assert "sales" in result.agents
        assert result.method == "keyword"

    async def test_keyword_router_custom_agents_only(self):
        from koboi.orchestration.router import KeywordRouter

        agent_defs = [
            AgentDef(name="custom_a", keywords=["alpha", "beta"]),
            AgentDef(name="custom_b", keywords=["gamma", "delta"]),
        ]
        router = KeywordRouter(agent_defs=agent_defs)

        result = await router.route("Tell me about alpha things")
        assert "custom_a" in result.agents
        assert "custom_b" not in result.agents

    async def test_llm_router_with_agent_defs(self):
        from koboi.orchestration.router import LLMRouter

        agent_defs = [
            AgentDef(name="sales", description="Handles sales"),
            AgentDef(name="support", description="Handles support"),
        ]
        resp = make_mock_response(json.dumps({"agents": ["support"], "confidence": 0.9, "reasoning": "support query"}))
        client = MockClient([resp])
        router = LLMRouter(client=client, agent_defs=agent_defs)

        result = await router.route("I need help with my account")
        assert "support" in result.agents
        assert result.method == "llm"

    async def test_llm_router_validates_agent_names(self):
        from koboi.orchestration.router import LLMRouter

        agent_defs = [
            AgentDef(name="alpha", keywords=["alpha"]),
            AgentDef(name="beta", keywords=["beta"]),
        ]
        # LLM returns unknown agent name -- should fall back to keyword
        resp = make_mock_response(json.dumps({"agents": ["unknown_agent"], "confidence": 0.9, "reasoning": "test"}))
        client = MockClient([resp])
        router = LLMRouter(client=client, agent_defs=agent_defs)

        result = await router.route("alpha test query")
        # Should fall back since "unknown_agent" is not in valid_names
        assert "alpha" in result.agents


# -- Router building from config ----------------------------------------------


class TestBuildRouter:
    def test_builds_keyword_router(self, tmp_path):
        from koboi.facade import _build_router, _parse_agent_defs

        config_data = {
            "agent": {"name": "keyword-orch"},
            "llm": {"model": "gpt-4"},
            "orchestration": {
                "enabled": True,
                "router": {"type": "keyword"},
                "agents": [
                    {"name": "sales", "keywords": ["price"]},
                ],
            },
        }
        path = _write_config(tmp_path, config_data)
        config = Config.from_yaml(path)
        agent_defs = _parse_agent_defs(config)
        client = MockClient([])

        router = _build_router(config, client, agent_defs)
        assert router.__class__.__name__ == "KeywordRouter"

    def test_builds_llm_router(self, tmp_path):
        from koboi.facade import _build_router, _parse_agent_defs

        config_data = {
            "agent": {"name": "llm-orch"},
            "llm": {"model": "gpt-4"},
            "orchestration": {
                "enabled": True,
                "router": {"type": "llm"},
                "agents": [{"name": "a"}],
            },
        }
        path = _write_config(tmp_path, config_data)
        config = Config.from_yaml(path)
        agent_defs = _parse_agent_defs(config)
        client = MockClient([])

        router = _build_router(config, client, agent_defs)
        assert router.__class__.__name__ == "LLMRouter"

    def test_builds_hybrid_router(self, tmp_path):
        from koboi.facade import _build_router, _parse_agent_defs

        config_data = {
            "agent": {"name": "hybrid-orch"},
            "llm": {"model": "gpt-4"},
            "orchestration": {
                "enabled": True,
                "router": {"type": "hybrid"},
                "agents": [{"name": "a"}],
            },
        }
        path = _write_config(tmp_path, config_data)
        config = Config.from_yaml(path)
        agent_defs = _parse_agent_defs(config)
        client = MockClient([])

        router = _build_router(config, client, agent_defs)
        assert router.__class__.__name__ == "HybridRouter"


# -- AgentFactory config-driven -----------------------------------------------


class TestConfigDrivenFactory:
    def test_create_configured_agent(self):
        from koboi.orchestration.factory import AgentFactory

        agent_def = AgentDef(
            name="test_agent",
            system_prompt="You are a test agent.",
        )
        resp = make_mock_response("Test response")
        client = MockClient([resp])

        agent = AgentFactory.create_configured_agent(agent_def, client)
        assert agent is not None
        # System prompt is stored in memory, not as a direct attribute
        assert agent.memory._system_prompt == "You are a test agent."

    def test_create_configured_agent_with_custom_tools(self):
        from koboi.orchestration.factory import AgentFactory

        agent_def = AgentDef(
            name="tool_agent",
            system_prompt="You have tools.",
            tools_config={"builtin": ["calculate"]},
        )
        resp = make_mock_response("Tool response")
        client = MockClient([resp])

        agent = AgentFactory.create_configured_agent(agent_def, client)
        assert agent is not None
        assert "calculate" in agent.tools._tools

    def test_create_all_configured(self):
        from koboi.orchestration.factory import AgentFactory

        agent_defs = [
            AgentDef(name="alpha", system_prompt="Agent alpha."),
            AgentDef(name="beta", system_prompt="Agent beta."),
        ]
        resp = make_mock_response("Response")
        client = MockClient([resp, resp])

        agents = AgentFactory.create_all_configured(agent_defs, client)
        assert len(agents) == 2
        assert "alpha" in agents
        assert "beta" in agents


# -- Orchestrator with agents_map ---------------------------------------------


class TestOrchestratorAgentsMap:
    async def test_uses_agents_map(self):
        from koboi.loop import AgentCore
        from koboi.orchestration.orchestrator import Orchestrator
        from koboi.orchestration.router import KeywordRouter

        agent_defs = [
            AgentDef(name="sales", keywords=["price"], system_prompt="Sales agent."),
        ]
        resp = make_mock_response("Sales answer")
        client = MockClient([resp])

        # Create agents manually
        sales_agent = AgentCore(client=client, system_prompt="Sales agent.")
        agents_map = {"sales": sales_agent}

        router = KeywordRouter(agent_defs=agent_defs)
        orch = Orchestrator(client=client, router=router, agents_map=agents_map)

        result = await orch.run("What is the price?")
        assert result.final_answer
        assert len(result.agent_results) >= 1


# -- Orchestrator run_stream --------------------------------------------------


class TestOrchestratorStream:
    async def test_stream_yields_events(self):
        from koboi.orchestration.orchestrator import Orchestrator
        from koboi.orchestration.router import KeywordRouter

        router = KeywordRouter()
        resp = make_mock_response("Final answer")
        client = MockClient([resp])
        orch = Orchestrator(client=client, router=router)

        events = []
        async for event in orch.run_stream("What is the price?"):
            events.append(event)

        event_types = [type(e).__name__ for e in events]
        assert "RoutingDecisionEvent" in event_types
        assert "AgentDispatchEvent" in event_types
        assert "AgentResultEvent" in event_types
        assert "OrchestrationCompleteEvent" in event_types

    async def test_stream_routing_event_fields(self):
        from koboi.events import RoutingDecisionEvent
        from koboi.orchestration.orchestrator import Orchestrator
        from koboi.orchestration.router import KeywordRouter

        router = KeywordRouter()
        resp = make_mock_response("Answer")
        client = MockClient([resp])
        orch = Orchestrator(client=client, router=router)

        async for event in orch.run_stream("annual leave policy"):
            if isinstance(event, RoutingDecisionEvent):
                assert event.agents
                assert event.confidence > 0
                assert event.method == "keyword"
                break


# -- Facade orchestration integration -----------------------------------------


class TestFacadeOrchestration:
    def test_orchestration_creates_orchestrator(self, tmp_path):
        from koboi.facade import KoboiAgent

        config_data = {
            "agent": {"name": "orch-test", "system_prompt": "You are an orchestrator."},
            "llm": {"model": "gpt-4o-mini", "api_key": "test-key", "base_url": "http://localhost:8080/v1"},
            "orchestration": {
                "enabled": True,
                "router": {"type": "keyword"},
                "execution": {"mode": "sequential"},
                "agents": [
                    {"name": "sales", "keywords": ["price"], "system_prompt": "Sales."},
                    {"name": "hr", "keywords": ["leave"], "system_prompt": "HR."},
                ],
            },
        }
        path = _write_config(tmp_path, config_data)
        agent = KoboiAgent.from_config(path)

        assert agent.orchestrator is not None
        assert agent.core is None  # Single core not created

    async def test_orchestration_run_returns_result(self, tmp_path):
        """Test that orchestration can run end-to-end with mock client."""
        from koboi.facade import KoboiAgent, _parse_agent_defs, _build_router
        from koboi.orchestration.orchestrator import Orchestrator
        from koboi.orchestration.factory import AgentFactory
        from koboi.types import RunResult

        config_data = {
            "agent": {"name": "orch-test", "system_prompt": "Synthesize.", "max_iterations": 3},
            "llm": {"model": "gpt-4o-mini", "api_key": "test-key", "base_url": "http://localhost:8080/v1"},
            "orchestration": {
                "enabled": True,
                "router": {"type": "keyword"},
                "agents": [
                    {"name": "sales", "keywords": ["price"], "system_prompt": "Sales."},
                ],
            },
        }
        path = _write_config(tmp_path, config_data)
        config = Config.from_yaml(path)

        # Use mock client for testing
        resp = make_mock_response("Sales answer about pricing")
        mock_client = MockClient([resp, resp])  # One for agent, one for synthesis

        agent_defs = _parse_agent_defs(config)
        router = _build_router(config, mock_client, agent_defs)
        agents_map = AgentFactory.create_all_configured(agent_defs, mock_client)

        orchestrator = Orchestrator(
            client=mock_client,
            router=router,
            agents_map=agents_map,
        )

        # Build KoboiAgent with orchestrator
        agent = KoboiAgent(core=None, config=config, orchestrator=orchestrator)

        result = await agent.run("What is the price?")

        assert isinstance(result, RunResult)
        assert result.content
        assert result.metadata["routing_method"] == "keyword"

    def test_deep_research_mode_wires_system_prompt_into_orchestrator(self, tmp_path):
        """Fix: agent.system_prompt now reaches the Orchestrator for deep_research
        mode (facade.py previously dropped it entirely for this exec_mode)."""
        from koboi.facade import KoboiAgent

        config_data = {
            "agent": {"name": "insights", "system_prompt": "Balas ringkas dalam Bahasa Indonesia."},
            "llm": {"model": "gpt-4o-mini", "api_key": "test-key", "base_url": "http://localhost:8080/v1"},
            "orchestration": {
                "enabled": True,
                "router": {"type": "keyword"},
                "execution": {"mode": "deep_research"},
            },
        }
        path = _write_config(tmp_path, config_data)
        agent = KoboiAgent.from_config(path)

        assert agent.orchestrator is not None
        assert agent.orchestrator._system_prompt == "Balas ringkas dalam Bahasa Indonesia."

    def test_no_orchestration_when_not_enabled(self, tmp_path):
        from koboi.facade import KoboiAgent

        config_data = {
            "agent": {"name": "simple", "system_prompt": "Hello."},
            "llm": {"model": "gpt-4o-mini", "api_key": "test-key", "base_url": "http://localhost:8080/v1"},
        }
        path = _write_config(tmp_path, config_data)
        agent = KoboiAgent.from_config(path)

        assert agent.orchestrator is None
        assert agent.core is not None

    def test_orchestration_from_dict(self):
        from koboi.facade import KoboiAgent

        config_data = {
            "agent": {"name": "dict-orch", "system_prompt": "Orchestrate."},
            "llm": {"model": "gpt-4o-mini", "api_key": "test-key", "base_url": "http://localhost:8080/v1"},
            "orchestration": {
                "enabled": True,
                "router": {"type": "keyword"},
                "agents": [
                    {"name": "a", "keywords": ["test"], "system_prompt": "Agent A."},
                ],
            },
        }
        agent = KoboiAgent.from_dict(config_data)
        assert agent.orchestrator is not None

    async def test_orchestration_close_cleans_up(self, tmp_path):
        from koboi.facade import KoboiAgent

        config_data = {
            "agent": {"name": "close-test", "system_prompt": "Test."},
            "llm": {"model": "gpt-4o-mini", "api_key": "test-key", "base_url": "http://localhost:8080/v1"},
            "orchestration": {
                "enabled": True,
                "router": {"type": "keyword"},
                "agents": [
                    {"name": "a", "keywords": ["test"], "system_prompt": "Agent A."},
                ],
            },
        }
        path = _write_config(tmp_path, config_data)
        agent = KoboiAgent.from_config(path)

        await agent.close()
        # Should not raise

    def test_orchestration_builds_dedicated_per_agent_client(self):
        # The real _agent_client_builder closure (facade._build_orchestration) must
        # build a dedicated client for an agent whose llm_config carries overrides,
        # apply them, and keep max_context_tokens on the agent (not leak it into the
        # client). Exercises the true path end-to-end (not a fake builder).
        from koboi.facade import KoboiAgent

        config_data = {
            "agent": {"name": "dedicated-orch", "system_prompt": "Orchestrate."},
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "test-key",
                "base_url": "http://localhost:8080/v1",
                "temperature": 0.9,
            },
            "orchestration": {
                "enabled": True,
                "router": {"type": "keyword"},
                "agents": [
                    {
                        "name": "worker",
                        "system_prompt": "x",
                        "keywords": ["thing"],
                        "llm": {"temperature": 0.1, "max_tokens": 1234, "max_context_tokens": 4000},
                    },
                ],
            },
        }
        agent = KoboiAgent.from_dict(config_data)
        orch = agent.orchestrator
        worker = orch._agents_map["worker"]
        shared = orch.client

        assert worker.client is not shared  # dedicated client built by the real closure
        assert worker.client._impl._temperature == 0.1  # override applied
        assert worker.client._impl._max_tokens == 1234  # override applied
        assert worker.max_context_tokens == 4000  # consumed by agent, not leaked to client


# -- End-to-end orchestrated config -------------------------------------------


class TestOrchestratedConfig:
    def test_config_loads_orchestration_section(self, tmp_path):
        config_data = {
            "agent": {"name": "orch"},
            "llm": {"model": "gpt-4o-mini", "api_key": "test-key"},
            "orchestration": {
                "enabled": True,
                "router": {"type": "keyword"},
                "execution": {"mode": "sequential"},
                "agents": [
                    {"name": "sales", "keywords": ["price"], "system_prompt": "Sales."},
                    {"name": "hr", "keywords": ["leave"], "system_prompt": "HR."},
                ],
            },
        }
        path = _write_config(tmp_path, config_data)
        config = Config.from_yaml(path)

        assert config.orchestration.get("enabled") is True
        assert config.orchestration["router"]["type"] == "keyword"
        assert len(config.orchestration["agents"]) == 2

    def test_config_builder_orchestration(self):
        from koboi.config import ConfigBuilder

        builder = (
            ConfigBuilder()
            .agent(name="test")
            .llm(model="gpt-4o-mini", api_key="key")
            .orchestration(
                enabled=True,
                router_type="keyword",
                execution_mode="sequential",
                agents=[
                    {"name": "sales", "keywords": ["price"]},
                ],
            )
        )
        config = builder.build()

        assert config.orchestration.get("enabled") is True
        assert config.orchestration["router"]["type"] == "keyword"
        assert len(config.orchestration["agents"]) == 1
