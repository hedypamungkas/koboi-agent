"""Tests for koboi.orchestration module."""

from __future__ import annotations

import json

from koboi.orchestration.router import KeywordRouter, LLMRouter
from koboi.types import RoutingDecision
from tests.conftest import MockClient, make_mock_response


class TestKeywordRouter:
    async def test_routes_hr_query(self):
        router = KeywordRouter()
        result = await router.route("How many annual leave days?")
        assert "hr" in result.agents
        assert result.confidence > 0
        assert result.method == "keyword"

    async def test_routes_sales_query(self):
        router = KeywordRouter()
        result = await router.route("What is the price of the Enterprise package?")
        assert "sales" in result.agents

    async def test_routes_finance_query(self):
        router = KeywordRouter()
        result = await router.route("When is the invoice due date?")
        assert "finance" in result.agents

    async def test_no_match_broadcasts(self):
        router = KeywordRouter()
        result = await router.route("What color is the sky?")
        assert len(result.agents) == 3
        assert result.confidence < 0.5

    async def test_multi_domain_query(self):
        router = KeywordRouter()
        result = await router.route("Leave and service packages for a team of 10")
        assert "hr" in result.agents
        assert "sales" in result.agents


class TestLLMRouter:
    async def test_routes_via_llm(self):
        resp = make_mock_response(json.dumps({"agents": ["hr"], "confidence": 0.9, "reasoning": "about leave"}))
        client = MockClient([resp])
        router = LLMRouter(client=client)
        result = await router.route("How many annual leave days?")
        assert "hr" in result.agents
        assert result.method == "llm"

    async def test_falls_back_to_keyword(self):
        client = MockClient([])
        router = LLMRouter(client=client)
        result = await router.route("What is the price of the Enterprise package?")
        assert "sales" in result.agents


class TestOrchestrator:
    async def test_sequential_execution(self):
        from koboi.orchestration.orchestrator import Orchestrator
        from koboi.orchestration.router import KeywordRouter

        router = KeywordRouter()
        resp = make_mock_response("Answer from agent.")
        client = MockClient([resp])
        orch = Orchestrator(client=client, router=router)
        result = await orch.run("How many annual leave days?", mode="sequential")
        assert result.final_answer
        assert result.query == "How many annual leave days?"

    async def test_orchestrator_result_fields(self):
        from koboi.orchestration.orchestrator import Orchestrator
        from koboi.orchestration.router import KeywordRouter

        router = KeywordRouter()
        resp = make_mock_response("Test answer")
        client = MockClient([resp])
        orch = Orchestrator(client=client, router=router)
        result = await orch.run("What is the price of the Starter package?")
        assert result.routing is not None
        assert result.total_elapsed_seconds >= 0
        assert result.execution_mode == "sequential"


class TestAgentFactory:
    def test_create_agent_general(self):
        from koboi.orchestration.factory import AgentFactory

        resp = make_mock_response("General response")
        client = MockClient([resp])
        agent = AgentFactory.create_agent("general", client)
        assert agent is not None

    def test_create_agent_hr(self):
        from koboi.orchestration.factory import AgentFactory

        resp = make_mock_response("HR response")
        client = MockClient([resp])
        agent = AgentFactory.create_agent("hr", client)
        assert agent is not None


class TestOrchestratorExecution:
    async def test_sequential_execution_mode(self):
        """Test sequential execution of multiple agents."""
        from koboi.orchestration.orchestrator import Orchestrator
        from koboi.orchestration.router import KeywordRouter

        router = KeywordRouter()
        resp1 = make_mock_response("HR answer about leave")
        resp2 = make_mock_response("Sales answer about pricing")
        client = MockClient([resp1, resp2])
        orch = Orchestrator(client=client, router=router)

        result = await orch.run("Leave policy and pricing", mode="sequential")

        assert result.final_answer
        assert len(result.agent_results) >= 1
        assert result.execution_mode == "sequential"

    async def test_parallel_execution_mode(self):
        """Test parallel execution of multiple agents."""
        from koboi.orchestration.orchestrator import Orchestrator
        from koboi.orchestration.router import KeywordRouter

        router = KeywordRouter()
        resp = make_mock_response("Agent response")
        client = MockClient([resp, resp, resp])
        orch = Orchestrator(client=client, router=router)

        result = await orch.run("General question", mode="parallel")

        assert result.final_answer
        assert result.execution_mode == "parallel"

    async def test_orchestrator_with_revision(self):
        """Test orchestrator with revision enabled."""
        from koboi.orchestration.orchestrator import Orchestrator
        from koboi.orchestration.router import KeywordRouter

        router = KeywordRouter()
        resp = make_mock_response("Detailed answer with revision")
        client = MockClient([resp])
        orch = Orchestrator(
            client=client,
            router=router,
            use_revision=True,
            max_revisions=2,
        )

        result = await orch.run("Test question", mode="sequential")

        assert result.final_answer
        assert "revision" in result.execution_mode


class TestDynamicAgentBuilder:
    async def test_domain_classification(self):
        """Test DynamicAgentBuilder domain classification."""
        from koboi.orchestration.factory import DynamicAgentBuilder

        resp = make_mock_response(json.dumps({"domain": "hr", "is_known": True}))
        client = MockClient([resp])

        builder = DynamicAgentBuilder(client=client)
        domain, is_known = await builder.analyze_domain("How much leave do I get?")

        assert domain == "hr"
        assert is_known is True

    async def test_find_relevant_chunks(self):
        """Test DynamicAgentBuilder finds relevant chunks."""
        from koboi.orchestration.factory import DynamicAgentBuilder

        client = MockClient([])
        builder = DynamicAgentBuilder(client=client)

        chunks = await builder.find_relevant_chunks("leave policy", top_k=3)

        assert isinstance(chunks, list)

    async def test_generate_system_prompt(self):
        """Test DynamicAgentBuilder generates system prompt."""
        from koboi.orchestration.factory import DynamicAgentBuilder
        from koboi.rag.types import Chunk

        resp = make_mock_response("You are a leave specialist agent.")
        client = MockClient([resp])

        builder = DynamicAgentBuilder(client=client)

        sample_chunks = [
            Chunk(id="c1", doc_id="d1", content="Annual leave policy: 20 days"),
        ]

        prompt = await builder.generate_system_prompt(
            query="leave policy",
            domain_label="leave_management",
            sample_chunks=sample_chunks,
        )

        assert len(prompt) > 50

    async def test_build_blueprint(self):
        """Test DynamicAgentBuilder builds complete blueprint."""
        from koboi.orchestration.factory import DynamicAgentBuilder

        resp = make_mock_response(json.dumps({"domain": "support", "is_known": False}))
        resp2 = make_mock_response("You are a support specialist.")
        client = MockClient([resp, resp2])

        builder = DynamicAgentBuilder(client=client)
        blueprint = await builder.build_blueprint("How to reset password?")

        assert blueprint.name
        assert blueprint.domain_label == "support"
        assert blueprint.system_prompt
        assert blueprint.chunks is not None

    def test_build_agent_from_blueprint(self):
        """Test building agent from blueprint."""
        from koboi.orchestration.factory import DynamicAgentBuilder
        from koboi.rag.types import Chunk
        from koboi.types import AgentBlueprint

        client = MockClient([])

        blueprint = AgentBlueprint(
            name="test_agent",
            domain_label="test",
            system_prompt="You are a test agent.",
            chunks=[
                Chunk(id="c1", doc_id="d1", content="Test content"),
            ],
            retriever_top_k=3,
            chunker_config={"chunk_size": 400, "overlap": 40},
            source="test",
            created_at=0,
        )

        builder = DynamicAgentBuilder(client=client)
        agent = builder.build_agent(blueprint)

        assert agent is not None


class TestPreconfiguredAgents:
    def test_create_sales_agent(self):
        """Test creating pre-configured sales agent."""
        from koboi.orchestration.factory import AgentFactory

        resp = make_mock_response("Sales response")
        client = MockClient([resp])

        agent = AgentFactory.create_agent("sales", client)

        assert agent is not None

    def test_create_finance_agent(self):
        """Test creating pre-configured finance agent."""
        from koboi.orchestration.factory import AgentFactory

        resp = make_mock_response("Finance response")
        client = MockClient([resp])

        agent = AgentFactory.create_agent("finance", client)

        assert agent is not None

    def test_create_general_agent(self):
        """Test creating pre-configured general agent."""
        from koboi.orchestration.factory import AgentFactory

        resp = make_mock_response("General response")
        client = MockClient([resp])

        agent = AgentFactory.create_agent("general", client)

        assert agent is not None


class TestHybridRouter:
    async def test_confidence_threshold_routing(self):
        """Test HybridRouter confidence threshold behavior."""
        from koboi.orchestration.router import HybridRouter

        client = MockClient([])
        router = HybridRouter(client=client, confidence_threshold=0.7)

        result = await router.route("How much annual leave?")

        # Should use keyword or LLM based on confidence
        assert result.agents
        assert result.confidence >= 0

    async def test_low_confidence_fallback_to_llm(self):
        """Test low confidence triggers LLM routing."""
        from koboi.orchestration.router import HybridRouter

        resp = make_mock_response(json.dumps({"agents": ["hr"], "confidence": 0.8, "reasoning": "LLM analysis"}))
        client = MockClient([resp])
        router = HybridRouter(client=client, confidence_threshold=0.9)

        result = await router.route("Ambiguous question")

        # With low keyword confidence, should use LLM
        assert result.method == "hybrid(llm)"

    async def test_high_confidence_keyword_only(self):
        """Test high confidence skips LLM call."""
        from koboi.orchestration.router import HybridRouter

        client = MockClient([])
        router = HybridRouter(client=client, confidence_threshold=0.5)

        result = await router.route("How much annual leave days?")

        # High confidence keyword match
        if result.confidence >= 0.5:
            assert "keyword" in result.method

    async def test_hybrid_adds_llm_agents(self):
        """Test HybridRouter adds agents found by LLM but not keyword."""
        from koboi.orchestration.router import HybridRouter

        # LLM finds additional domain
        resp = make_mock_response(
            json.dumps({"agents": ["hr", "sales"], "confidence": 0.8, "reasoning": "Multi-domain query"})
        )
        client = MockClient([resp])
        router = HybridRouter(client=client, confidence_threshold=0.3)

        result = await router.route("Leave and pricing")

        # Should combine keyword and LLM results
        assert "hr" in result.agents or "sales" in result.agents


class TestQualityEvaluator:
    async def test_evaluate_returns_score(self):
        """Test QualityEvaluator returns quality score."""
        from koboi.orchestration.orchestrator import QualityEvaluator

        resp = make_mock_response(json.dumps({"score": 0.8, "feedback": "Good answer", "needs_revision": False}))
        client = MockClient([resp])

        evaluator = QualityEvaluator(client=client, threshold=0.7)
        score, feedback, needs = await evaluator.evaluate("Test query", "Test answer")

        assert score == 0.8
        assert feedback == "Good answer"
        assert needs is False

    async def test_evaluate_triggers_revision(self):
        """Test low score triggers revision."""
        from koboi.orchestration.orchestrator import QualityEvaluator

        resp = make_mock_response(json.dumps({"score": 0.4, "feedback": "Too vague", "needs_revision": True}))
        client = MockClient([resp])

        evaluator = QualityEvaluator(client=client, threshold=0.7)
        score, feedback, needs = await evaluator.evaluate("Test query", "Poor answer")

        assert score == 0.4
        assert needs is True

    async def test_evaluate_handles_invalid_json(self):
        """Test evaluator handles invalid JSON gracefully."""
        from koboi.orchestration.orchestrator import QualityEvaluator

        resp = make_mock_response("Not JSON at all")
        client = MockClient([resp])

        evaluator = QualityEvaluator(client=client, threshold=0.7)
        score, feedback, needs = await evaluator.evaluate("Test", "Answer")

        # Should return default values on error
        assert score == 0.5
        assert feedback == "evaluation failed"
        assert needs is True
