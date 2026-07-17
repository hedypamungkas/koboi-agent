"""tests/test_self_healing_p4.py -- CRITIC tool-verification + self-consistency (P4).

Part A: ReflectionHook._tool_verify_claims (math mismatch via calculate; no-tools fallback).
Part B: aggregate_structured (majority, single, invalid-JSON fallback, usage sum) +
        _self_consistency_applies gating + an AgentCore integration (structured terminal).
"""

from __future__ import annotations

from koboi.hooks.reflection_hook import ReflectionHook
from koboi.loop import AgentCore
from koboi.memory import ConversationMemory
from koboi.self_consistency import aggregate_structured
from koboi.tools.registry import ToolRegistry
from koboi.types import AgentResponse, TokenUsage
from tests.conftest import make_mock_response, make_tool_registry


class _DecomposeCritic:
    """Returns a fixed typed-decompose JSON regardless of prompt."""

    def __init__(self, decompose_json: str):
        self.decompose = decompose_json

    async def complete(self, messages, tools=None, response_format=None):
        return AgentResponse(content=self.decompose)


class _SequenceCritic:
    """Returns scripted responses in order (decompose, then NLI verdict, ...)."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self._i = 0

    async def complete(self, messages, tools=None, response_format=None):
        r = self.responses[min(self._i, len(self.responses) - 1)]
        self._i += 1
        return AgentResponse(content=r)


# --------------------------------------------------------------------------- Part A: CRITIC


class TestCriticToolVerification:
    async def test_math_mismatch_flagged(self):
        reg = make_tool_registry()  # has calculate
        critic = _DecomposeCritic('[{"claim":"2+2=5","kind":"math","hint":"2+2"}]')
        hook = ReflectionHook(client=critic, tools=reg, verifier_tools=["calculate"])
        critique = await hook._tool_verify_claims("2+2=5")
        assert critique is not None
        assert "2+2=5" in critique
        assert "4" in critique  # calculate evaluated 2+2 -> 4
        assert "5" in critique  # the claimed (wrong) result

    async def test_math_correct_no_critique(self):
        reg = make_tool_registry()
        critic = _DecomposeCritic('[{"claim":"2+2=4","kind":"math","hint":"2+2"}]')
        hook = ReflectionHook(client=critic, tools=reg, verifier_tools=["calculate"])
        # 2+2=4 verifies OK -> no tool-grounded critique -> None (falls back to intrinsic).
        assert await hook._tool_verify_claims("2+2=4") is None

    async def test_no_tools_returns_none(self):
        hook = ReflectionHook(client=_DecomposeCritic("[]"), tools=None)
        assert await hook._tool_verify_claims("anything") is None

    async def test_unsafe_verifier_tool_filtered(self):
        # A tool outside the SAFE allowlist (run_shell) is never called even if configured.
        reg = ToolRegistry()
        reg.register("run_shell", "shell", {"type": "object", "properties": {}}, lambda: "pwned")
        critic = _DecomposeCritic('[{"claim":"x","kind":"math","hint":"1+1"}]')
        hook = ReflectionHook(client=critic, tools=reg, verifier_tools=["run_shell", "calculate"])
        # run_shell filtered out; only calculate remains. "x" has no number -> no mismatch.
        assert "run_shell" not in hook._verifier_tools
        assert "calculate" in hook._verifier_tools

    async def test_fact_check_flags_refuted_claim(self):
        reg = ToolRegistry()

        def _search(query):
            return "Marseille is the capital of France."

        reg.register("web_search", "search", {"type": "object", "properties": {"query": {"type": "string"}}}, _search)
        # critic: call 1 = typed-decompose JSON, call 2 = NLI verdict (REFUTE -> flagged).
        critic = _SequenceCritic(['[{"claim":"Paris is the capital of France","kind":"fact"}]', "REFUTE"])
        hook = ReflectionHook(client=critic, tools=reg, verifier_tools=["web_search"])
        critique = await hook._tool_verify_claims("Paris is the capital of France")
        assert critique is not None
        assert "contradicted" in critique

    async def test_fact_check_supported_claim_no_critique(self):
        reg = ToolRegistry()
        reg.register(
            "web_search",
            "search",
            {"type": "object", "properties": {"query": {"type": "string"}}},
            lambda query: "Paris is the capital of France.",
        )
        # NLI verdict SUPPORT -> the snippet supports the claim -> no critique (review C1).
        critic = _SequenceCritic(['[{"claim":"Paris is the capital of France","kind":"fact"}]', "SUPPORT"])
        hook = ReflectionHook(client=critic, tools=reg, verifier_tools=["web_search"])
        assert await hook._tool_verify_claims("Paris is the capital of France") is None


# --------------------------------------------------------------------------- Part B: self-consistency


class TestAggregateStructured:
    def test_majority_wins(self):
        a = AgentResponse(content='{"x": 1}')
        b = AgentResponse(content='{"x": 1}')
        c = AgentResponse(content='{"x": 2}')
        canonical, agreement = aggregate_structured([a, b, c])
        assert canonical.content in ('{"x": 1}', '{"x":1}')  # the majority
        assert agreement == 2 / 3

    def test_single_sample(self):
        a = AgentResponse(content='{"x": 1}')
        canonical, agreement = aggregate_structured([a])
        assert canonical is a
        assert agreement == 1.0

    def test_invalid_json_falls_back_to_first(self):
        good = AgentResponse(content='{"x": 1}')
        bad = AgentResponse(content="not json")
        canonical, agreement = aggregate_structured([good, bad])
        assert canonical is good
        assert agreement == 1.0

    def test_usage_summed(self):
        samples = [
            AgentResponse(content='{"x":1}', usage=TokenUsage(prompt_tokens=10, completion_tokens=5)),
            AgentResponse(content='{"x":1}', usage=TokenUsage(prompt_tokens=10, completion_tokens=5)),
        ]
        canonical, _ = aggregate_structured(samples)
        assert canonical.usage is not None
        assert canonical.usage.prompt_tokens == 20  # summed across both samples
        assert canonical.usage.completion_tokens == 10


class TestSelfConsistencyGating:
    def test_disabled_or_no_schema(self):
        agent = AgentCore(
            client=None,  # not used by the gate
            memory=ConversationMemory(),
            self_consistency_config={"enabled": True, "n_samples": 3},
        )
        assert agent._self_consistency_applies(None) is False  # no response_format -> off
        agent2 = AgentCore(
            client=None,
            memory=ConversationMemory(),
            self_consistency_config={"enabled": False, "n_samples": 3},
        )
        assert agent2._self_consistency_applies({"type": "object"}) is False  # disabled
        agent3 = AgentCore(
            client=None,
            memory=ConversationMemory(),
            self_consistency_config={"enabled": True, "n_samples": 1},
        )
        assert agent3._self_consistency_applies({"type": "object"}) is False  # n < 2

    def test_enabled_with_schema(self):
        agent = AgentCore(
            client=None,
            memory=ConversationMemory(),
            self_consistency_config={"enabled": True, "n_samples": 3},
        )
        assert agent._self_consistency_applies({"type": "object"}) is True


class TestSelfConsistencyIntegration:
    async def test_structured_terminal_aggregates(self, mock_client):
        # 3 identical structured answers -> self-consistency fires, agreement 1.0.
        client = mock_client(responses=[make_mock_response('{"x": 1}')] * 3)
        agent = AgentCore(
            client=client,
            memory=ConversationMemory(),
            tools=ToolRegistry(),
            max_iterations=2,
            output_schema={"type": "object", "properties": {"x": {"type": "number"}}},
            self_consistency_config={"enabled": True, "n_samples": 3},
        )
        result = await agent.run("q")
        assert result.content == '{"x": 1}' or result.content == '{"x":1}'
        assert result.metadata["self_consistency"] is not None
        assert result.metadata["self_consistency"]["n"] == 3
        assert result.metadata["self_consistency"]["agreement"] == 1.0
        assert client.call_count == 3  # 1 loop + 2 sampler
        # Usage accounts for ALL N samples (each make_mock_response = 10 prompt tokens).
        assert result.token_usage.prompt_tokens == 30
