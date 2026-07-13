"""Tests for koboi.guardrails.grounding.GroundingGuardrail (Wave 2 A3).

The guardrail is driven by a scripted side-LLM judge (no real API calls): the
decompose call returns a JSON claim array, each NLI call returns SUPPORTED /
UNSUPPORTED. Coverage math + the abstain decision + fail-soft are asserted.
"""

from __future__ import annotations

from koboi.guardrails.grounding import GroundingGuardrail
from koboi.types import AgentResponse, TokenUsage


class _ScriptedJudge:
    """Returns scripted ``complete()`` replies in order; counts calls."""

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls = 0

    async def complete(self, messages, tools=None, response_format=None):
        self.calls += 1
        r = self._replies.pop(0) if self._replies else ""
        return AgentResponse(content=r, tool_calls=[], usage=TokenUsage(0, 0))

    async def get_embeddings(self, text):
        return None

    async def close(self):
        pass


class _BoomJudge:
    async def complete(self, messages, tools=None, response_format=None):
        raise RuntimeError("judge down")

    async def get_embeddings(self, text):
        return None

    async def close(self):
        pass


def _guard(judge=None, threshold: float = 0.8) -> GroundingGuardrail:
    g = GroundingGuardrail(
        provider="openai", model="gpt-4o-mini", api_key="x", threshold=threshold
    )
    if judge is not None:
        g._client = judge  # bypass lazy create_client
    return g


class TestGroundingGuardrail:
    async def test_empty_context_passes_no_judge_call(self):
        # Cost-gate: no retrieved context -> A2 already cued abstention; skip the judge.
        judge = _ScriptedJudge(["should not be called"])
        g = _guard(judge)
        result = await g.check("some answer", context=[])
        assert result.passed is True
        assert judge.calls == 0
        assert g.last_coverage is None

    async def test_no_context_arg_passes(self):
        g = _guard(_ScriptedJudge([]))
        result = await g.check("some answer", context=None)
        assert result.passed is True

    async def test_high_coverage_passes(self):
        # 2 claims, both SUPPORTED -> coverage 1.0 >= 0.8 -> pass.
        judge = _ScriptedJudge(['["a is x", "b is y"]', "SUPPORTED", "SUPPORTED"])
        g = _guard(judge)
        result = await g.check("a is x. b is y.", context=["a is x", "b is y"])
        assert result.passed is True
        assert g.last_coverage == 1.0

    async def test_low_coverage_abstains(self):
        # 2 claims, 1 SUPPORTED -> coverage 0.5 < 0.8 -> abstain (refusal swap).
        judge = _ScriptedJudge(['["a is x", "b is z"]', "SUPPORTED", "UNSUPPORTED"])
        g = _guard(judge)
        result = await g.check("a is x. b is z.", context=["a is x"])
        assert result.passed is False
        assert result.action == "abstain"
        assert "grounding coverage 0.50" in result.reason
        assert result.sanitized_content  # the refusal text
        assert g.last_coverage == 0.5

    async def test_threshold_is_configurable(self):
        # Same 0.5 coverage passes when threshold is lowered to 0.5.
        judge = _ScriptedJudge(['["a", "b"]', "SUPPORTED", "UNSUPPORTED"])
        g = _guard(judge, threshold=0.5)
        result = await g.check("a. b.", context=["a"])
        assert result.passed is True
        assert g.last_coverage == 0.5

    async def test_fail_soft_on_judge_error(self):
        g = _guard(_BoomJudge())
        result = await g.check("answer", context=["ctx"])
        assert result.passed is True  # never breaks the run
        assert g.last_coverage is None

    async def test_fail_soft_on_bad_json_decompose(self):
        # Non-JSON decompose output -> fallback line-split; the check still completes.
        judge = _ScriptedJudge(["not json at all", "SUPPORTED"])
        g = _guard(judge)
        result = await g.check("answer", context=["ctx"])
        assert result.passed is True  # 1 claim (the line), SUPPORTED -> 1.0

    async def test_client_build_failure_passes(self):
        # Unknown provider -> create_client raises -> _get_client returns None -> pass.
        g = GroundingGuardrail(provider="badprovider", model="x", api_key="")
        result = await g.check("answer", context=["ctx"])
        assert result.passed is True

    async def test_registered_as_grounding_check(self):
        from koboi.guardrails.registry import GuardrailRegistry, register_builtin_guardrails

        register_builtin_guardrails()
        grd = GuardrailRegistry.create("grounding_check", provider="openai", api_key="x")
        assert isinstance(grd, GroundingGuardrail)


class TestGroundingGuardrailIntegration:
    """End-to-end: a hallucinated answer with a low-coverage judge is swapped for
    the refusal by the loop's abstain action (A3.2)."""

    async def test_hallucinated_answer_swapped_for_refusal(self):
        from koboi.loop import AgentCore
        from koboi.memory import ConversationMemory
        from koboi.rag.types import Chunk, RetrievalResult
        from koboi.tools.registry import ToolRegistry
        from tests.conftest import MockClient, make_mock_response

        # The agent LLM returns a hallucination; the judge marks 1/2 claims unsupported.
        judge = _ScriptedJudge(['["acme ceo is john", "acme revenue is 9bn"]', "SUPPORTED", "UNSUPPORTED"])
        guard = _guard(judge, threshold=0.8)

        class _FakeAug:
            # Non-empty last_results so the loop threads a non-empty context to the
            # guardrail (otherwise the cost-gate passes and no judge call happens).
            last_results = [
                RetrievalResult(Chunk(id="c1", doc_id="d", content="Acme CEO is John."), 0.9, "keyword")
            ]
            last_rewrite = None

            async def augment_for_memory(self, user_message: str) -> str:
                return user_message

            async def augment_for_llm(self, messages):
                return messages

        core = AgentCore(
            client=MockClient([make_mock_response(content="Acme CEO is John. Acme revenue is 9bn.")]),
            memory=ConversationMemory(),
            tools=ToolRegistry(),
            output_guardrails=[guard],
            max_iterations=1,
        )
        core.augmentation = _FakeAug()  # type: ignore[assignment]
        result = await core.run("tell me about acme")
        assert "I don't have enough grounded information" in result.content
        assert "9bn" not in result.content  # the hallucinated claim did not survive
        assert result.metadata.get("guardrail_outcomes", [{}])[0].get("action") == "abstain"
