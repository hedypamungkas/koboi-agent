"""Tests for koboi.guardrails.grounding.GroundingGuardrail (Wave 2 A3).

The guardrail is driven by a scripted side-LLM judge (no real API calls): the
decompose call returns a JSON claim array, the batch-NLI call returns a JSON
array of verdicts. Coverage math + the abstain decision + fail-soft are asserted.
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


def _guard(judge=None, threshold: float = 0.8, fail_closed: bool = False) -> GroundingGuardrail:
    g = GroundingGuardrail(
        provider="openai", model="gpt-4o-mini", api_key="x",
        threshold=threshold, fail_closed=fail_closed,
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
        judge = _ScriptedJudge(['["a is x", "b is y"]', '["SUPPORTED", "SUPPORTED"]'])
        g = _guard(judge)
        result = await g.check("a is x. b is y.", context=["a is x", "b is y"])
        assert result.passed is True
        assert g.last_coverage == 1.0

    async def test_low_coverage_abstains(self):
        # 2 claims, 1 SUPPORTED -> coverage 0.5 < 0.8 -> abstain (refusal swap).
        judge = _ScriptedJudge(['["a is x", "b is z"]', '["SUPPORTED", "UNSUPPORTED"]'])
        g = _guard(judge)
        result = await g.check("a is x. b is z.", context=["a is x"])
        assert result.passed is False
        assert result.action == "abstain"
        assert "grounding coverage 0.50" in result.reason
        assert result.sanitized_content  # the refusal text
        assert g.last_coverage == 0.5

    async def test_threshold_is_configurable(self):
        # Same 0.5 coverage passes when threshold is lowered to 0.5.
        judge = _ScriptedJudge(['["a", "b"]', '["SUPPORTED", "UNSUPPORTED"]'])
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
        judge = _ScriptedJudge(["not json at all", '["SUPPORTED"]'])
        g = _guard(judge)
        result = await g.check("answer", context=["ctx"])
        assert result.passed is True  # 1 claim (the line), SUPPORTED -> 1.0

    async def test_batch_nli_fallback_on_bad_json(self):
        # Batch NLI returns non-JSON -> falls back to per-claim NLI (2nd + 3rd calls).
        judge = _ScriptedJudge(['["a", "b"]', "not json", "SUPPORTED", "UNSUPPORTED"])
        g = _guard(judge)
        result = await g.check("a. b.", context=["a"])
        assert result.passed is False  # 1 SUPPORTED of 2 -> 0.5 < 0.8
        assert g.last_coverage == 0.5
        assert judge.calls == 4  # decompose + batch (failed) + 2 per-claim fallback

    async def test_client_build_failure_passes(self):
        # Unknown provider -> create_client raises -> _get_client returns None -> pass.
        g = GroundingGuardrail(provider="badprovider", model="x", api_key="")
        result = await g.check("answer", context=["ctx"])
        assert result.passed is True

    # ---- T2: fail-closed mode (opt-in). Default False preserves v0.18.3 fail-soft. ----

    async def test_fail_closed_on_judge_error_hands_over(self):
        # fail_closed=True + judge raises -> action="handover" (NOT pass-through).
        g = _guard(_BoomJudge(), fail_closed=True)
        result = await g.check("answer", context=["ctx"])
        assert result.passed is False
        assert result.action == "handover"
        assert "fail-closed" in result.reason
        assert result.sanitized_content  # the refusal text
        assert g.last_coverage is None

    async def test_fail_closed_on_no_client_hands_over(self):
        # Unknown provider -> _get_client returns None -> fail_closed hands over.
        g = GroundingGuardrail(provider="badprovider", model="x", api_key="", fail_closed=True)
        result = await g.check("answer", context=["ctx"])
        assert result.passed is False
        assert result.action == "handover"

    async def test_fail_closed_on_no_context_hands_over(self):
        g = _guard(_ScriptedJudge(["should not be called"]), fail_closed=True)
        result = await g.check("answer", context=[])
        assert result.passed is False
        assert result.action == "handover"

    async def test_fail_closed_on_no_claims_hands_over(self):
        # judge returns an empty claim array -> fail_closed hands over.
        judge = _ScriptedJudge(["[]"])
        g = _guard(judge, fail_closed=True)
        result = await g.check("answer", context=["ctx"])
        assert result.passed is False
        assert result.action == "handover"

    async def test_fail_closed_false_preserves_fail_soft(self):
        # Regression: fail_closed=False (default) keeps v0.18.3 pass-through on judge error.
        g = _guard(_BoomJudge(), fail_closed=False)
        result = await g.check("answer", context=["ctx"])
        assert result.passed is True

    async def test_fail_closed_does_not_change_low_coverage_abstain(self):
        # fail_closed only flips error/skip paths; a real low-coverage answer still abstains.
        judge = _ScriptedJudge(['["a is x", "b is z"]', '["SUPPORTED", "UNSUPPORTED"]'])
        g = _guard(judge, threshold=0.8, fail_closed=True)
        result = await g.check("a is x. b is z.", context=["a is x"])
        assert result.passed is False
        assert result.action == "abstain"  # NOT "handover"
        assert g.last_coverage == 0.5

    async def test_fail_closed_factory_passthrough(self):
        # YAML -> GuardrailRegistry.create(**kwargs) -> __init__: fail_closed flows through.
        from koboi.guardrails.registry import GuardrailRegistry, register_builtin_guardrails

        register_builtin_guardrails()
        grd = GuardrailRegistry.create("grounding_check", provider="openai", api_key="x", fail_closed=True)
        assert isinstance(grd, GroundingGuardrail)
        assert grd._fail_closed is True
        grd2 = GuardrailRegistry.create("grounding_check", provider="openai", api_key="x")
        assert grd2._fail_closed is False  # default preserved when omitted

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
        judge = _ScriptedJudge(['["acme ceo is john", "acme revenue is 9bn"]', '["SUPPORTED", "UNSUPPORTED"]'])
        guard = _guard(judge, threshold=0.8)

        class _FakeAug:
            # Non-empty last_results so the loop threads a non-empty context to the
            # guardrail (otherwise the cost-gate passes and no judge call happens).
            last_results = [RetrievalResult(Chunk(id="c1", doc_id="d", content="Acme CEO is John."), 0.9, "keyword")]
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

    async def test_fail_closed_grounding_raises_handover_in_loop(self):
        """T2 e2e: a fail-closed GroundingGuardrail whose judge errors causes
        ``_process_output`` to save the refusal to memory and then raise
        ``AgentHandoverError`` (reusing the B1 pipeline -> awaiting_human /
        HandoverEvent + handover.webhooks) instead of passing the unverified answer."""
        import pytest
        from koboi.exceptions import AgentHandoverError
        from koboi.loop import AgentCore
        from koboi.memory import ConversationMemory
        from koboi.rag.types import Chunk, RetrievalResult
        from koboi.tools.registry import ToolRegistry
        from tests.conftest import MockClient

        guard = _guard(_BoomJudge(), threshold=0.8, fail_closed=True)

        class _FakeAug:
            # Non-empty last_results so the loop threads non-empty context to the
            # guardrail (otherwise the no-context cost-gate short-circuits the judge).
            last_results = [RetrievalResult(Chunk(id="c1", doc_id="d", content="some context"), 0.9, "keyword")]
            last_rewrite = None

            async def augment_for_memory(self, user_message: str) -> str:
                return user_message

            async def augment_for_llm(self, messages):
                return messages

        core = AgentCore(
            client=MockClient([]),
            memory=ConversationMemory(),
            tools=ToolRegistry(),
            output_guardrails=[guard],
            max_iterations=1,
        )
        core.augmentation = _FakeAug()  # type: ignore[assignment]
        with pytest.raises(AgentHandoverError):
            await core._process_output("some unverified answer", response=None, iteration=0)
        # flag cleared after the raise (no stale state for the next run)
        assert core._handover_from_guardrail is None
