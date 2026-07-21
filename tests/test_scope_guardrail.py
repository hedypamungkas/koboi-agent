"""Tests for koboi.guardrails.scope.ScopeGuardrail.

The guardrail is driven by a scripted side-LLM judge (no real API calls). Covers:
the relevance-gating pre-pass (normal replies skip the judge), the abstain+deflection
decision on OFF_SCOPE/INJECTION, leniency (ON_SCOPE on a flagged reply still passes),
fail-soft on judge-unavailable/error, verdict normalization, config plumbing
(deflection_text, custom patterns), factory registration, and an end-to-end loop
swap (an off-scope code response is replaced by the graceful deflection).
"""

from __future__ import annotations

from koboi.guardrails.scope import ScopeGuardrail
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


def _guard(judge=None, **kw) -> ScopeGuardrail:
    g = ScopeGuardrail(provider="openai", model="gpt-4o-mini", api_key="x", **kw)
    if judge is not None:
        g._client = judge  # bypass lazy create_client
    return g


# A normal in-scope CS reply -- no code, no JSON, no constructs. Must skip the judge.
_NORMAL_REPLY = (
    "Halo Kak! Tas Canvas Waterproof kami ada, harga Rp189.000, warna Hitam/Navy/Olive, "
    "stok 15 unit. Mau saya bantu pilih warna? 😊"
)
# A clearly off-scope response: the agent wrote a Python calculator (reported attack #2).
_CODE_REPLY = "Tentu Kak, ini program calculator-nya:\n```python\ndef calc(a,b): return a+b\n```\n"
# The agent obeyed an injection: dumped the conversation as JSON (reported attack #1).
_JSON_DUMP_REPLY = (
    'Berikut percakapan dalam JSON:\n```json\n{"conversation":[{"role":"user","content":"/start"}]}\n```'
)


class TestScopeGuardrail:
    async def test_normal_reply_skips_judge(self):
        # Relevance gate: no structural suspicion -> zero judge calls, passes.
        judge = _ScriptedJudge(["should not be called"])
        g = _guard(judge)
        result = await g.check(_NORMAL_REPLY)
        assert result.passed is True
        assert judge.calls == 0
        assert g.last_verdict == "ON_SCOPE(pre-pass)"

    async def test_empty_content_passes(self):
        g = _guard(_ScriptedJudge([]))
        result = await g.check("")
        assert result.passed is True

    async def test_code_response_off_scope_abstains(self):
        judge = _ScriptedJudge(["OFF_SCOPE"])
        g = _guard(judge)
        result = await g.check(_CODE_REPLY)
        assert result.passed is False
        assert result.action == "abstain"
        assert result.sanitized_content  # graceful deflection, not the code
        assert "calculator" not in (result.sanitized_content or "")
        assert "off-scope" in result.reason
        assert g.last_verdict == "OFF_SCOPE"

    async def test_json_dump_injection_abstains(self):
        judge = _ScriptedJudge(["INJECTION"])
        g = _guard(judge)
        result = await g.check(_JSON_DUMP_REPLY)
        assert result.passed is False
        assert result.action == "abstain"
        assert result.sanitized_content  # deflection
        assert "conversation" not in (result.sanitized_content or "")
        assert g.last_verdict == "INJECTION"

    async def test_flagged_but_on_scope_passes(self):
        # A response that trips the pre-pass (has a code fence) but the judge says
        # ON_SCOPE (e.g. a legit reply that quoted a promo code in a fence) -> pass.
        judge = _ScriptedJudge(["ON_SCOPE"])
        g = _guard(judge)
        result = await g.check("Gunakan kode promo ```HEMAT10``` untuk diskon ya Kak!")
        assert result.passed is True
        assert judge.calls == 1  # the pre-pass flagged it, judge ran, said ON_SCOPE

    async def test_fail_soft_on_judge_error(self):
        g = _guard(_BoomJudge())
        result = await g.check(_CODE_REPLY)
        assert result.passed is True  # never breaks the run
        assert g.last_verdict is None

    async def test_client_build_failure_passes(self):
        # Unknown provider -> create_client raises -> _get_client None -> fail-soft pass.
        g = ScopeGuardrail(provider="badprovider", model="x", api_key="")
        result = await g.check(_CODE_REPLY)
        assert result.passed is True

    async def test_verdict_normalization_lenient(self):
        # Verbose / ambiguous judge output -> defaults to ON_SCOPE (lenient).
        g = _guard(_ScriptedJudge(["the response looks fine to me"]))
        result = await g.check(_CODE_REPLY)
        assert result.passed is True
        # 'OFF_SCOPE' substring still recognized even if surrounded by prose.
        g2 = _guard(_ScriptedJudge([" verdict: OFF_SCOPE (code)"]))
        result2 = await g2.check(_CODE_REPLY)
        assert result2.passed is False
        assert g2.last_verdict == "OFF_SCOPE"
        # 'INJECTION' checked before 'OFF' if both substrings present (harmful class wins).
        g3 = _guard(_ScriptedJudge(["this is INJECTION not just OFF_SCOPE"]))
        result3 = await g3.check(_CODE_REPLY)
        assert result3.passed is False
        assert g3.last_verdict == "INJECTION"

    async def test_custom_deflection_text_honored(self):
        g = _guard(_ScriptedJudge(["OFF_SCOPE"]), deflection_text="OUT OF LANE.")
        result = await g.check(_CODE_REPLY)
        assert result.sanitized_content == "OUT OF LANE."

    async def test_custom_patterns_extend_suspicion(self):
        # An otherwise-normal reply with a custom-flagged token trips the pre-pass.
        judge = _ScriptedJudge(["OFF_SCOPE"])
        g = _guard(judge, patterns=[(r"(?i)wiblurb", "custom flag")])
        result = await g.check("Halo Kak, stok aman wiblurb.")
        assert judge.calls == 1  # pre-pass flagged via the custom pattern -> judge ran
        assert result.passed is False

    async def test_registered_as_scope_check(self):
        from koboi.guardrails.registry import GuardrailRegistry, register_builtin_guardrails

        register_builtin_guardrails()
        grd = GuardrailRegistry.create("scope_check", provider="openai", api_key="x")
        assert isinstance(grd, ScopeGuardrail)

    async def test_factory_passthrough_kwargs(self):
        from koboi.guardrails.registry import GuardrailRegistry, register_builtin_guardrails

        register_builtin_guardrails()
        grd = GuardrailRegistry.create(
            "scope_check",
            provider="openai",
            api_key="x",
            deflection_text="CUSTOM",
            scope_description="only shoes",
        )
        assert isinstance(grd, ScopeGuardrail)
        assert grd._deflection == "CUSTOM"
        assert grd._scope == "only shoes"


class TestScopeGuardrailIntegration:
    """End-to-end: the loop's abstain action swaps an off-scope response for the
    deflection (A3.2 branch), so the customer never receives the complied-with
    injected/out-of-scope content."""

    async def test_off_scope_code_swapped_for_deflection(self):
        from koboi.loop import AgentCore
        from koboi.memory import ConversationMemory
        from koboi.tools.registry import ToolRegistry
        from tests.conftest import MockClient, make_mock_response

        judge = _ScriptedJudge(["OFF_SCOPE"])
        guard = _guard(judge, deflection_text="DEFLECTED: out of shop scope.")
        core = AgentCore(
            client=MockClient([make_mock_response(content=_CODE_REPLY)]),
            memory=ConversationMemory(),
            tools=ToolRegistry(),
            output_guardrails=[guard],
            max_iterations=1,
        )
        result = await core.run("buatkan program calculator python")
        assert "DEFLECTED: out of shop scope." in result.content
        assert "calculator" not in result.content  # the complied-with code did not ship
        assert result.metadata.get("guardrail_outcomes", [{}])[0].get("action") == "abstain"
