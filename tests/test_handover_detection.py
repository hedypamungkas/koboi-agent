"""Tests for B1.5 -- HandoverDetectionHook (structural, A3-fed handover).

The hook sets ``ctx.metadata["handover_requested"]`` (NOT raise -- emit swallows
hook exceptions into ``ctx.abort`` -> ``AgentAbortedError``, the wrong class).
The emit-site (``_validate_input`` / ``_process_output``) reads the flag and raises
``AgentHandoverError`` -> B1's HandoverEvent machinery.
"""

from __future__ import annotations

import pytest

from koboi.exceptions import AgentHandoverError
from koboi.hooks.chain import Hook, HookChain, HookContext, HookEvent
from koboi.hooks.handover_detection_hook import HandoverDetectionHook


# ---------------------------------------------------------------------------
# Unit (no fastapi) -- the flag logic
# ---------------------------------------------------------------------------


class _G:
    """Fake GroundingGuardrail (only last_coverage matters)."""

    def __init__(self, coverage):
        self.last_coverage = coverage


async def _emit(hook, event, **ctx_kwargs):
    return await HookChain([hook]).emit(HookContext(event=event, **ctx_kwargs))


class TestHandoverDetectionHookFlags:
    async def test_explicit_ask_sets_flag_at_pre_input(self):
        ctx = await _emit(HandoverDetectionHook(), HookEvent.PRE_INPUT, user_message="please talk to a human")
        hr = ctx.metadata.get("handover_requested")
        assert hr is not None and hr["reason"] == "user requested a human"
        assert "talk to a human" in hr["summary"]

    async def test_no_ask_no_flag(self):
        ctx = await _emit(HandoverDetectionHook(), HookEvent.PRE_INPUT, user_message="what is the refund window?")
        assert ctx.metadata.get("handover_requested") is None

    async def test_low_coverage_sets_flag_at_post_output(self):
        ctx = await _emit(
            HandoverDetectionHook(grounding=_G(0.3), coverage_threshold=0.5), HookEvent.POST_OUTPUT
        )
        hr = ctx.metadata.get("handover_requested")
        assert hr is not None and "0.30" in hr["reason"]

    async def test_high_coverage_no_flag(self):
        ctx = await _emit(
            HandoverDetectionHook(grounding=_G(0.9), coverage_threshold=0.5), HookEvent.POST_OUTPUT
        )
        assert ctx.metadata.get("handover_requested") is None

    async def test_no_grounding_no_coverage_flag(self):
        # grounding=None -> POST_OUTPUT never sets the coverage flag (explicit-ask still works).
        ctx = await _emit(HandoverDetectionHook(grounding=None), HookEvent.POST_OUTPUT)
        assert ctx.metadata.get("handover_requested") is None

    async def test_coverage_none_no_flag(self):
        # last_coverage is None (A3 cost-gated / skipped) -> no coverage flag.
        ctx = await _emit(
            HandoverDetectionHook(grounding=_G(None), coverage_threshold=0.5), HookEvent.POST_OUTPUT
        )
        assert ctx.metadata.get("handover_requested") is None

    async def test_custom_ask_patterns_override_defaults(self):
        h = HandoverDetectionHook(ask_patterns=[r"escalate"])
        ctx_match = await _emit(h, HookEvent.PRE_INPUT, user_message="please escalate this")
        assert ctx_match.metadata.get("handover_requested") is not None
        # Default pattern no longer matches (overridden).
        ctx_nomatch = await _emit(HandoverDetectionHook(ask_patterns=[r"escalate"]), HookEvent.PRE_INPUT, user_message="talk to a human")
        assert ctx_nomatch.metadata.get("handover_requested") is None

    async def test_hook_does_not_set_abort(self):
        # CRITICAL: setting abort would raise AgentAbortedError (wrong class) at the emit-site.
        ctx = await _emit(HandoverDetectionHook(), HookEvent.PRE_INPUT, user_message="talk to a human agent")
        assert ctx.abort is False


# ---------------------------------------------------------------------------
# Emit-site integration (no fastapi) -- the flag -> AgentHandoverError raise
# ---------------------------------------------------------------------------


class _FlagOnPreInput(Hook):
    def handles(self):
        return [HookEvent.PRE_INPUT]

    async def execute(self, ctx):
        ctx.metadata["handover_requested"] = {"reason": "test pre-input", "summary": "s"}
        return ctx


class _FlagOnPostOutput(Hook):
    def handles(self):
        return [HookEvent.POST_OUTPUT]

    async def execute(self, ctx):
        ctx.metadata["handover_requested"] = {"reason": "test post-output", "summary": "s"}
        return ctx


class TestEmitSiteRaisesHandover:
    @staticmethod
    def _core(hook):
        from koboi.loop import AgentCore
        from koboi.memory import ConversationMemory
        from koboi.modes import AgentMode, ModeManager
        from koboi.tools.registry import ToolRegistry
        from tests.conftest import MockClient, make_mock_response

        return AgentCore(
            client=MockClient([make_mock_response(content="hi")]),
            memory=ConversationMemory(),
            tools=ToolRegistry(),
            hook_chain=HookChain([hook]),
            mode_manager=ModeManager(initial_mode=AgentMode.ACT),
            max_iterations=1,
        )

    async def test_validate_input_raises_handover_on_pre_input_flag(self):
        with pytest.raises(AgentHandoverError) as ei:
            await self._core(_FlagOnPreInput()).run("anything")
        assert ei.value.reason == "test pre-input"

    async def test_process_output_raises_handover_on_post_output_flag(self):
        # The LLM answers first (MockClient), then POST_OUTPUT fires -> flag -> raise.
        with pytest.raises(AgentHandoverError) as ei:
            await self._core(_FlagOnPostOutput()).run("anything")
        assert ei.value.reason == "test post-output"


# ---------------------------------------------------------------------------
# Server e2e (fastapi) -- explicit-ask yields WITHOUT an LLM call
# ---------------------------------------------------------------------------


class TestHandoverDetectionServerE2E:
    async def test_explicit_ask_yields_without_llm_call(self):
        pytest.importorskip("fastapi")
        import httpx
        from koboi.config import Config
        from koboi.server import create_app
        from tests.conftest import MockClient, make_mock_response

        cfg = Config.from_dict(
            {
                "agent": {"name": "srv", "system_prompt": "h", "max_iterations": 3},
                "llm": {"provider": "openai", "model": "gpt-4o-mini", "api_key": "test"},
                "memory": {"backend": "in_memory"},
                "sandbox": {"backend": "restricted"},
                "server": {"auth_required": False},
                "handover": {"detection": {"enabled": True}},
            },
            validate=True,
        )
        mock = MockClient([make_mock_response(content="should not be reached")])

        app = create_app(cfg, client_factory=lambda: mock, enable_cors=False)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            async with c.stream(
                "POST",
                "/v1/chat/stream",
                json={"message": "I want to speak to a human"},
                headers={"X-Session-Id": "s-hd"},
            ) as r:
                body = (await r.aread()).decode()
        # The PRE_INPUT hook fired BEFORE the LLM -> AgentHandoverError -> HandoverEvent.
        assert '"type":"handover"' in body or '"type": "handover"' in body, body
        assert mock.call_count == 0, "the LLM must NOT be called (PRE_INPUT hook pre-empts it)"
