"""koboi/hooks/chain.py -- branch coverage for MetadataBag, HookContext, HookChain."""

from __future__ import annotations


from koboi.hooks.chain import (
    AgentInfo,
    Hook,
    HookChain,
    HookContext,
    HookEvent,
    HookOutcome,
    MetadataBag,
)


class TestMetadataBag:
    def test_mode_properties(self):
        b = MetadataBag()
        assert b.mode_blocked is False
        assert b.mode_block_reason == ""
        b.mode_blocked = True
        b["mode_block_reason"] = "not allowed"
        assert b.mode_blocked is True
        assert b.mode_block_reason == "not allowed"

    def test_policy_properties(self):
        b = MetadataBag()
        assert b.policy_decision is None
        assert b.policy_needs_confirmation is False
        b["policy_decision"] = {"deny": True}
        b["policy_needs_confirmation"] = True
        assert b.policy_decision == {"deny": True}
        assert b.policy_needs_confirmation is True

    def test_guardrail_properties(self):
        b = MetadataBag()
        assert b.guardrail_blocked is False
        assert b.input_guardrail_result is None
        assert b.output_guardrail_result is None
        assert b.output_warning == ""
        b["guardrail_blocked"] = True
        b["input_guardrail_result"] = {"r": 1}
        b["output_guardrail_result"] = {"r": 2}
        b["output_warning"] = "careful"
        assert b.guardrail_blocked is True
        assert b.input_guardrail_result == {"r": 1}
        assert b.output_guardrail_result == {"r": 2}
        assert b.output_warning == "careful"

    def test_doom_loop_properties(self):
        b = MetadataBag()
        assert b.doom_loop_detected is False
        assert b.doom_loop_info is None
        b["doom_loop_detected"] = True
        b["doom_loop"] = {"count": 3}
        assert b.doom_loop_detected is True
        assert b.doom_loop_info == {"count": 3}

    def test_subagent_properties(self):
        b = MetadataBag()
        assert b.subagent_label is None
        assert b.subagent_task is None
        assert b.subagent_success is None
        assert b.subagent_error is None
        b["subagent_label"] = "w1"
        b["subagent_task"] = "do thing"
        b["subagent_success"] = True
        b["subagent_error"] = "boom"
        assert b.subagent_label == "w1"
        assert b.subagent_task == "do thing"
        assert b.subagent_success is True
        assert b.subagent_error == "boom"

    def test_rag_skills_context_properties(self):
        b = MetadataBag()
        assert b.rag_strategy == ""
        assert b.rag_augmentation == ""
        assert b.skills_detected == []
        assert b.context_managed is False
        b["rag_strategy"] = "hybrid"
        b["rag_augmentation"] = "ctx"
        b["skills_detected"] = ["a"]
        b["context_managed"] = True
        assert b.rag_strategy == "hybrid"
        assert b.rag_augmentation == "ctx"
        assert b.skills_detected == ["a"]
        assert b.context_managed is True


class TestHookContext:
    def test_defaults_and_metadata_copy(self):
        ctx = HookContext(HookEvent.PRE_TOOL_USE, metadata={"mode_blocked": True})
        assert ctx.event == HookEvent.PRE_TOOL_USE
        assert ctx.abort is False
        assert ctx.inject_messages == []
        assert ctx.inject_message is None
        assert ctx.metadata.mode_blocked is True  # metadata wrapped into MetadataBag

    def test_inject_message_single(self):
        ctx = HookContext(HookEvent.PRE_INPUT, inject_message="hello")
        assert ctx.inject_messages == ["hello"]
        assert ctx.inject_message == "hello"

    def test_inject_message_setter_appends(self):
        ctx = HookContext(HookEvent.PRE_INPUT)
        ctx.inject_message = "first"
        ctx.inject_message = "second"
        assert ctx.inject_messages == ["first", "second"]
        assert ctx.inject_message == "second"
        # setting None is a no-op
        ctx.inject_message = None
        assert ctx.inject_message == "second"

    def test_hook_outcomes_passed_through(self):
        ctx = HookContext(HookEvent.POST_TOOL_USE, hook_outcomes=[("H", HookOutcome.SUCCESS, None)])
        assert ctx.hook_outcomes == [("H", HookOutcome.SUCCESS, None)]

    def test_agent_info_defaults(self):
        ai = AgentInfo()
        assert ai.model == "" and ai.agent_name == "" and ai.iteration == 0


class _SpyHook(Hook):
    priority = 50

    def __init__(self, events, marker=None):
        self._events = events
        self.marker = marker
        self.calls = 0

    def handles(self):
        return self._events

    async def execute(self, ctx):
        self.calls += 1
        if self.marker:
            ctx.metadata["touched_by"] = self.marker
        return ctx


class _AbortHook(Hook):
    priority = 10

    def handles(self):
        return [HookEvent.PRE_TOOL_USE]

    async def execute(self, ctx):
        ctx.abort = True
        return ctx


class _BoomHook(Hook):
    priority = 50

    def handles(self):
        return [HookEvent.POST_LLM_CALL]

    async def execute(self, ctx):
        raise RuntimeError("kaboom")


class TestHookChain:
    async def test_emit_runs_subscribers_in_priority_order(self):
        a = _SpyHook([HookEvent.PRE_TOOL_USE], marker="a")
        a.priority = 60
        b = _SpyHook([HookEvent.PRE_TOOL_USE], marker="b")
        b.priority = 10
        chain = HookChain([a, b])
        ctx = await chain.emit(HookContext(HookEvent.PRE_TOOL_USE))
        assert b.calls == 1 and a.calls == 1
        # lower priority (b=10) ran first and set the marker, then a=60 overwrote
        assert ctx.metadata.get("touched_by") == "a"

    async def test_emit_abort_stops_chain(self):
        late = _SpyHook([HookEvent.PRE_TOOL_USE], marker="late")
        late.priority = 90
        chain = HookChain([_AbortHook(), late])
        ctx = await chain.emit(HookContext(HookEvent.PRE_TOOL_USE))
        assert ctx.abort is True
        assert late.calls == 0  # aborted before reaching it
        assert ctx.hook_outcomes[-1] == ("_AbortHook", HookOutcome.SUCCESS, None)

    async def test_emit_hook_error_sets_abort_and_outcome(self):
        chain = HookChain([_BoomHook()])
        ctx = await chain.emit(HookContext(HookEvent.POST_LLM_CALL))
        assert ctx.abort is True
        assert ctx.hook_outcomes[0][0] == "_BoomHook"
        assert ctx.hook_outcomes[0][1] == HookOutcome.ERRORED
        assert "kaboom" in ctx.hook_outcomes[0][2]

    async def test_emit_no_subscribers_returns_ctx(self):
        chain = HookChain([_SpyHook([HookEvent.SESSION_START])])
        ctx = await chain.emit(HookContext(HookEvent.SESSION_END))
        assert ctx.abort is False

    def test_add_rebuilds_index(self):
        chain = HookChain([])
        chain.add(_SpyHook([HookEvent.PRE_INPUT]))
        assert chain.find_hook(lambda h: True) is not None

    def test_list_hooks_and_find(self):
        h = _SpyHook([HookEvent.PRE_INPUT, HookEvent.POST_LLM_CALL])
        h2 = _SpyHook([HookEvent.SESSION_START])
        chain = HookChain([h, h2])
        names = [d["name"] for d in chain.list_hooks()]
        assert "_SpyHook" in names
        assert chain.find_hook(lambda x: x is h) is h
        assert chain.find_hook(lambda x: False) is None

    def test_priority_sort(self):
        z = _SpyHook([HookEvent.SESSION_START])
        z.priority = 99
        a = _SpyHook([HookEvent.SESSION_START])
        a.priority = 1
        chain = HookChain([z, a])
        # _hooks sorted ascending by priority
        assert chain._hooks[0] is a and chain._hooks[1] is z
