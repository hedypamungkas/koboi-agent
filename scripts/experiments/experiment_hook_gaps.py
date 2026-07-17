"""experiment_hook_gaps.py -- Empirically verify 4 latent hook-system issues.

Each experiment exercises REAL koboi classes (no mocks of the system under test)
and prints PASS/FAIL with evidence. Run: `python experiment_hook_gaps.py`.
"""

from __future__ import annotations

import asyncio

from koboi.hooks.chain import Hook, HookChain, HookContext, HookEvent
from koboi.hooks.doom_loop_hook import DoomLoopHook
from koboi.hooks.mode_hook import ModeHook
from koboi.harness.doom_loop import DoomLoopConfig
from koboi.loop_pipeline import ToolExecutionPipeline
from koboi.memory import ConversationMemory
from koboi.modes import AgentMode, ModeManager
from koboi.tools.registry import ToolRegistry
from koboi.types import ToolCall

# Used only by the AgentCore.run() experiment (issue 3) -- mirrors tests/conftest.py
from koboi.types import AgentResponse, TokenUsage
from koboi.loop import AgentCore


def _hr(label: str) -> None:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")


# ---------------------------------------------------------------------------
# Experiment 1: DOOM_LOOP_DETECTED is never emitted (only a metadata flag)
# ---------------------------------------------------------------------------

class DoomSubscriber(Hook):
    """Subscribes to DOOM_LOOP_DETECTED -- counts how often it actually fires."""

    def __init__(self):
        self.fired = 0

    def handles(self):
        return [HookEvent.DOOM_LOOP_DETECTED]

    async def execute(self, ctx):
        self.fired += 1
        return ctx


async def experiment_1():
    _hr("ISSUE 1: DOOM_LOOP_DETECTED is never emitted (subscribers silently inert)")

    doom_hook = DoomLoopHook(config=DoomLoopConfig(consecutive_identical_threshold=2))
    subscriber = DoomSubscriber()
    chain = HookChain([doom_hook, subscriber])

    last_meta_detected = False
    for _ in range(3):  # 3 identical failing calls -> doom must trigger
        ctx = HookContext(
            event=HookEvent.POST_TOOL_USE,
            tool_name="flaky_tool",
            tool_arguments='{"q": 1}',
            tool_result="Error: connection refused",
        )
        ctx = await chain.emit(ctx)
        last_meta_detected = ctx.metadata.get("doom_loop_detected", False)

    print(f"  doom detected (metadata flag set on POST_TOOL_USE ctx): {last_meta_detected}")
    print(f"  DOOM_LOOP_DETECTED subscriber fired count:              {subscriber.fired}")

    bug_confirmed = last_meta_detected and subscriber.fired == 0
    print(f"  => {'CONFIRMED BUG' if bug_confirmed else 'NOT REPRODUCED'}: "
          f"doom is detected but the event is never emitted, so any hook ")
    print(f"     subscribed to DOOM_LOOP_DETECTED (AuditHook, TelemetryHook, ")
    print(f"     LangfuseTracingHook, NotificationHook) NEVER fires.")
    return bug_confirmed


# ---------------------------------------------------------------------------
# Experiment 2: approval runs BEFORE the mode-block check
# ---------------------------------------------------------------------------

class AlwaysApprove:
    """Approval handler that always says yes and records that it was consulted."""

    def __init__(self):
        self.asked_for = []

    def should_approve(self, tool_name, arguments, risk):
        self.asked_for.append(tool_name)
        return True  # user approves the destructive action


async def experiment_2():
    _hr("ISSUE 2: an APPROVED tool can still be MODE-BLOCKED (chat/plan)")

    mode_manager = ModeManager(initial_mode=AgentMode.CHAT)
    approval = AlwaysApprove()

    tools = ToolRegistry()
    tools.register(
        name="custom_action",
        description="A non-read-only custom tool",
        parameters={"type": "object", "properties": {}, "required": []},
        fn=lambda: "executed",
    )

    pipeline = ToolExecutionPipeline(
        tools=tools,
        memory=ConversationMemory(),
        approval_handler=approval,                       # a human says "yes"
        hook_chain=HookChain([ModeHook(mode_manager)]),  # sets mode_blocked flag
        mode_manager=mode_manager,
    )

    tc = ToolCall(id="tc1", name="custom_action", arguments="{}")
    pr = await pipeline.execute_tool_call(tc, iteration=0)

    print(f"  approval handler consulted for:        {approval.asked_for}")
    print(f"  approval returned:                     APPROVED (proceed)")
    print(f"  pipeline result.skipped:               {pr.skipped}")
    print(f"  pipeline result.skip_reason:           {pr.skip_reason!r}")

    bug_confirmed = bool(approval.asked_for) and pr.skipped and pr.skip_reason == "mode_blocked"
    print(f"  => {'CONFIRMED' if bug_confirmed else 'NOT REPRODUCED'}: tool was approved at step 3, "
          f"then blocked at step 5 by ModeHook metadata.")
    return bug_confirmed


# ---------------------------------------------------------------------------
# Experiment 3: tool_calls_made records blocked/skipped tools (false positive)
# ---------------------------------------------------------------------------

class _ScriptedClient:
    """Returns a canned sequence of AgentResponses (like conftest.MockClient)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._i = 0
        self._model = "mock-model"

    @property
    def model(self):
        return self._model

    async def complete(self, messages, tools=None):
        if self._i < len(self._responses):
            r = self._responses[self._i]
            self._i += 1
            return r
        return AgentResponse(content="done", tool_calls=[], usage=TokenUsage(1, 1))

    async def complete_stream(self, messages, tools=None):
        for r in [await self.complete(messages, tools)]:
            from koboi.events import TextDeltaEvent, CompleteEvent
            if r.content:
                yield TextDeltaEvent(content=r.content)
            yield CompleteEvent(response=r, content=r.content or "")

    async def get_embeddings(self, text):
        return None

    async def close(self):
        pass


async def experiment_3():
    _hr("ISSUE 3: blocked tools still appear in RunResult.tool_calls_made")

    # Iteration 0: LLM asks for custom_action (mode-blocked in CHAT).
    # Iteration 1: LLM finishes.
    tool_resp = AgentResponse(
        content=None,
        tool_calls=[ToolCall(id="tc1", name="custom_action", arguments="{}")],
        usage=TokenUsage(1, 1),
    )
    done_resp = AgentResponse(content="finished", tool_calls=[], usage=TokenUsage(1, 1))

    tools = ToolRegistry()
    tools.register(
        name="custom_action",
        description="non-read-only tool",
        parameters={"type": "object", "properties": {}, "required": []},
        fn=lambda: "executed",
    )

    mode_manager = ModeManager(initial_mode=AgentMode.CHAT)
    agent = AgentCore(
        client=_ScriptedClient([tool_resp, done_resp]),
        memory=ConversationMemory(),
        tools=tools,
        max_iterations=5,
        mode_manager=mode_manager,
        hook_chain=HookChain([ModeHook(mode_manager)]),
    )

    result = await agent.run("please run custom_action")

    made_names = [tc.name for tc in result.tool_calls_made]
    outcome = next((o for o in result.pipeline_outcomes if o["tool_name"] == "custom_action"), None)

    print(f"  RunResult.tool_calls_made names: {made_names}")
    print(f"  matching pipeline_outcomes:      {outcome}")
    print(f"  => tool_calls_made claims it ran, but pipeline says skipped={outcome['skipped'] if outcome else '?'}")

    bug_confirmed = (
        "custom_action" in made_names
        and outcome is not None
        and outcome["skipped"] is True
    )
    print(f"  => {'CONFIRMED BUG' if bug_confirmed else 'NOT REPRODUCED'}: a mode-blocked tool is "
          f"falsely counted in tool_calls_made (eval false positive).")
    return bug_confirmed


# ---------------------------------------------------------------------------
# Experiment 4: RAGHook / GuardrailHook / ContextHook are never wired
# ---------------------------------------------------------------------------

def experiment_4():
    _hr("ISSUE 4: RAGHook / GuardrailHook / ContextHook are dead or redundant")

    from koboi.hooks.registry import list_entries

    registered = {e.name for e in list_entries()}
    suspects = ["RAGHook", "GuardrailHook", "ContextHook"]
    print(f"  Hooks auto-wired by _REGISTRY: {sorted(registered)}")
    for s in suspects:
        present = s in registered
        print(f"  {s:14} registered? {present}  -> {'DEAD' if not present else 'wired'}")

    # Confirm the loop wires guardrails + context DIRECTLY, bypassing the hooks.
    import inspect, koboi.loop as loop_mod

    src = inspect.getsource(loop_mod)
    direct_guardrails = "self.input_guardrails" in src and "self.output_guardrails" in src
    direct_context = "context_manager" in src
    print(f"  loop.py does guardrails directly (input/output lists): {direct_guardrails}")
    print(f"  loop.py does context management directly (context_manager): {direct_context}")

    confirmed = all(s not in registered for s in suspects) and direct_guardrails and direct_context
    print(f"  => {'CONFIRMED' if confirmed else 'NOT REPRODUCED'}: 3 hooks exist in the codebase but are "
          f"never registered; the loop achieves the same behavior directly.")
    return confirmed


async def main():
    print("koboi-agent hook-system gap verification\n"
          "(exercising real HookChain / ToolExecutionPipeline / AgentCore.run)")
    r1 = await experiment_1()
    r2 = await experiment_2()
    r3 = await experiment_3()
    r4 = experiment_4()
    _hr("SUMMARY")
    print(f"  Issue 1 (DOOM_LOOP_DETECTED never emitted):     {'CONFIRMED' if r1 else 'NO'}")
    print(f"  Issue 2 (approval before mode-block):           {'CONFIRMED' if r2 else 'NO'}")
    print(f"  Issue 3 (tool_calls_made false positive):       {'CONFIRMED' if r3 else 'NO'}")
    print(f"  Issue 4 (RAG/Guardrail/ContextHook dead):       {'CONFIRMED' if r4 else 'NO'}")
    n = sum([r1, r2, r3, r4])
    print(f"\n  {n}/4 issues empirically reproduced.")


if __name__ == "__main__":
    asyncio.run(main())
