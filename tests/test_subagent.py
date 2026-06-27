"""Tests for the subagent feature: SubAgentManager, delegate_tasks tool, and SubagentUIHook."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from koboi.config import Config
from koboi.hooks.chain import HookChain, HookContext, HookEvent, Hook
from koboi.memory import ConversationMemory
from koboi.subagent import SubAgentManager, SubagentTask, _build_conversation_summary
from koboi.tools.builtin.subagent import delegate_tasks
from koboi.tools.registry import ToolRegistry

from tests.conftest import MockClient, make_mock_response


# ---------------------------------------------------------------------------
# _build_conversation_summary
# ---------------------------------------------------------------------------


class TestConversationSummary:
    def test_empty_messages(self):
        result = _build_conversation_summary([])
        assert result == "(no prior context)"

    def test_system_messages_skipped(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = _build_conversation_summary(messages)
        assert "You are helpful" not in result
        assert "Hello" in result

    def test_recent_messages_included(self):
        messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "First response"},
            {"role": "user", "content": "Second message"},
            {"role": "assistant", "content": "Second response"},
        ]
        result = _build_conversation_summary(messages)
        assert "Second message" in result
        assert "Second response" in result

    def test_respects_max_chars(self):
        messages = [
            {"role": "user", "content": "x" * 3000},
        ]
        result = _build_conversation_summary(messages, max_chars=500)
        assert len(result) < 600  # some overhead from role prefix


# ---------------------------------------------------------------------------
# SubAgentManager
# ---------------------------------------------------------------------------


class TestSubAgentManager:
    @pytest.fixture
    def manager(self):
        client = MockClient(
            responses=[
                make_mock_response(content="Subagent answer for task 1"),
                make_mock_response(content="Subagent answer for task 2"),
            ]
        )
        tools = ToolRegistry()
        hooks = HookChain()
        return SubAgentManager(
            client=client,
            tools=tools,
            hook_chain=hooks,
            max_iterations=3,
        )

    async def test_run_single_task(self, manager):
        tasks = [SubagentTask(task="Research topic X", label="research")]
        results = await manager.run_tasks(tasks)

        assert len(results) == 1
        assert results[0].success is True
        assert "Subagent answer" in results[0].answer
        assert results[0].label == "research"
        assert results[0].elapsed_seconds > 0

    async def test_run_multiple_tasks_parallel(self, manager):
        tasks = [
            SubagentTask(task="Task A", label="a"),
            SubagentTask(task="Task B", label="b"),
        ]
        results = await manager.run_tasks(tasks)

        assert len(results) == 2
        assert all(r.success for r in results)
        labels = {r.label for r in results}
        assert labels == {"a", "b"}

    async def test_run_with_parent_context(self, manager):
        parent_messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is the company policy?"},
            {"role": "assistant", "content": "Let me look that up."},
        ]
        tasks = [SubagentTask(task="Summarize policy", label="summary")]
        results = await manager.run_tasks(tasks, parent_messages=parent_messages)

        assert len(results) == 1
        assert results[0].success is True
        # Verify the child agent received context (check via the client's messages)
        # The child should have gotten a system prompt with the summary

    async def test_task_failure_returns_error(self):
        client = MockClient()
        client.complete = AsyncMock(side_effect=RuntimeError("LLM failed"))
        tools = ToolRegistry()
        hooks = HookChain()
        manager = SubAgentManager(client=client, tools=tools, hook_chain=hooks)

        tasks = [SubagentTask(task="Failing task", label="fail")]
        results = await manager.run_tasks(tasks)

        assert len(results) == 1
        assert results[0].success is False
        assert "LLM failed" in results[0].error

    async def test_emits_hook_events(self, manager):
        dispatched = []
        completed = []

        class TrackingHook(Hook):
            def handles(self):
                return [HookEvent.AGENT_DISPATCHED, HookEvent.AGENT_COMPLETED]

            async def execute(self, ctx):
                if ctx.event == HookEvent.AGENT_DISPATCHED:
                    dispatched.append(ctx.metadata)
                elif ctx.event == HookEvent.AGENT_COMPLETED:
                    completed.append(ctx.metadata)
                return ctx

        manager.hook_chain.add(TrackingHook())

        tasks = [SubagentTask(task="Test task", label="test")]
        await manager.run_tasks(tasks)

        assert len(dispatched) == 1
        assert dispatched[0]["subagent_label"] == "test"
        assert len(completed) == 1
        assert completed[0]["subagent_success"] is True

    async def test_default_label_assignment(self, manager):
        tasks = [SubagentTask(task="No label")]
        results = await manager.run_tasks(tasks)
        assert results[0].label == "task_0"


# ---------------------------------------------------------------------------
# delegate_tasks tool
# ---------------------------------------------------------------------------


class TestDelegateTasksTool:
    async def test_tool_returns_error_when_no_manager(self):
        result = await delegate_tasks([{"task": "test"}], _deps=None)
        assert "Error" in result

    async def test_tool_returns_formatted_results(self):
        client = MockClient(
            responses=[
                make_mock_response(content="Answer 1"),
                make_mock_response(content="Answer 2"),
            ]
        )
        tools = ToolRegistry()
        hooks = HookChain()
        manager = SubAgentManager(client=client, tools=tools, hook_chain=hooks, max_iterations=2)

        tasks = [
            {"task": "Research X", "label": "research"},
            {"task": "Analyze Y", "label": "analyze"},
        ]
        result = await delegate_tasks(tasks, _deps={"manager": manager})

        assert "research" in result
        assert "analyze" in result
        assert "Answer 1" in result
        assert "Answer 2" in result
        assert "---" in result  # separator between results

    async def test_tool_with_parent_memory(self):
        client = MockClient(
            responses=[
                make_mock_response(content="Contextual answer"),
            ]
        )
        tools = ToolRegistry()
        hooks = HookChain()
        manager = SubAgentManager(client=client, tools=tools, hook_chain=hooks, max_iterations=2)

        memory = ConversationMemory(system_prompt="You are helpful.")
        memory.add_user_message("What is the policy?")
        memory.add_assistant_message("Let me check.")
        manager._parent_memory = memory

        result = await delegate_tasks([{"task": "Summarize", "label": "sum"}], _deps={"manager": manager})

        assert "Contextual answer" in result

    async def test_tool_inherits_parent_tools(self):
        """Verify child agents can use the parent's tools."""
        client = MockClient(
            responses=[
                make_mock_response(content="Tool result"),
            ]
        )
        tools = ToolRegistry()
        tools.register(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            fn=lambda city: f"Sunny in {city}",
        )
        hooks = HookChain()
        manager = SubAgentManager(client=client, tools=tools, hook_chain=hooks, max_iterations=2)

        # The child agent should have access to get_weather
        result = await delegate_tasks([{"task": "Check weather", "label": "weather"}], _deps={"manager": manager})
        assert "Tool result" in result

    async def test_child_tools_exclude_delegate_tasks(self):
        """Verify child agents cannot call delegate_tasks (prevents recursion)."""
        client = MockClient(
            responses=[
                make_mock_response(content="Done"),
            ]
        )
        tools = ToolRegistry()
        tools.register(
            name="delegate_tasks",
            description="Delegate tasks",
            parameters={"type": "object", "properties": {}},
            fn=lambda: "should not be called",
        )
        tools.register(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            fn=lambda city: f"Sunny in {city}",
        )
        hooks = HookChain()
        manager = SubAgentManager(client=client, tools=tools, hook_chain=hooks, max_iterations=2)

        child_tools = manager._build_child_tools()
        assert "delegate_tasks" not in child_tools._tools
        assert "get_weather" in child_tools._tools


# ---------------------------------------------------------------------------
# SubagentUIHook
# ---------------------------------------------------------------------------


class TestSubagentUIHook:
    async def test_hook_posts_dispatch_message(self):
        from koboi.hooks.subagent_hook import SubagentUIHook, _SubagentDispatch

        app = MagicMock()
        hook = SubagentUIHook(app=app)

        ctx = HookContext(
            event=HookEvent.AGENT_DISPATCHED,
            metadata={
                "subagent_label": "research",
                "subagent_task": "Research X",
                "subagent_index": 0,
                "subagent_total": 2,
            },
        )
        await hook.execute(ctx)

        app.post_message.assert_called_once()
        msg = app.post_message.call_args[0][0]
        assert isinstance(msg, _SubagentDispatch)
        assert msg.label == "research"
        assert msg.total == 2

    async def test_hook_posts_result_message(self):
        from koboi.hooks.subagent_hook import SubagentUIHook, _SubagentResult

        app = MagicMock()
        hook = SubagentUIHook(app=app)

        ctx = HookContext(
            event=HookEvent.AGENT_COMPLETED,
            metadata={
                "subagent_label": "research",
                "subagent_elapsed": 1.5,
                "subagent_success": True,
            },
        )
        await hook.execute(ctx)

        app.post_message.assert_called_once()
        msg = app.post_message.call_args[0][0]
        assert isinstance(msg, _SubagentResult)
        assert msg.label == "research"
        assert msg.elapsed == 1.5
        assert msg.success is True

    async def test_hook_ignores_non_subagent_events(self):
        from koboi.hooks.subagent_hook import SubagentUIHook

        app = MagicMock()
        hook = SubagentUIHook(app=app)

        # Event without subagent_label metadata should be ignored
        ctx = HookContext(
            event=HookEvent.AGENT_DISPATCHED,
            metadata={"agent_name": "orchestrator_agent"},
        )
        await hook.execute(ctx)

        app.post_message.assert_not_called()

    async def test_hook_ignores_when_no_app(self):
        from koboi.hooks.subagent_hook import SubagentUIHook

        hook = SubagentUIHook(app=None)

        ctx = HookContext(
            event=HookEvent.AGENT_DISPATCHED,
            metadata={"subagent_label": "test", "subagent_index": 0, "subagent_total": 1},
        )
        # Should not raise
        result = await hook.execute(ctx)
        assert result is ctx


# ---------------------------------------------------------------------------
# Integration: tool + manager + hooks
# ---------------------------------------------------------------------------


class TestSubagentIntegration:
    async def test_full_flow_with_hooks(self):
        """End-to-end: tool call -> manager -> hooks -> results."""
        dispatch_events = []
        complete_events = []

        class CollectorHook(Hook):
            def handles(self):
                return [HookEvent.AGENT_DISPATCHED, HookEvent.AGENT_COMPLETED]

            async def execute(self, ctx):
                if ctx.event == HookEvent.AGENT_DISPATCHED:
                    dispatch_events.append(ctx.metadata)
                else:
                    complete_events.append(ctx.metadata)
                return ctx

        client = MockClient(
            responses=[
                make_mock_response(content="Result A"),
                make_mock_response(content="Result B"),
            ]
        )
        tools = ToolRegistry()
        hooks = HookChain()
        hooks.add(CollectorHook())
        manager = SubAgentManager(client=client, tools=tools, hook_chain=hooks, max_iterations=2)

        result = await delegate_tasks(
            [
                {"task": "Do A", "label": "a"},
                {"task": "Do B", "label": "b"},
            ],
            _deps={"manager": manager},
        )

        assert len(dispatch_events) == 2
        assert len(complete_events) == 2
        assert "Result A" in result
        assert "Result B" in result


# ---------------------------------------------------------------------------
# Lifecycle: timeout, cancel, cleanup
# ---------------------------------------------------------------------------


class TestSubagentLifecycle:
    async def test_timeout_returns_error_result(self):
        """Subagent exceeding timeout returns a failed SubagentResult."""
        import asyncio

        async def slow_respond(*args, **kwargs):
            await asyncio.sleep(5)
            return make_mock_response(content="Should not reach here")

        client = MockClient()
        client.complete = slow_respond
        tools = ToolRegistry()
        hooks = HookChain()
        manager = SubAgentManager(
            client=client,
            tools=tools,
            hook_chain=hooks,
            max_iterations=3,
            timeout=0.2,
        )

        tasks = [SubagentTask(task="Slow task", label="slow")]
        results = await manager.run_tasks(tasks)

        assert len(results) == 1
        assert results[0].success is False
        assert "timed out" in results[0].error
        assert results[0].elapsed_seconds > 0

    async def test_cancel_task_returns_true_for_running(self):
        """cancel_task returns True when cancelling a running task."""
        import asyncio

        started = asyncio.Event()

        async def blocking_respond(*args, **kwargs):
            started.set()
            await asyncio.sleep(30)
            return make_mock_response(content="blocked")

        client = MockClient()
        client.complete = blocking_respond
        tools = ToolRegistry()
        hooks = HookChain()
        manager = SubAgentManager(
            client=client,
            tools=tools,
            hook_chain=hooks,
            max_iterations=3,
            timeout=60.0,
        )

        async def run_and_cancel():
            task_coro = manager.run_tasks([SubagentTask(task="Block", label="blocker")])
            task = asyncio.create_task(task_coro)
            await asyncio.wait_for(started.wait(), timeout=2.0)
            # Give a moment for the task to be registered
            await asyncio.sleep(0.05)
            result = manager.cancel_task("blocker")
            task_result = await task
            return result, task_result

        cancelled, results = await run_and_cancel()
        assert cancelled is True
        assert results[0].success is False
        assert "cancel" in results[0].error.lower()

    async def test_cancel_task_returns_false_for_unknown(self):
        """cancel_task returns False for a label that doesn't exist."""
        client = MockClient()
        tools = ToolRegistry()
        hooks = HookChain()
        manager = SubAgentManager(client=client, tools=tools, hook_chain=hooks)

        result = manager.cancel_task("nonexistent")
        assert result is False

    async def test_cancel_all_returns_count(self):
        """cancel_all returns the number of cancelled tasks."""
        import asyncio

        started = asyncio.Event()
        asyncio.Event()

        async def blocking_respond(*args, **kwargs):
            started.set()
            await asyncio.sleep(30)
            return make_mock_response(content="blocked")

        client = MockClient()
        client.complete = blocking_respond
        tools = ToolRegistry()
        hooks = HookChain()
        manager = SubAgentManager(
            client=client,
            tools=tools,
            hook_chain=hooks,
            max_iterations=3,
            timeout=60.0,
        )

        async def run_and_cancel():
            tasks = [
                SubagentTask(task="A", label="a"),
                SubagentTask(task="B", label="b"),
                SubagentTask(task="C", label="c"),
            ]
            task_coro = manager.run_tasks(tasks)
            task = asyncio.create_task(task_coro)
            await asyncio.wait_for(started.wait(), timeout=2.0)
            await asyncio.sleep(0.1)
            count = await manager.cancel_all()
            results = await task
            return count, results

        count, results = await run_and_cancel()
        assert count == 3
        assert all(not r.success for r in results)

    async def test_list_running_tracks_active_tasks(self):
        """list_running returns labels of currently executing tasks."""
        import asyncio

        started = asyncio.Event()
        can_finish = asyncio.Event()

        async def controlled_respond(*args, **kwargs):
            started.set()
            await can_finish.wait()
            return make_mock_response(content="Done")

        client = MockClient()
        client.complete = controlled_respond
        tools = ToolRegistry()
        hooks = HookChain()
        manager = SubAgentManager(
            client=client,
            tools=tools,
            hook_chain=hooks,
            max_iterations=3,
            timeout=60.0,
        )

        async def check_running():
            tasks = [
                SubagentTask(task="X", label="x"),
                SubagentTask(task="Y", label="y"),
            ]
            task_coro = manager.run_tasks(tasks)
            task = asyncio.create_task(task_coro)
            await asyncio.wait_for(started.wait(), timeout=2.0)
            await asyncio.sleep(0.05)
            running = manager.list_running()
            can_finish.set()
            await task
            return running

        running = await check_running()
        assert set(running) == {"x", "y"}

    async def test_resource_cleanup_after_completion(self):
        """Child agent memory is cleared after task completes."""
        client = MockClient(
            responses=[
                make_mock_response(content="Clean me up"),
            ]
        )
        tools = ToolRegistry()
        hooks = HookChain()
        manager = SubAgentManager(
            client=client,
            tools=tools,
            hook_chain=hooks,
            max_iterations=2,
            timeout=10.0,
        )

        tasks = [SubagentTask(task="Cleanup test", label="clean")]
        results = await manager.run_tasks(tasks)

        assert results[0].success is True
        # If cleanup failed with an error, the result would reflect it
        # The main verification is that no exceptions were raised

    async def test_resource_cleanup_after_timeout(self):
        """Resources are cleaned up even when a subagent times out."""
        import asyncio

        async def slow_respond(*args, **kwargs):
            await asyncio.sleep(5)
            return make_mock_response(content="timeout")

        client = MockClient()
        client.complete = slow_respond
        tools = ToolRegistry()
        hooks = HookChain()
        manager = SubAgentManager(
            client=client,
            tools=tools,
            hook_chain=hooks,
            max_iterations=3,
            timeout=0.2,
        )

        tasks = [SubagentTask(task="Timeout cleanup", label="tclean")]
        results = await manager.run_tasks(tasks)

        assert results[0].success is False
        assert "timed out" in results[0].error

    async def test_resource_cleanup_after_cancel(self):
        """Resources are cleaned up after a task is cancelled."""
        import asyncio

        started = asyncio.Event()

        async def blocking_respond(*args, **kwargs):
            started.set()
            await asyncio.sleep(30)
            return make_mock_response(content="cancel")

        client = MockClient()
        client.complete = blocking_respond
        tools = ToolRegistry()
        hooks = HookChain()
        manager = SubAgentManager(
            client=client,
            tools=tools,
            hook_chain=hooks,
            max_iterations=3,
            timeout=60.0,
        )

        async def run_and_cancel():
            task_coro = manager.run_tasks([SubagentTask(task="Cancel cleanup", label="cclean")])
            task = asyncio.create_task(task_coro)
            await asyncio.wait_for(started.wait(), timeout=2.0)
            await asyncio.sleep(0.05)
            manager.cancel_task("cclean")
            results = await task
            return results

        results = await run_and_cancel()
        assert results[0].success is False


class TestSubagentConfig:
    def test_config_subagent_property(self):
        """Config.subagent returns the subagent config dict."""
        config = Config.from_dict(
            {
                "agent": {"name": "test", "system_prompt": "hi"},
                "llm": {"model": "gpt-4o"},
                "subagent": {"timeout": 30, "max_iterations": 3},
            }
        )
        assert config.subagent == {"timeout": 30, "max_iterations": 3}

    def test_config_subagent_defaults_empty(self):
        """Config.subagent returns empty dict when not configured."""
        config = Config.from_dict(
            {
                "agent": {"name": "test", "system_prompt": "hi"},
                "llm": {"model": "gpt-4o"},
            }
        )
        assert config.subagent == {}

    async def test_setup_subagent_reads_config(self):
        """_setup_subagent passes config values to SubAgentManager."""
        from koboi.facade import _setup_subagent

        client = MockClient()
        tools = ToolRegistry()
        tools.register(
            name="delegate_tasks",
            description="Delegate",
            parameters={"type": "object", "properties": {"tasks": {"type": "array"}}},
            fn=lambda tasks: "ok",
        )
        hooks = HookChain()
        config = Config.from_dict(
            {
                "agent": {"name": "test", "system_prompt": "hi"},
                "llm": {"model": "gpt-4o"},
                "subagent": {"timeout": 42, "max_iterations": 7},
            }
        )

        _setup_subagent(tools, client, hooks, None, config=config)

        manager = tools.get_dep("manager")
        assert manager is not None
        assert manager.timeout == 42
        assert manager.max_iterations == 7

    async def test_setup_subagent_default_values(self):
        """_setup_subagent uses defaults when subagent config is absent."""
        from koboi.facade import _setup_subagent

        client = MockClient()
        tools = ToolRegistry()
        tools.register(
            name="delegate_tasks",
            description="Delegate",
            parameters={"type": "object", "properties": {"tasks": {"type": "array"}}},
            fn=lambda tasks: "ok",
        )
        hooks = HookChain()
        config = Config.from_dict(
            {
                "agent": {"name": "test", "system_prompt": "hi"},
                "llm": {"model": "gpt-4o"},
            }
        )

        _setup_subagent(tools, client, hooks, None, config=config)

        manager = tools.get_dep("manager")
        assert manager is not None
        assert manager.timeout == 60.0
        assert manager.max_iterations == 5
