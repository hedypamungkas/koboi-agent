"""Tests for developer mode features: __version__, ConfigBuilder, context manager, agent.on()."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from koboi.config import Config, ConfigBuilder
from koboi.hooks.chain import HookEvent, HookContext


# ─── Helpers ────────────────────────────────────────────────────────────────


def _write_config(tmp_path, config_data: dict) -> str:
    path = tmp_path / "test_config.yaml"
    with open(path, "w") as f:
        yaml.dump(config_data, f)
    return str(path)


def _base_config_data() -> dict:
    return {
        "agent": {"name": "test-agent", "system_prompt": "You are helpful.", "max_iterations": 3},
        "llm": {"model": "gpt-4o-mini", "api_key": "test-key", "base_url": "http://localhost:8080/v1"},
    }


# ─── __version__ ────────────────────────────────────────────────────────────


class TestVersion:
    def test_version_is_string(self):
        import koboi

        assert isinstance(koboi.__version__, str)
        assert len(koboi.__version__) > 0

    def test_version_matches_pyproject(self):
        import koboi

        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        if pyproject.exists():
            content = pyproject.read_text()
            for line in content.splitlines():
                if line.strip().startswith("version"):
                    expected = line.split("=")[1].strip().strip('"').strip("'")
                    assert koboi.__version__ == expected
                    return
        # If pyproject.toml not found or no version line, just check it's a valid version
        assert "." in koboi.__version__

    def test_version_in_all(self):
        import koboi

        assert "__version__" in koboi.__all__


# ─── ConfigBuilder ──────────────────────────────────────────────────────────


class TestConfigBuilder:
    def test_builder_minimal(self):
        config = Config.builder().agent(name="test").llm(model="gpt-4o-mini").build()
        assert config.agent_name == "test"
        assert config.model == "gpt-4o-mini"

    def test_builder_full(self):
        config = (
            Config.builder()
            .agent(name="full-agent", system_prompt="Be helpful", max_iterations=5, description="A test agent")
            .llm(provider="openai", model="gpt-4o", api_key="sk-test", temperature=0.7)
            .tools(builtin=["calculator", "web_search"])
            .rag(enabled=True, documents=["data/sample/product_catalog.md"], top_k=5)
            .context(strategy="smart_truncation", max_context_tokens=8000)
            .guardrails(input={"max_length": 10000}, rate_limit={"max_calls_per_minute": 20})
            .memory(backend="sqlite", db_path="test.db")
            .harness(telemetry=True, carryover=True)
            .tracing(provider="langfuse", base_url="http://localhost:3300")
            .policy(rules=[{"tool": "shell", "action": "deny"}])
            .skills(search_paths=["./skills"])
            .mcp(servers=[{"command": "python", "args": ["server.py"]}])
            .build()
        )
        assert config.agent_name == "full-agent"
        assert config.system_prompt == "Be helpful"
        assert config.max_iterations == 5
        assert config.provider == "openai"
        assert config.model == "gpt-4o"
        assert config.api_key == "sk-test"
        assert config.temperature == 0.7
        assert config.rag_enabled is True
        assert config.get("rag", "top_k") == 5
        assert config.get("rag", "documents") == [{"path": "data/sample/product_catalog.md"}]
        assert config.get("context", "strategy") == "smart_truncation"
        assert config.get("context", "max_context_tokens") == 8000
        assert config.get("guardrails", "input", "max_length") == 10000
        assert config.get("guardrails", "rate_limit", "max_calls_per_minute") == 20
        assert config.get("memory", "backend") == "sqlite"
        assert config.get("harness", "telemetry") is True
        assert config.get("tracing", "provider") == "langfuse"
        assert config.get("policy", "rules") == [{"tool": "shell", "action": "deny"}]
        assert config.get("skills", "search_paths") == ["./skills"]
        assert config.get("mcp", "servers") == [{"command": "python", "args": ["server.py"]}]

    def test_builder_produces_valid_config(self):
        config = Config.builder().agent(name="x").llm(model="y").build()
        assert isinstance(config, Config)
        assert config.raw is not None
        assert isinstance(config.raw, dict)

    def test_builder_validation_error_missing_name(self):
        with pytest.raises(ValueError, match="agent.name is required"):
            Config.builder().llm(model="gpt-4o-mini").build()

    def test_builder_validation_error_missing_model(self):
        with pytest.raises(ValueError, match="llm.model is required"):
            Config.builder().agent(name="test").build()

    def test_builder_validation_error_missing_both(self):
        with pytest.raises(ValueError, match="agent.name is required"):
            Config.builder().build()

    def test_builder_env_resolution(self, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "resolved-secret-key")
        config = Config.builder().agent(name="env-test").llm(model="gpt-4", api_key="${TEST_API_KEY}").build()
        assert config.api_key == "resolved-secret-key"

    def test_builder_env_resolution_with_default(self):
        config = Config.builder().agent(name="env-test").llm(model="gpt-4", api_key="${MISSING_KEY:fallback}").build()
        assert config.api_key == "fallback"

    def test_builder_fluent_chaining(self):
        builder = Config.builder()
        result = builder.agent(name="test")
        assert result is builder
        result = builder.llm(model="gpt-4")
        assert result is builder
        result = builder.tools(builtin=["calculator"])
        assert result is builder
        result = builder.rag(enabled=True)
        assert result is builder
        result = builder.context(strategy="noop")
        assert result is builder
        result = builder.guardrails(input={"max_length": 100})
        assert result is builder
        result = builder.memory(backend="in_memory")
        assert result is builder
        result = builder.harness(telemetry=True)
        assert result is builder
        result = builder.tracing(provider="langfuse")
        assert result is builder
        result = builder.policy(rules=[])
        assert result is builder
        result = builder.skills(search_paths=[])
        assert result is builder
        result = builder.mcp(servers=[])
        assert result is builder

    def test_builder_defaults(self):
        config = Config.builder().agent(name="d").llm(model="m").build()
        assert config.provider == "openai"
        assert config.max_iterations == 10
        assert config.llm_timeout == 120.0
        assert config.llm_max_tokens == 4096
        assert config.max_retries == 3
        assert config.rag_enabled is False

    def test_builder_rag_documents_as_strings(self):
        config = (
            Config.builder().agent(name="rag-test").llm(model="gpt-4").rag(documents=["doc1.md", "doc2.md"]).build()
        )
        docs = config.get("rag", "documents")
        assert docs == [{"path": "doc1.md"}, {"path": "doc2.md"}]

    def test_builder_rag_documents_as_dicts(self):
        config = (
            Config.builder()
            .agent(name="rag-test")
            .llm(model="gpt-4")
            .rag(documents=[{"path": "doc1.md", "title": "Doc 1"}])
            .build()
        )
        docs = config.get("rag", "documents")
        assert docs == [{"path": "doc1.md", "title": "Doc 1"}]

    def test_builder_custom_tools(self):
        config = (
            Config.builder()
            .agent(name="custom-tools")
            .llm(model="gpt-4")
            .tools(custom=[{"module": "my.module", "function": "my_func"}])
            .build()
        )
        assert config.get("tools", "custom") == [{"module": "my.module", "function": "my_func"}]

    def test_builder_multiple_calls_merge(self):
        config = Config.builder().agent(name="merge-test").llm(model="gpt-4").llm(api_key="sk-new").build()
        assert config.model == "gpt-4"
        assert config.api_key == "sk-new"

    def test_builder_from_classmethod(self):
        builder = Config.builder()
        assert isinstance(builder, ConfigBuilder)


# ─── Async Context Manager ─────────────────────────────────────────────────


class TestContextManager:
    async def test_context_manager_enter_returns_self(self, tmp_path):
        from koboi.facade import KoboiAgent

        path = _write_config(tmp_path, _base_config_data())
        agent = KoboiAgent.from_config(path)
        async with agent as a:
            assert a is agent
        # cleanup already happened

    async def test_context_manager_closes_client(self, tmp_path):
        from koboi.facade import KoboiAgent

        path = _write_config(tmp_path, _base_config_data())
        agent = KoboiAgent.from_config(path)
        closed = []
        original_close = agent._core.client.close

        async def tracking_close():
            closed.append("client")
            await original_close()

        agent._core.client.close = tracking_close
        async with agent:
            pass
        assert "client" in closed

    async def test_context_manager_closes_mcp_clients(self, tmp_path):
        from koboi.facade import KoboiAgent

        path = _write_config(tmp_path, _base_config_data())
        agent = KoboiAgent.from_config(path)

        closed = []
        mock_mcp1 = type("MCP1", (), {"close": lambda self: closed.append("mcp1")})()
        mock_mcp2 = type("MCP2", (), {"close": lambda self: closed.append("mcp2")})()
        agent._mcp_clients = [mock_mcp1, mock_mcp2]

        async with agent:
            pass
        assert "mcp1" in closed
        assert "mcp2" in closed

    async def test_context_manager_closes_sqlite_memory(self, tmp_path):
        from koboi.facade import KoboiAgent

        config_data = _base_config_data()
        config_data["memory"] = {"backend": "sqlite", "db_path": str(tmp_path / "test.db")}
        path = _write_config(tmp_path, config_data)
        agent = KoboiAgent.from_config(path)

        closed = []
        original_close = agent._core.memory.close

        def tracking_close():
            closed.append("sqlite")
            original_close()

        agent._core.memory.close = tracking_close
        async with agent:
            pass
        assert "sqlite" in closed

    async def test_context_manager_handles_close_errors(self, tmp_path):
        from koboi.facade import KoboiAgent

        path = _write_config(tmp_path, _base_config_data())
        agent = KoboiAgent.from_config(path)

        broken_mcp = type("BrokenMCP", (), {"close": lambda self: (_ for _ in ()).throw(RuntimeError("broken"))})()
        agent._mcp_clients = [broken_mcp]

        # Should not raise
        async with agent:
            pass

    async def test_close_without_context_manager(self, tmp_path):
        from koboi.facade import KoboiAgent

        path = _write_config(tmp_path, _base_config_data())
        agent = KoboiAgent.from_config(path)
        closed = []
        original_close = agent._core.client.close

        async def tracking_close():
            closed.append("client")
            await original_close()

        agent._core.client.close = tracking_close
        await agent.close()
        assert "client" in closed

    async def test_mcp_clients_stored_on_agent(self, tmp_path):
        from koboi.facade import KoboiAgent

        path = _write_config(tmp_path, _base_config_data())
        agent = KoboiAgent.from_config(path)
        assert hasattr(agent, "_mcp_clients")
        assert isinstance(agent._mcp_clients, list)


# ─── agent.on() ─────────────────────────────────────────────────────────────


class TestAgentOn:
    def _make_agent(self, tmp_path):
        from koboi.facade import KoboiAgent

        path = _write_config(tmp_path, _base_config_data())
        return KoboiAgent.from_config(path)

    def test_on_with_string_event(self, tmp_path):
        agent = self._make_agent(tmp_path)
        callback = lambda ctx: ctx
        result = agent.on("post_output", callback)
        assert result is agent  # chaining
        hook_names = [h["name"] for h in agent._core.hooks.list_hooks()]
        assert "CallbackHook" in hook_names

    def test_on_with_hook_event_enum(self, tmp_path):
        agent = self._make_agent(tmp_path)
        agent.on(HookEvent.POST_OUTPUT, lambda ctx: ctx)
        hooks = agent._core.hooks.list_hooks()
        callback_hooks = [h for h in hooks if h["name"] == "CallbackHook"]
        assert len(callback_hooks) == 1
        assert "post_output" in callback_hooks[0]["events"]

    def test_on_with_list_of_events(self, tmp_path):
        agent = self._make_agent(tmp_path)
        agent.on(["pre_tool_use", "post_tool_use"], lambda ctx: ctx)
        hooks = agent._core.hooks.list_hooks()
        callback_hooks = [h for h in hooks if h["name"] == "CallbackHook"]
        assert len(callback_hooks) == 1
        events = callback_hooks[0]["events"]
        assert "pre_tool_use" in events
        assert "post_tool_use" in events

    def test_on_returns_self(self, tmp_path):
        agent = self._make_agent(tmp_path)
        result = agent.on("post_output", lambda ctx: ctx)
        assert result is agent

    def test_on_chaining(self, tmp_path):
        agent = self._make_agent(tmp_path)
        agent.on("post_output", lambda ctx: ctx).on("pre_tool_use", lambda ctx: ctx)
        hooks = agent._core.hooks.list_hooks()
        callback_hooks = [h for h in hooks if h["name"] == "CallbackHook"]
        assert len(callback_hooks) == 2

    async def test_on_sync_callback_invoked(self, tmp_path):
        agent = self._make_agent(tmp_path)
        calls = []
        agent.on("post_output", lambda ctx: calls.append(ctx.event.value) or ctx)

        ctx = HookContext(event=HookEvent.POST_OUTPUT)
        await agent._core.hooks.emit(ctx)
        assert "post_output" in calls

    async def test_on_async_callback_invoked(self, tmp_path):
        agent = self._make_agent(tmp_path)
        calls = []

        async def async_handler(ctx):
            calls.append(ctx.event.value)
            return ctx

        agent.on("post_output", async_handler)
        ctx = HookContext(event=HookEvent.POST_OUTPUT)
        await agent._core.hooks.emit(ctx)
        assert "post_output" in calls

    async def test_on_callback_receives_context(self, tmp_path):
        agent = self._make_agent(tmp_path)
        received = []
        agent.on("pre_tool_use", lambda ctx: received.append(ctx) or ctx)

        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="my_tool")
        await agent._core.hooks.emit(ctx)
        assert len(received) == 1
        assert received[0].tool_name == "my_tool"
        assert received[0].event == HookEvent.PRE_TOOL_USE

    def test_on_invalid_event_string(self, tmp_path):
        agent = self._make_agent(tmp_path)
        with pytest.raises(ValueError, match="Unknown event"):
            agent.on("nonexistent_event", lambda ctx: ctx)

    def test_on_with_mixed_string_and_enum(self, tmp_path):
        agent = self._make_agent(tmp_path)
        agent.on(["post_output", HookEvent.PRE_TOOL_USE], lambda ctx: ctx)
        hooks = agent._core.hooks.list_hooks()
        callback_hooks = [h for h in hooks if h["name"] == "CallbackHook"]
        assert len(callback_hooks) == 1
        events = callback_hooks[0]["events"]
        assert "post_output" in events
        assert "pre_tool_use" in events


# ─── Integration ─────────────────────────────────────────────────────────────


class TestDeveloperWorkflowIntegration:
    async def test_full_developer_workflow(self, tmp_path):
        """Config.builder() -> KoboiAgent -> on() -> run() -> close()."""
        from koboi.facade import KoboiAgent
        from tests.conftest import MockClient, make_mock_response

        config = (
            Config.builder()
            .agent(name="dev-agent", system_prompt="You are a dev agent.", max_iterations=3)
            .llm(model="gpt-4o-mini", api_key="test-key", base_url="http://localhost:8080/v1")
            .build()
        )

        # Create agent from built config
        path = _write_config(tmp_path, config.raw)
        agent = KoboiAgent.from_config(path)

        # Replace client with mock
        agent._core.client = MockClient(
            responses=[
                make_mock_response(content="Hello from dev agent!"),
            ]
        )

        # Register event callback
        output_received = []
        agent.on("post_output", lambda ctx: output_received.append(True) or ctx)

        # Use as context manager
        async with agent as a:
            result = await a.run("Say hello")
            assert "Hello from dev agent!" in result.content
            assert result.success is True
