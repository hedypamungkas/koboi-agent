"""koboi/facade.py -- KoboiAgent: async facade pattern entry point.

Single class that hides all subsystem complexity. Creates everything from
YAML config and delegates to AgentCore.
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable

from koboi.config import Config
from koboi.client import RetryClient
from koboi.memory import ConversationMemory
from koboi.modes import AgentMode, ModeManager
from koboi.tools.registry import ToolRegistry, register_decorated
from koboi.logger import AgentLogger
from koboi.types import RunResult, RiskLevel
from koboi.hooks.chain import HookEvent

if TYPE_CHECKING:
    from koboi.events import StreamEvent
    from koboi.hooks.chain import HookChain
    from koboi.loop import AgentCore


class KoboiAgent:
    """Facade for the koboi-agent framework.

    Usage:
        agent = KoboiAgent.from_config("configs/sales_agent.yaml")
        result = await agent.run("What products are available?")

        # Or as async context manager:
        async with KoboiAgent.from_config("config.yaml") as agent:
            result = await agent.run("Hello")
    """

    def __init__(
        self,
        core: AgentCore | None = None,
        config: Config | None = None,
        logger: AgentLogger | None = None,
        mcp_clients: list | None = None,
        mode_manager: ModeManager | None = None,
        trust_db: object | None = None,
        orchestrator: object | None = None,
    ):
        self._core = core
        self._config = config
        self._logger = logger
        self._mcp_clients = mcp_clients or []
        self._sync_loop: asyncio.AbstractEventLoop | None = None
        self._bg_loop: asyncio.AbstractEventLoop | None = None
        self._bg_thread: object | None = None  # threading.Thread
        self._mode_manager = mode_manager
        self._trust_db = trust_db
        self._orchestrator = orchestrator

    @classmethod
    def from_config(cls, config_path: str | Path, verbose: bool = False) -> KoboiAgent:
        """Factory method: create a KoboiAgent from YAML config."""
        config = Config.from_yaml(config_path)
        return cls._from_config(config, verbose=verbose)

    @classmethod
    def from_dict(cls, data: dict, verbose: bool = False) -> KoboiAgent:
        """Factory method: create a KoboiAgent from a Python dict.

        Usage:
            agent = KoboiAgent.from_dict({
                "agent": {"name": "my-agent", "system_prompt": "You are helpful"},
                "llm": {"provider": "openai", "model": "gpt-4o"},
            })
        """
        config = Config.from_dict(data)
        return cls._from_config(config, verbose=verbose)

    @classmethod
    def from_config_string(cls, yaml_string: str, verbose: bool = False) -> KoboiAgent:
        """Factory method: create a KoboiAgent from a YAML string.

        Usage:
            agent = KoboiAgent.from_config_string('''
                agent:
                  name: my-agent
                llm:
                  provider: openai
                  model: gpt-4o
            ''')
        """
        config = Config.from_string(yaml_string)
        return cls._from_config(config, verbose=verbose)

    @classmethod
    def _from_config(cls, config: Config, verbose: bool = False) -> KoboiAgent:
        """Shared builder: assemble all subsystems from a Config object."""
        # Orchestration mode: transparent to caller
        if config.orchestration.get("enabled"):
            return _build_orchestration(config, verbose=verbose)

        assembler = AgentAssembler(config, verbose=verbose)
        return assembler.build()

    async def run(self, message: str | list) -> RunResult:
        if self._orchestrator is not None:
            return await _run_orchestrator(self._orchestrator, message)
        return await self._core.run(message)

    async def run_stream(self, message: str | list) -> AsyncGenerator[StreamEvent, None]:
        if self._orchestrator is not None:
            async for event in self._orchestrator.run_stream(message):
                yield event
        else:
            async for event in self._core.run_stream(message):
                yield event

    async def chat(self, message: str | list) -> RunResult:
        if self._orchestrator is not None:
            return await _run_orchestrator(self._orchestrator, message)
        return await self._core.chat(message)

    def run_sync(self, message: str | list) -> RunResult:
        """Blocking wrapper for sync callers.

        Works from both sync and async contexts (Jupyter, FastAPI, etc.).
        Uses a persistent event loop to avoid httpx AsyncClient issues
        with loop reuse across multiple calls.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop -- safe to use our own
            if self._sync_loop is None or self._sync_loop.is_closed():
                self._sync_loop = asyncio.new_event_loop()
            return self._sync_loop.run_until_complete(self.run(message))
        else:
            # Already in an async context -- run in a dedicated background loop.
            # Using a persistent thread+loop keeps httpx connections alive across calls.
            if not hasattr(self, "_bg_thread") or self._bg_thread is None or not self._bg_thread.is_alive():
                import threading

                self._bg_loop = asyncio.new_event_loop()
                self._bg_thread = threading.Thread(target=self._bg_loop.run_forever, daemon=True)
                self._bg_thread.start()
            future = asyncio.run_coroutine_threadsafe(self.run(message), self._bg_loop)
            return future.result()

    def reset(self) -> None:
        if self._core is not None:
            self._core.reset()

    async def close(self) -> None:
        """Clean up all resources (MCP subprocesses, HTTP transport, SQLite)."""
        for mcp in self._mcp_clients:
            try:
                mcp.close()
            except Exception:
                pass
        if self._orchestrator is not None:
            # Clean up orchestrator's sub-agent memories
            for agent in getattr(self._orchestrator, "_agents_map", {}).values():
                if hasattr(agent, "memory") and hasattr(agent.memory, "close"):
                    agent.memory.close()
            await self._orchestrator.client.close()
        elif self._core is not None:
            if hasattr(self._core.memory, "close"):
                self._core.memory.close()
            if hasattr(self._core, "audit_trail") and self._core.audit_trail:
                if hasattr(self._core.audit_trail, "close"):
                    self._core.audit_trail.close()
            await self._core.client.close()
        # Clean up logger
        if self._logger is not None:
            self._logger.close()
        # Clean up background event loop if used
        if hasattr(self, "_bg_loop") and self._bg_loop is not None:
            self._bg_loop.call_soon_threadsafe(self._bg_loop.stop)
            if hasattr(self, "_bg_thread") and self._bg_thread is not None:
                self._bg_thread.join(timeout=2.0)
            self._bg_loop.close()
            self._bg_loop = None
            self._bg_thread = None

    def __del__(self) -> None:
        """Best-effort sync cleanup when close() was not called.

        Handles synchronous resources only (MCP subprocesses, background loop,
        logger). HTTP clients and SQLite connections are safe under CPython
        reference counting and will be collected by the GC.
        """
        for mcp in self._mcp_clients:
            try:
                mcp.close()
            except Exception:
                pass
        if self._logger is not None:
            try:
                self._logger.close()
            except Exception:
                pass
        bg_loop = getattr(self, "_bg_loop", None)
        if bg_loop is not None:
            try:
                bg_loop.call_soon_threadsafe(bg_loop.stop)
            except Exception:
                pass
            bg_thread = getattr(self, "_bg_thread", None)
            if bg_thread is not None:
                try:
                    bg_thread.join(timeout=1.0)
                except Exception:
                    pass
            try:
                bg_loop.close()
            except Exception:
                pass

    async def __aenter__(self) -> KoboiAgent:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    def on(
        self,
        event: str | HookEvent | list[str | HookEvent],
        callback: Callable,
    ) -> KoboiAgent:
        """Register an event callback (sync or async).

        Usage:
            agent.on("tool_use", lambda ctx: print(ctx.tool_name))
            agent.on(HookEvent.POST_OUTPUT, my_handler)
            agent.on(["pre_tool_use", "post_tool_use"], my_handler)
        """
        if isinstance(event, (str, HookEvent)):
            event = [event]
        events: list[HookEvent] = []
        for e in event:
            if isinstance(e, HookEvent):
                events.append(e)
            else:
                try:
                    events.append(HookEvent(e))
                except ValueError:
                    valid = [ev.value for ev in HookEvent]
                    raise ValueError(f"Unknown event '{e}'. Valid events: {valid}")
        from koboi.hooks.callback_hook import CallbackHook

        if self._core is not None:
            self._core.hooks.add(CallbackHook(callback=callback, events=events))
        return self

    def add_hook(
        self,
        callback: Callable | Awaitable,
        events: list[HookEvent] | None = None,
    ) -> None:
        """Register a callback as a hook without subclassing Hook."""
        from koboi.hooks.callback_hook import CallbackHook

        if self._core is not None:
            self._core.hooks.add(CallbackHook(callback=callback, events=events))

    def add_tool(
        self,
        name: str,
        fn: Callable,
        description: str,
        parameters: dict,
        risk_level: RiskLevel = RiskLevel.SAFE,
    ) -> None:
        """Register a tool on the agent."""
        if self._core is not None:
            self._core.tools.register(name, description, parameters, fn, risk_level=risk_level)

    def inject_tool_definitions(self, tool_definitions: list[dict]) -> None:
        """Register external tool definitions (e.g., from eval cases)."""
        if self._core is None:
            return
        tools = self._core.tools
        if not tools:
            return

        def _dummy_handler(**kwargs):
            return '{"status": "ok"}'

        for tool_def in tool_definitions:
            fn_def = tool_def.get("function", tool_def)
            name = fn_def.get("name", "")
            if not name or name in tools:
                continue
            params = fn_def.get("parameters", {})
            if params.get("type") == "dict":
                params = {**params, "type": "object"}
            tools.register(
                name=name,
                description=fn_def.get("description", ""),
                parameters=params,
                fn=_dummy_handler,
            )

    def get_telemetry(self) -> object | None:
        """Return the TelemetryCollector from the hook chain, if present."""
        if self._core is None:
            return None
        tel = getattr(self._core, "telemetry", None)
        if tel:
            return tel
        found = self._core.hooks.find_hook(lambda h: hasattr(h, "telemetry"))
        return found.telemetry if found else None

    def ensure_telemetry_hook(self) -> None:
        """Attach a TelemetryHook if not already present."""
        if self._core is None:
            return
        if self._core.hooks.find_hook(lambda h: hasattr(h, "telemetry")):
            return
        try:
            from koboi.hooks.telemetry_hook import TelemetryHook
            from koboi.harness.telemetry import TelemetryCollector

            self._core.hooks.add(TelemetryHook(telemetry=TelemetryCollector()))
        except ImportError:
            pass

    def push_langfuse_scores(self, trace_id: str, scores: list) -> None:
        """Push evaluation scores to Langfuse via the tracing hook."""
        if self._core is None:
            return
        hook = self._core.hooks.find_hook(lambda h: type(h).__name__ == "LangfuseTracingHook")
        if hook is None or not hasattr(hook, "available") or not hook.available:
            return
        client = hook.get_client()
        if not client:
            return
        try:
            for s in scores:
                client.score(
                    trace_id=trace_id,
                    name=s.name,
                    value=s.value,
                    comment=s.reason,
                )
            client.flush()
        except Exception as e:
            import logging

            logging.getLogger(__name__).debug("Failed to push scores to Langfuse: %s", e)

    def replace_from(self, other: KoboiAgent) -> None:
        """Replace internal state from another agent (used by /run hot-load)."""
        self._core = other._core
        self._config = other._config
        self._mcp_clients = other._mcp_clients
        self._mode_manager = other._mode_manager
        self._trust_db = other._trust_db

    @property
    def config(self) -> Config:
        return self._config

    @property
    def core(self) -> AgentCore | None:
        return self._core

    @property
    def orchestrator(self) -> object | None:
        return self._orchestrator

    @property
    def mode_manager(self) -> ModeManager | None:
        return self._mode_manager

    @property
    def trust_db(self) -> object | None:
        return self._trust_db


def _build_client(config: Config, logger: AgentLogger) -> RetryClient:
    return RetryClient(
        provider=config.provider,
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        logger=logger,
        timeout=config.llm_timeout,
        max_tokens=config.llm_max_tokens,
        auth_token=config.llm_auth_token,
        auth_type=config.auth_type,
        max_retries=config.max_retries,
        retry_backoff_base=config.retry_backoff_base,
        temperature=config.temperature,
    )


def _build_tools(config: Config) -> ToolRegistry:
    registry = ToolRegistry()
    tool_defaults = config.get("tools", "defaults", default={})
    tool_overrides = config.get("tools", "overrides", default={})
    if tool_defaults or tool_overrides:
        registry.set_tool_config(tool_defaults, tool_overrides)
    builtin_list = config.get("tools", "builtin", default=[])
    if builtin_list:
        from koboi.tools.builtin import register_all

        register_all(registry)
        # Inject per-agent memory store so agents don't share state
        from koboi.tools.builtin.memory import _MemoryStore

        memory_file = config.get("tools", "memory_file", default=".agent_memory.json")
        registry.set_dep("memory_store_ref", _MemoryStore(filepath=memory_file))
        if builtin_list and isinstance(builtin_list, list):
            registry.keep_only(builtin_list)

    custom_list = config.get("tools", "custom", default=[])
    if custom_list:
        for tool_conf in custom_list:
            module_name = tool_conf.get("module", "")
            if module_name:
                try:
                    mod = importlib.import_module(module_name)
                    register_decorated(registry, mod)
                except ImportError as e:
                    import logging

                    logging.getLogger(__name__).warning("Failed to import custom tool module '%s': %s", module_name, e)

    return registry


def _build_context(config: Config, logger: AgentLogger, client: RetryClient | None = None):
    strategy = config.get("context", "strategy", default="noop")
    if strategy == "noop":
        return None

    from koboi.context.registry import build_context, load_custom_context_modules

    # Ensure built-in strategies are registered
    import koboi.context.manager  # noqa: F401

    custom_modules = config.get("context", "custom_modules", default=[])
    if custom_modules:
        load_custom_context_modules(custom_modules)

    kwargs: dict = {}
    keep_last = config.get("context", "keep_last")
    if keep_last is not None:
        kwargs["keep_last"] = keep_last
    summarization_truncation = config.get("context", "summarization_truncation")
    if summarization_truncation is not None:
        kwargs["summarization_truncation"] = summarization_truncation

    return build_context(strategy, logger=logger, client=client, **kwargs)


def _build_rag(config: Config, client: RetryClient, logger: AgentLogger):
    if not config.rag_enabled:
        return None

    from koboi.rag.registry import build_rag, load_custom_components

    custom_modules = config.get("rag", "custom_modules", default=[])
    if custom_modules:
        load_custom_components(custom_modules)

    rag_dict = config.get("rag", default={})
    rag_dict["enabled"] = True
    if "augmentation" not in rag_dict:
        rag_dict["augmentation"] = "on_the_fly"

    return build_rag(rag_dict, client=client, logger=logger)


def _normalize_guardrail_config(conf: dict | list | None) -> list[dict]:
    """Normalize guardrail config to list-of-dicts format.

    Supports legacy single-dict format (auto-wrapped) and new list format.
    Empty/None returns empty list.
    """
    if not conf:
        return []
    if isinstance(conf, dict):
        # Legacy: single guardrail as dict -> wrap in list, inferring name
        if "name" in conf:
            return [conf]
        # Legacy: config block like {max_length: 100} -> wrap with default name
        return [{"name": "injection_detector", **conf}]
    if isinstance(conf, list):
        return [c for c in conf if isinstance(c, dict) and c.get("name")]
    return []


def _build_guardrails(config: Config, logger: AgentLogger | None = None):
    from koboi.guardrails.registry import GuardrailRegistry

    input_grds: list = []
    output_grds: list = []
    rate_limiter = None
    audit_trail = None

    # Input guardrails -- supports both legacy and new config formats
    input_conf = config.get("guardrails", "input", default={})
    input_configs = _normalize_guardrail_config(input_conf)
    if input_configs:
        input_grds = GuardrailRegistry.from_config(input_configs)
    elif input_conf:
        # Legacy: bare dict without "name" key -> default injection_detector
        input_grds = GuardrailRegistry.from_config([{"name": "injection_detector", **input_conf}])

    # Output guardrails
    output_conf = config.get("guardrails", "output", default={})
    output_configs = _normalize_guardrail_config(output_conf)
    if output_configs:
        output_grds = GuardrailRegistry.from_config(output_configs)
    elif output_conf:
        output_grds = GuardrailRegistry.from_config([{"name": "content_filter"}])

    rl_conf = config.get("guardrails", "rate_limit", default={})
    if rl_conf:
        from koboi.guardrails.rate_limiter import RateLimiter
        from koboi.types import RateLimitConfig

        rate_limiter = RateLimiter(
            config=RateLimitConfig(
                max_tool_calls_per_session=rl_conf.get("max_calls_per_session", 100),
                max_calls_per_minute=rl_conf.get("max_calls_per_minute", 20),
                rate_window_seconds=rl_conf.get("rate_window_seconds", 60.0),
            ),
        )

    audit_conf = config.get("guardrails", "audit", default={})
    if audit_conf:
        db_path = audit_conf.get("db_path") if isinstance(audit_conf, dict) else None
        if db_path:
            from koboi.guardrails.audit import SQLiteAuditTrail

            audit_trail = SQLiteAuditTrail(db_path=db_path, logger=logger)
        else:
            from koboi.guardrails.audit import AuditTrail

            audit_trail = AuditTrail(logger=logger)

    return input_grds, output_grds, rate_limiter, audit_trail


def _build_approval(config: Config):
    handler_conf = config.get("guardrails", "approval", default={})
    handler_type = handler_conf.get("handler", "auto")
    if handler_type == "cli":
        from koboi.guardrails.approval import CLIApprovalHandler

        return CLIApprovalHandler()
    elif handler_type == "callback":
        from koboi.guardrails.approval import CallbackApprovalHandler

        return CallbackApprovalHandler(handler_conf.get("callback", lambda *a: True))
    return None


def _build_skills(config: Config, logger: AgentLogger):
    search_paths = config.get("skills", "search_paths", default=[])
    if not search_paths:
        return None
    from koboi.skills.registry import SkillRegistry

    registry = SkillRegistry()
    for path in search_paths:
        resolved = str(Path(path).expanduser().resolve())
        registry.discover([resolved])
    return registry


def _build_mode_manager(config: Config) -> ModeManager:
    """Build ModeManager from config."""
    mode_str = config.mode
    try:
        initial_mode = AgentMode(mode_str)
    except ValueError:
        initial_mode = AgentMode.CHAT
    return ModeManager(initial_mode=initial_mode)


def _build_trust_db(config: Config):
    """Build TrustDatabase if graduated permissions are enabled."""
    if not config.graduated_permissions:
        return None
    try:
        from koboi.trust import TrustDatabase

        return TrustDatabase(db_path=config.trust_db_path)
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning("TrustDatabase init failed, graduated permissions disabled: %s", exc)
        return None


def _build_hooks(
    config: Config,
    logger: AgentLogger,
    audit_trail,
    mode_manager: ModeManager | None = None,
    verbose: bool = False,
    policy_engine=None,
    tool_registry=None,
):
    from koboi.hooks.registry import build_hook_chain

    return build_hook_chain(
        config,
        logger,
        audit_trail=audit_trail,
        mode_manager=mode_manager,
        verbose=verbose,
        policy_engine=policy_engine,
        tool_registry=tool_registry,
    )


class AgentAssembler:
    """Assembles a KoboiAgent from a Config with inspectable intermediate state.

    Each build step stores its result as an instance attribute, making the
    dependency graph explicit and each component inspectable before assembly.

    Usage:
        assembler = AgentAssembler(config, verbose=True)
        agent = assembler.build()
        # Inspect intermediate state:
        print(assembler.client, assembler.tools, assembler.hook_chain)
    """

    def __init__(self, config: Config, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        # Intermediate state -- populated by build steps
        self.logger: AgentLogger | None = None
        self.client: RetryClient | None = None
        self.memory: object | None = None
        self.tools: ToolRegistry | None = None
        self.mcp_clients: list | None = None
        self.context_manager: object | None = None
        self.augmentation: object | None = None
        self.input_guardrails: list = []
        self.output_guardrails: list = []
        self.rate_limiter: object | None = None
        self.audit_trail: object | None = None
        self.approval_handler: object | None = None
        self.policy_engine: object | None = None
        self.skills: object | None = None
        self.mode_manager: ModeManager | None = None
        self.trust_db: object | None = None
        self.hook_chain: object | None = None

    def build_logger(self) -> AgentLogger:
        self.logger = AgentLogger(session_id=self.config.agent_name)
        return self.logger

    def build_client(self) -> RetryClient:
        self.client = _build_client(self.config, self.logger)
        return self.client

    def build_memory(self) -> object:
        memory_conf = self.config.get("memory", default={})
        memory_backend = memory_conf.get("backend", "sqlite")
        if memory_backend == "sqlite":
            from koboi.memory_sqlite import SQLiteMemory

            self.memory = SQLiteMemory(
                db_path=memory_conf.get("db_path", "koboi_memory.db"),
                session_id=memory_conf.get("session_id"),
                logger=self.logger,
                system_prompt=self.config.system_prompt or None,
            )
        else:
            self.memory = ConversationMemory(
                logger=self.logger,
                system_prompt=self.config.system_prompt or None,
            )
        return self.memory

    def build_tools(self) -> ToolRegistry:
        self.tools = _build_tools(self.config)
        return self.tools

    def build_mcp(self) -> list:
        self.mcp_clients = _build_mcp(self.config, self.tools, self.logger)
        return self.mcp_clients

    def build_context(self) -> object:
        self.context_manager = _build_context(self.config, self.logger, client=self.client)
        return self.context_manager

    def build_rag(self) -> object:
        self.augmentation = _build_rag(self.config, self.client, self.logger)
        return self.augmentation

    def build_guardrails(self) -> tuple:
        self.input_guardrails, self.output_guardrails, self.rate_limiter, self.audit_trail = _build_guardrails(
            self.config, logger=self.logger
        )
        return self.input_guardrails, self.output_guardrails, self.rate_limiter, self.audit_trail

    def build_approval(self) -> object:
        self.approval_handler = _build_approval(self.config)
        return self.approval_handler

    def build_policy(self) -> object:
        self.policy_engine = _build_policy(self.config)
        return self.policy_engine

    def build_skills(self) -> object:
        self.skills = _build_skills(self.config, self.logger)
        return self.skills

    def build_mode_manager(self) -> ModeManager:
        self.mode_manager = _build_mode_manager(self.config)
        return self.mode_manager

    def build_trust_db(self) -> object:
        self.trust_db = _build_trust_db(self.config)
        return self.trust_db

    def build_hooks(self) -> object:
        self.hook_chain = _build_hooks(
            self.config,
            self.logger,
            self.audit_trail,
            mode_manager=self.mode_manager,
            verbose=self.verbose,
            policy_engine=self.policy_engine,
            tool_registry=self.tools,
        )
        return self.hook_chain

    def build(self) -> KoboiAgent:
        """Run all build steps in dependency order and return assembled agent."""
        self.build_logger()
        self.build_client()
        self.build_memory()
        self.build_tools()
        self.build_mcp()
        self.build_context()
        self.build_rag()
        self.build_guardrails()
        self.build_approval()
        self.build_policy()
        self.build_skills()
        self.build_mode_manager()
        self.build_trust_db()
        self.build_hooks()

        _setup_subagent(self.tools, self.client, self.hook_chain, self.logger, memory=self.memory, config=self.config)
        _setup_tasks(self.tools, self.config, hook_chain=self.hook_chain)

        from koboi.loop import AgentCore

        core = AgentCore(
            client=self.client,
            memory=self.memory,
            tools=self.tools,
            max_iterations=self.config.max_iterations,
            verbose=self.verbose,
            logger=self.logger,
            system_prompt=None,
            context_manager=self.context_manager,
            max_context_tokens=self.config.get("context", "max_context_tokens", default=8000),
            augmentation=self.augmentation,
            input_guardrails=self.input_guardrails,
            output_guardrails=self.output_guardrails,
            rate_limiter=self.rate_limiter,
            audit_trail=self.audit_trail,
            approval_handler=self.approval_handler,
            skills=self.skills,
            hook_chain=self.hook_chain,
            mode_manager=self.mode_manager,
        )

        return KoboiAgent(
            core=core,
            config=self.config,
            logger=self.logger,
            mcp_clients=self.mcp_clients,
            mode_manager=self.mode_manager,
            trust_db=self.trust_db,
        )


def _build_mcp(config: Config, tools: ToolRegistry, logger: AgentLogger) -> list:
    """Connect to MCP servers from config and register their tools."""
    servers = config.get("mcp", "servers", default=[])
    if not servers:
        return []

    from koboi.mcp.base import register_mcp_tools

    clients = []
    for server_conf in servers:
        transport = server_conf.get("transport", "stdio")
        try:
            mcp_client = _create_mcp_client(server_conf, transport, logger)
            mcp_client.connect()
            register_mcp_tools(mcp_client, tools)
            clients.append(mcp_client)
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(
                "MCP server connection failed for '%s': %s",
                server_conf.get("url") or server_conf.get("command", "?"),
                e,
            )

    return clients


def _create_mcp_client(server_conf: dict, transport: str, logger: AgentLogger):
    """Factory: create the right MCPClient subclass based on transport config."""
    if transport == "streamable-http":
        from koboi.mcp.http_client import StreamableHTTPMCPClient

        url = server_conf.get("url", "")
        if not url:
            raise ValueError("streamable-http transport requires 'url'")
        return StreamableHTTPMCPClient(
            url=url,
            logger=logger,
            auth_config=server_conf.get("auth"),
            headers=server_conf.get("headers", {}),
            timeout=server_conf.get("timeout", 30.0),
        )
    else:
        from koboi.mcp.client import MCPClient

        command = server_conf.get("command", "")
        args = server_conf.get("args", [])
        if not command:
            raise ValueError("stdio transport requires 'command'")
        timeout = server_conf.get("timeout", 15.0)
        return MCPClient(server_command=[command] + args, logger=logger, connect_timeout=timeout)


def _build_policy(config: Config):
    """Build PolicyEngine from policy.rules in config.

    Always returns an engine -- the hardcoded deny-list works without user rules.
    """
    from koboi.harness.policy import PolicyEngine, PolicyRule, PolicyAction

    engine = PolicyEngine()
    rules_conf = config.get("policy", "rules", default=[])
    for rule_conf in rules_conf:
        tool = rule_conf.get("tool", "*")
        pattern = rule_conf.get("pattern", "")
        action_str = rule_conf.get("action", "allow")
        try:
            action = PolicyAction(action_str)
        except ValueError:
            action = PolicyAction.ALLOW

        engine.add_rule(
            PolicyRule(
                name=f"config_{tool}_{action_str}",
                action=action,
                tool_pattern=tool,
                argument_patterns={"command": pattern} if pattern else None,
                description=f"From config: {tool} {pattern} -> {action_str}",
            )
        )

    return engine


# ---------------------------------------------------------------------------
# Orchestration support
# ---------------------------------------------------------------------------


def _parse_agent_defs(config: Config) -> list:
    """Parse AgentDef list from orchestration.agents config."""
    from koboi.types import AgentDef

    agents_conf = config.orchestration.get("agents", [])
    if not agents_conf:
        raise ValueError("orchestration.agents must have at least one agent")

    defs = []
    for ac in agents_conf:
        name = ac.get("name", "")
        if not name:
            raise ValueError("Each orchestration agent must have a 'name'")
        defs.append(
            AgentDef(
                name=name,
                system_prompt=ac.get("system_prompt", ""),
                description=ac.get("description", ""),
                keywords=ac.get("keywords", []),
                tools_config=ac.get("tools"),
                rag_config=ac.get("rag"),
                llm_config=ac.get("llm"),
            )
        )
    return defs


def _build_router(config: Config, client: RetryClient, agent_defs: list):
    """Build a router from orchestration config."""
    from koboi.orchestration.router import KeywordRouter, LLMRouter, HybridRouter

    router_conf = config.orchestration.get("router", {})
    router_type = router_conf.get("type", "keyword")
    enable_dynamic = router_conf.get("enable_dynamic", False)
    confidence_threshold = router_conf.get("confidence_threshold", 0.5)

    if router_type == "llm":
        return LLMRouter(client=client, enable_dynamic=enable_dynamic, agent_defs=agent_defs)
    elif router_type == "hybrid":
        return HybridRouter(
            client=client,
            confidence_threshold=confidence_threshold,
            enable_dynamic=enable_dynamic,
            agent_defs=agent_defs,
        )
    else:
        return KeywordRouter(agent_defs=agent_defs)


def _build_orchestration(config: Config, verbose: bool = False):
    """Build a KoboiAgent backed by the orchestration engine.

    Reuses AgentAssembler for common subsystems (logger, client, guardrails,
    policy, hooks, skills, mode_manager, trust_db) so orchestration mode
    gets the same policy enforcement and guardrails as single-agent mode.
    """
    from koboi.orchestration.factory import AgentFactory
    from koboi.orchestration.orchestrator import Orchestrator

    assembler = AgentAssembler(config, verbose=verbose)
    assembler.build_logger()
    assembler.build_client()
    assembler.build_guardrails()
    assembler.build_approval()
    assembler.build_policy()
    assembler.build_skills()
    assembler.build_mode_manager()
    assembler.build_trust_db()
    assembler.build_hooks()

    agent_defs = _parse_agent_defs(config)
    router = _build_router(config, assembler.client, agent_defs)

    parent_rag = config.rag
    agents_map = AgentFactory.create_all_configured(
        agent_defs,
        assembler.client,
        assembler.logger,
        parent_rag_config=parent_rag,
        hook_chain=assembler.hook_chain,
    )

    orch_conf = config.orchestration
    exec_conf = orch_conf.get("execution", {})

    orchestrator = Orchestrator(
        client=assembler.client,
        router=router,
        logger=assembler.logger,
        max_revisions=exec_conf.get("max_revisions", 2),
        use_revision=exec_conf.get("use_revision", False),
        enable_dynamic=orch_conf.get("router", {}).get("enable_dynamic", False),
        agents_map=agents_map,
    )

    return KoboiAgent(
        core=None,
        config=config,
        logger=assembler.logger,
        mode_manager=assembler.mode_manager,
        trust_db=assembler.trust_db,
        orchestrator=orchestrator,
    )


def _setup_subagent(
    tools: ToolRegistry,
    client: RetryClient,
    hook_chain: "HookChain",
    logger: AgentLogger,
    memory: "ConversationMemory | None" = None,
    config: Config | None = None,
) -> None:
    """Initialize the subagent system if delegate_tasks tool is registered."""
    if "delegate_tasks" in tools:
        from koboi.subagent import SubAgentManager

        # Read subagent config with defaults
        sub_conf = config.subagent if config else {}
        timeout = sub_conf.get("timeout", 60.0)
        max_iterations = sub_conf.get("max_iterations", 5)

        manager = SubAgentManager(
            client=client,
            tools=tools,
            hook_chain=hook_chain,
            logger=logger,
            max_iterations=max_iterations,
            timeout=timeout,
        )
        if memory is not None:
            manager._parent_memory = memory
        tools.set_dep("manager", manager)


def _setup_tasks(tools: ToolRegistry, config: Config, hook_chain: object | None = None) -> None:
    """Initialize task management if task tools are registered."""
    if "task_create" in tools:
        from koboi.task import TaskManager

        mgr = TaskManager()
        tools.set_dep("manager", mgr)
        # Inject manager into TaskHook if present in the chain
        if hook_chain is not None:
            for hook in getattr(hook_chain, "_hooks", []):
                if type(hook).__name__ == "TaskHook":
                    hook.manager = mgr
                    break


async def _run_orchestrator(orchestrator, message: str) -> RunResult:
    """Run orchestrator and adapt OrchestratorResult to RunResult."""
    import time

    start = time.time()

    result = await orchestrator.run(message)
    elapsed = time.time() - start

    total_tokens = sum(r.tokens_used for r in result.agent_results)

    return RunResult(
        content=result.final_answer,
        iterations_used=len(result.agent_results),
        tool_calls_made=[],
        token_usage=None,
        elapsed_seconds=elapsed,
        metadata={
            "routing_method": result.routing.method,
            "routing_confidence": result.routing.confidence,
            "agents_used": [r.agent_name for r in result.agent_results],
            "execution_mode": result.execution_mode,
            "total_tokens": total_tokens,
        },
    )
