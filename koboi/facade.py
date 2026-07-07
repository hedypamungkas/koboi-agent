"""koboi/facade.py -- KoboiAgent: async facade pattern entry point.

Single class that hides all subsystem complexity. Creates everything from
YAML config and delegates to AgentCore.
"""

from __future__ import annotations

import asyncio
import importlib
import os
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import TYPE_CHECKING
from collections.abc import Awaitable

from koboi.config import Config, extract_extra_params
from koboi.client import Client, RetryClient
from koboi.llm.pool import CircuitBreaker, FailoverPolicy, ProviderPool
from koboi.llm.resolve import resolve_llm_spec
from koboi.memory import ConversationMemory
from koboi.modes import AgentMode, ModeManager
from koboi.tools.registry import ToolRegistry, register_decorated
from koboi.logger import AgentLogger
from koboi.types import RunResult, RiskLevel
from koboi.hooks.chain import HookEvent

if TYPE_CHECKING:
    import threading

    from koboi.context.manager import ContextManager
    from koboi.events import StreamEvent
    from koboi.guardrails.approval import ApprovalHandler
    from koboi.guardrails.audit import AuditTrail
    from koboi.guardrails.base import BaseGuardrail
    from koboi.guardrails.rate_limiter import RateLimiter
    from koboi.harness.policy import PolicyEngine
    from koboi.hooks.chain import HookChain
    from koboi.journal import StepJournal
    from koboi.loop import AgentCore
    from koboi.orchestration.orchestrator import Orchestrator
    from koboi.rag.augmentation import AugmentationStrategy
    from koboi.sandbox.base import BaseSandbox
    from koboi.skills.registry import SkillRegistry
    from koboi.trust import TrustDatabase


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
        trust_db: TrustDatabase | None = None,
        orchestrator: Orchestrator | None = None,
    ):
        self._core = core
        self._config = config
        self._logger = logger
        self._mcp_clients = mcp_clients or []
        self._sync_loop: asyncio.AbstractEventLoop | None = None
        self._bg_loop: asyncio.AbstractEventLoop | None = None
        self._bg_thread: threading.Thread | None = None
        self._mode_manager = mode_manager
        self._trust_db = trust_db
        self._orchestrator = orchestrator

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        verbose: bool = False,
        resume_session: str | None = None,
    ) -> KoboiAgent:
        """Factory method: create a KoboiAgent from YAML config.

        Pass ``resume_session`` to rehydrate-and-continue an interrupted session
        (P2-A): the SQLite memory reloads that session's conversation and the
        journal inherits its turn numbering. Call ``agent.resume()`` to actually
        resume the loop.
        """
        config = Config.from_yaml(config_path)
        return cls._from_config(config, verbose=verbose, resume_session=resume_session)

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
    def _from_config(
        cls,
        config: Config,
        verbose: bool = False,
        resume_session: str | None = None,
    ) -> KoboiAgent:
        """Shared builder: assemble all subsystems from a Config object."""
        if resume_session:
            # Point the SQLite memory at the target session so it rehydrates that
            # conversation (and the journal inherits its turn numbering).
            config._data.setdefault("memory", {})["session_id"] = resume_session
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
            from koboi.loop import _extract_text

            query = message if isinstance(message, str) else _extract_text(message)
            async for event in self._orchestrator.run_stream(
                query, mode=getattr(self._orchestrator, "default_mode", "sequential")
            ):
                yield event
        else:
            async for event in self._core.run_stream(message):
                yield event

    async def chat(self, message: str | list) -> RunResult:
        if self._orchestrator is not None:
            return await _run_orchestrator(self._orchestrator, message)
        return await self._core.chat(message)

    async def resume(self) -> RunResult:
        """Rehydrate-and-continue an interrupted session (P2-A).

        Construct the agent with ``from_config(..., resume_session=<id>)`` (or
        the ``--resume`` CLI flag) so memory rehydrates that session, then call
        this to resume the loop without re-asking the user message. Not supported
        in orchestration mode (v1).
        """
        from koboi.exceptions import AgentError

        if self._orchestrator is not None:
            raise AgentError("Resume is not supported in orchestration mode (v1)")
        if self._core is None:
            raise AgentError("No core agent to resume")
        return await self._core.resume()

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
            except Exception:  # nosec B110 - best-effort; intentionally swallows transient errors (cleanup/export/teardown)
                pass
        if self._orchestrator is not None:
            shared_client = self._orchestrator.client
            # Clean up orchestrator's sub-agent memories + their dedicated LLM
            # clients. Per-agent llm_config overrides build a separate RetryClient
            # per agent; those connection pools must be closed, not just the shared one.
            for agent in getattr(self._orchestrator, "_agents_map", {}).values():
                if hasattr(agent, "memory") and hasattr(agent.memory, "close"):
                    agent.memory.close()
                agent_client = getattr(agent, "client", None)
                if agent_client is not None and agent_client is not shared_client and hasattr(agent_client, "close"):
                    await agent_client.close()
            await shared_client.close()
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
            except Exception:  # nosec B110 - best-effort; intentionally swallows transient errors (cleanup/export/teardown)
                pass
        if self._logger is not None:
            try:
                self._logger.close()
            except Exception:  # nosec B110 - best-effort; intentionally swallows transient errors (cleanup/export/teardown)
                pass
        bg_loop = getattr(self, "_bg_loop", None)
        if bg_loop is not None:
            try:
                bg_loop.call_soon_threadsafe(bg_loop.stop)
            except Exception:  # nosec B110 - best-effort; intentionally swallows transient errors (cleanup/export/teardown)
                pass
            bg_thread = getattr(self, "_bg_thread", None)
            if bg_thread is not None:
                try:
                    bg_thread.join(timeout=1.0)
                except Exception:  # nosec B110 - best-effort; intentionally swallows transient errors (cleanup/export/teardown)
                    pass
            try:
                bg_loop.close()
            except Exception:  # nosec B110 - best-effort; intentionally swallows transient errors (cleanup/export/teardown)
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
                    raise ValueError(f"Unknown event '{e}'. Valid events: {valid}") from None
        from koboi.hooks.callback_hook import CallbackHook

        if self._core is not None:
            self._core.hooks.add(
                CallbackHook(callback=callback, events=events)  # type: ignore[arg-type]  # on()/add_hook() accept sync or async callbacks
            )
        return self

    def add_hook(
        self,
        callback: Callable | Awaitable,
        events: list[HookEvent] | None = None,
    ) -> None:
        """Register a callback as a hook without subclassing Hook."""
        from koboi.hooks.callback_hook import CallbackHook

        if self._core is not None:
            self._core.hooks.add(
                CallbackHook(callback=callback, events=events)  # type: ignore[arg-type]  # on()/add_hook() accept sync or async callbacks
            )

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
        return found.telemetry if found else None  # type: ignore[attr-defined]  # dynamic attr, guarded by the hasattr lambda above

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
        client = hook.get_client()  # type: ignore[attr-defined]  # dynamic attr on LangfuseTracingHook, guarded by hasattr above
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
    def orchestrator(self) -> Orchestrator | None:
        return self._orchestrator

    @property
    def mode_manager(self) -> ModeManager | None:
        return self._mode_manager

    @property
    def trust_db(self) -> TrustDatabase | None:
        return self._trust_db


def _build_client_from_dict(llm: dict, logger: AgentLogger) -> RetryClient:
    """Build a ``RetryClient`` from a resolved inline llm dict.

    Shared by the top-level client build (Tier 0/1) and the per-agent full-replace
    path (a named ``providers:`` ref). All generation knobs -- temperature,
    max_tokens, and the forward-as-is extra params -- are read here.
    """
    return RetryClient(
        provider=llm.get("provider", "openai"),
        model=llm.get("model", "gpt-4o-mini"),
        api_key=llm.get("api_key", ""),
        base_url=llm.get("base_url", ""),
        logger=logger,
        timeout=llm.get("timeout", 120.0),
        max_tokens=llm.get("max_tokens"),
        auth_token=llm.get("auth_token", ""),
        auth_type=llm.get("auth_type", "api_key"),
        max_retries=llm.get("max_retries", 3),
        retry_backoff_base=llm.get("retry_backoff_base", 2.0),
        temperature=llm.get("temperature"),
        extra_params=extract_extra_params(llm),
    )


def _build_client(config: Config, logger: AgentLogger, llm_overrides: dict | None = None) -> Client:
    # Resolve the top-level ``llm:`` spec (inline / named ``providers:`` ref /
    # ``{pool: name}``) and merge per-agent overrides over it. If the override
    # switches provider, don't inherit the parent's connection credentials -- an
    # OpenAI key must not be sent to Anthropic (opaque 401). Blank them so
    # ProviderRegistry resolves the new provider's env (or raises a clear "key
    # not configured").
    base = resolve_llm_spec(config.llm, config) or {}
    if llm_overrides:
        llm = {**base, **llm_overrides}
        if llm.get("provider", "openai") != base.get("provider", "openai"):
            for _conn_key in ("api_key", "auth_token", "base_url"):
                if _conn_key not in llm_overrides:
                    llm[_conn_key] = ""
    else:
        llm = base
    return _build_client_from_dict(llm, logger)


def _build_pool_from_spec(
    pool_name: str,
    config: Config,
    logger: AgentLogger | None,
    member_builder=None,
) -> ProviderPool:
    """Build a ``ProviderPool`` from a named ``pools:`` entry (Tier 2).

    ``member_builder(inline_dict, logger)`` constructs each member; it defaults
    to ``_build_client_from_dict`` (chat). Embeddings pass a dedicated builder.
    Each member ref resolves through ``providers:`` (inline or named). The
    ``failover`` policy + circuit breaker are wired from the pool spec.
    """
    pools = config.pools
    if pool_name not in pools:
        raise ValueError(
            f"Unknown pool reference {pool_name!r}. Define it under `pools:`. Available: {sorted(pools) or '(none)'}"
        )
    spec = pools[pool_name] or {}
    refs = spec.get("providers") or []
    if not refs:
        raise ValueError(f"Pool {pool_name!r} has no `providers:` members")
    build = member_builder or _build_client_from_dict
    members: list[Client] = []
    for ref in refs:
        inline = resolve_llm_spec(ref, config)
        if not inline:
            raise ValueError(f"Pool {pool_name!r} member {ref!r} did not resolve to a provider spec")
        members.append(build(inline, logger))
    policy_name = (spec.get("policy") or "failover").lower()
    if policy_name != "failover":
        raise NotImplementedError(f"Pool policy {policy_name!r} not implemented (W2 ships failover).")
    cb_cfg = spec.get("circuit_breaker") or {}
    breaker = CircuitBreaker(
        failure_threshold=cb_cfg.get("failures", cb_cfg.get("failure_threshold", 3)),
        cooldown_s=cb_cfg.get("cooldown_s", 30.0),
    )
    return ProviderPool(members, FailoverPolicy(), breaker)


def _resolve_chat_client(config: Config, logger: AgentLogger | None) -> Client:
    """Top-level chat client: inline dict / named ref (Tier 0/1) or pool (Tier 2)."""
    spec = config.llm
    if isinstance(spec, dict) and "pool" in spec:
        return _build_pool_from_spec(spec["pool"], config, logger)
    return _build_client(config, logger)


def _build_tools(config: Config) -> ToolRegistry:
    registry = ToolRegistry()
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

    # Apply defaults/overrides (with alias normalization), disabled denylist,
    # and groups -- via the shared helper so this stays in sync with the
    # orchestration factory's _build_tools_from_config. The helper also mirrors
    # defaults onto the env-hygiene module config (see apply_tool_selection).
    from koboi.tools.registry import apply_tool_selection

    apply_tool_selection(registry, config.get("tools", default={}))
    return registry


def _build_context(config: Config, logger: AgentLogger, client: Client | None = None):
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


def _embedding_member_from_dict(inline: dict, logger: AgentLogger | None) -> Client:
    """Build one embedding-pool member from a resolved inline provider dict.

    Uses ``create_client`` (bare ``LLMClient``, like ``build_embedding_client``)
    so ``embedding_model`` is honored; pool members must carry ``api_key``.
    """
    from koboi.llm.factory import create_client

    if not inline.get("api_key"):
        raise ValueError("Embedding pool member requires api_key")
    model = inline.get("model") or "text-embedding-3-small"
    return create_client(
        provider=inline.get("provider", "openai"),
        model=model,
        api_key=inline.get("api_key", ""),
        base_url=inline.get("base_url", ""),
        embedding_model=model,
        logger=logger,
    )


def _build_embedding_client(config: Config, logger: AgentLogger):
    """Build a dedicated embedding client from an optional ``embedding:`` config
    section (decoupled from chat). Delegates to the shared
    ``koboi.llm.factory.build_embedding_client``; returns ``None`` when no usable
    section is configured so callers fall back to the chat client.

    The ``embedding:`` spec may be an inline dict (today), a named ``providers:``
    ref (Tier 1), or a ``{pool: name}`` (Tier 2 -- an embedding pool for RAG
    resilience). Inline/named normalize to a dict; pool builds a ProviderPool.
    """
    from koboi.llm.factory import build_embedding_client

    emb_spec = config.get("embedding")
    if isinstance(emb_spec, dict) and "pool" in emb_spec:
        return _build_pool_from_spec(emb_spec["pool"], config, logger, member_builder=_embedding_member_from_dict)
    emb = resolve_llm_spec(emb_spec, config)
    return build_embedding_client(emb, logger)


def _build_rag(config: Config, client: Client, logger: AgentLogger):
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

    # Use a dedicated embedding provider when configured (decoupled from chat);
    # else fall back to the chat client. Only the SemanticRetriever consumes it.
    rag_client = _build_embedding_client(config, logger) or client
    return build_rag(rag_dict, client=rag_client, logger=logger)


def _normalize_guardrail_config(conf: dict | list | None, default_name: str = "injection_detector") -> list[dict]:
    """Normalize guardrail config to list-of-dicts format.

    Supports legacy single-dict format (auto-wrapped) and new list format.
    Empty/None returns empty list. ``default_name`` selects the fallback
    guardrail for a bare config block (no ``name`` key) -- the *input* slot
    defaults to ``injection_detector``, but the *output* slot must default to
    ``content_filter`` so e.g. ``{detect_sensitive: true}`` builds an
    ``OutputGuardrail`` (which tolerates empty output) rather than an
    ``InputGuardrail`` whose "Input is empty" check clobbers tool-call turns.
    """
    if not conf:
        return []
    if isinstance(conf, dict):
        # Legacy: single guardrail as dict -> wrap in list, inferring name
        if "name" in conf:
            return [conf]
        # Legacy: config block like {max_length: 100} -> wrap with default name
        return [{"name": default_name, **conf}]
    if isinstance(conf, list):
        return [c for c in conf if isinstance(c, dict) and c.get("name")]
    return []


def _build_guardrails(config: Config, logger: AgentLogger | None = None):
    from koboi.guardrails.registry import GuardrailRegistry

    input_grds: list = []
    output_grds: list = []
    rate_limiter: RateLimiter | None = None
    audit_trail: AuditTrail | None = None

    # Input guardrails -- supports both legacy and new config formats
    input_conf = config.get("guardrails", "input", default={})
    input_configs = _normalize_guardrail_config(input_conf)
    if input_configs:
        input_grds = GuardrailRegistry.from_config(input_configs)
    elif input_conf:
        # Legacy: bare dict without "name" key -> default injection_detector
        input_grds = GuardrailRegistry.from_config([{"name": "injection_detector", **input_conf}])

    # Output guardrails -- a bare block (no "name") defaults to content_filter,
    # NOT injection_detector, so output is checked by OutputGuardrail (which
    # passes on empty output) rather than InputGuardrail (which blocks it).
    output_conf = config.get("guardrails", "output", default={})
    output_configs = _normalize_guardrail_config(output_conf, default_name="content_filter")
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


def _build_approval(config: Config, trust_db=None):
    handler_conf = config.get("guardrails", "approval", default={})
    handler_type = handler_conf.get("handler", "auto")
    if handler_type == "cli":
        from koboi.guardrails.approval import CLIApprovalHandler

        return CLIApprovalHandler()
    elif handler_type == "callback":
        from koboi.guardrails.approval import CallbackApprovalHandler

        return CallbackApprovalHandler(handler_conf.get("callback", lambda *a: True))
    elif handler_type == "async_callback":
        # REST/SSE-friendly non-blocking handler. ``callback``/``audit_trail`` are
        # caller-injected (non-serializable) -- the M2 server bootstrap populates
        # them programmatically; absent a callback we return None (no handler).
        from koboi.guardrails.approval import AsyncCallbackApprovalHandler

        callback = handler_conf.get("callback")
        if callback is None:
            return None
        timeout = handler_conf.get("timeout", 120)
        audit_trail = handler_conf.get("audit_trail")
        return AsyncCallbackApprovalHandler(
            callback=callback, trust_db=trust_db, audit_trail=audit_trail, timeout=timeout
        )
    return None


def _build_skills(config: Config, logger: AgentLogger):
    search_paths = config.get("skills", "search_paths", default=[])
    if not search_paths:
        return None
    from koboi.skills.registry import SkillRegistry

    budget_chars = config.get("skills", "budget_chars", default=8000)
    registry = SkillRegistry(budget_chars=budget_chars)
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
        self.client: Client | None = None
        self.memory: ConversationMemory | None = None
        self.tools: ToolRegistry | None = None
        self.mcp_clients: list | None = None
        self.context_manager: ContextManager | None = None
        self.augmentation: AugmentationStrategy | None = None
        self.input_guardrails: list[BaseGuardrail] = []
        self.output_guardrails: list[BaseGuardrail] = []
        self.rate_limiter: RateLimiter | None = None
        self.audit_trail: AuditTrail | None = None
        self.approval_handler: ApprovalHandler | None = None
        self.policy_engine: PolicyEngine | None = None
        self.skills: SkillRegistry | None = None
        self.mode_manager: ModeManager | None = None
        self.trust_db: TrustDatabase | None = None
        self.hook_chain: HookChain | None = None
        self.sandbox: BaseSandbox | None = None
        self.journal: StepJournal | None = None

    def build_logger(self) -> AgentLogger:
        self.logger = AgentLogger(session_id=self.config.agent_name)
        return self.logger

    def build_client(self) -> Client:
        self.client = _resolve_chat_client(self.config, self.logger)
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
            # Record the session row so `koboi sessions` lists it (and resume can
            # target it). Upsert -- safe on resume (re-hydrated session_id).
            self.memory.ensure_session_record(
                agent_name=self.config.agent_name,
                model=self.config.model,
            )
        else:
            self.memory = ConversationMemory(
                logger=self.logger,
                system_prompt=self.config.system_prompt or None,
            )
        return self.memory

    def build_journal(self) -> object:
        """Build the step journal (P2-A) for SQLite-backed memory.

        Returns None when journaling is disabled or the memory backend isn't
        SQLite (the journal borrows the SQLite connection). Built right after
        memory so it can read the session_id that memory rehydrated.
        """
        journal_conf = self.config.get("journal", default={})
        if not journal_conf.get("enabled", True):
            self.journal = None
            return None
        from koboi.memory_sqlite import SQLiteMemory

        # The journal borrows the SQLite connection; only SQLiteMemory qualifies.
        if not isinstance(self.memory, SQLiteMemory):
            self.journal = None
            return None
        from koboi.journal import StepJournal

        record_tool_calls = journal_conf.get("record_tool_calls", True)
        self.journal = StepJournal(
            self.memory._ensure_conn(),
            self.memory.session_id,
            record_tool_calls=record_tool_calls,
        )
        return self.journal

    def build_tools(self) -> ToolRegistry:
        self.tools = _build_tools(self.config)
        return self.tools

    def build_sandbox(self) -> object:
        """Build the sandbox and inject it into the tool registry.

        Always built (even for ``passthrough``) so tools can rely on
        ``_deps["sandbox"]`` being non-None; the passthrough backend reproduces
        pre-P0b behavior exactly.
        """
        self.sandbox = _build_sandbox(self.config, self.logger)
        if self.tools is not None:
            self.tools.set_dep("sandbox", self.sandbox)
            # M6: per-session tool state (read-before-write tracking) so concurrent
            # agents don't share the module-global _read_paths.
            from koboi.tools.state import ToolState

            self.tools.set_dep("tool_state", ToolState())
        return self.sandbox

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
        self.approval_handler = _build_approval(self.config, trust_db=self.trust_db)
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
        self.build_journal()
        self.build_tools()
        self.build_sandbox()
        self.build_mcp()
        self.build_context()
        self.build_rag()
        self.build_guardrails()
        self.build_trust_db()
        self.build_approval()
        self.build_policy()
        self.build_skills()
        self.build_mode_manager()
        self.build_hooks()

        _setup_subagent(self.tools, self.client, self.hook_chain, self.logger, memory=self.memory, config=self.config)
        _setup_tasks(self.tools, self.config, hook_chain=self.hook_chain)

        # Add skill persistence hook if skills are present
        if self.skills and self.hook_chain:
            from koboi.hooks.skill_persistence_hook import SkillPersistenceHook

            self.hook_chain.add(SkillPersistenceHook(skills=self.skills))

        # P3b: compaction-aware tool-state preservation hooks.
        # TaskPersistenceHook re-injects the active todo list after a compact;
        # ReadBeforeWriteResetHook clears stale read-tracking on session start
        # and real compaction. Each is added only when its collaborator exists.
        if self.hook_chain:
            task_mgr = self.tools.get_dep("task_manager")
            if task_mgr is not None:
                from koboi.hooks.task_persistence_hook import TaskPersistenceHook

                self.hook_chain.add(TaskPersistenceHook(manager=task_mgr))
            if "read_file" in self.tools:
                from koboi.hooks.read_before_write_reset_hook import ReadBeforeWriteResetHook

                self.hook_chain.add(ReadBeforeWriteResetHook())

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
            journal=self.journal,
            trust_db=self.trust_db,
            output_schema=self.config.get("agent", "output_schema", default=None),
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

    from koboi.mcp.base import default_risk_heuristic, register_mcp_tools
    from koboi.types import RiskLevel

    clients = []
    for server_conf in servers:
        transport = server_conf.get("transport", "stdio")
        try:
            mcp_client = _create_mcp_client(server_conf, transport, logger, config)
            mcp_client.connect()
            group = server_conf.get("group")
            # #5: MCP risk gating. Default SAFE (pre-#5); risk_level overrides for all
            # tools from this server, risk_heuristic infers per-tool risk from the name.
            try:
                risk_level = RiskLevel(server_conf.get("risk_level", "safe"))
            except ValueError:
                risk_level = RiskLevel.SAFE
            resolver = default_risk_heuristic if server_conf.get("risk_heuristic", False) else None
            register_mcp_tools(mcp_client, tools, group=group, risk_level=risk_level, risk_resolver=resolver)
            clients.append(mcp_client)
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(
                "MCP server connection failed for '%s': %s",
                server_conf.get("url") or server_conf.get("command", "?"),
                e,
            )

    return clients


# M4: allowed MCP stdio runners (basename match). Extend via mcp.allowlist_commands.
_MCP_DEFAULT_RUNNERS = frozenset({"npx", "uvx", "python", "python3", "node", "uv", "deno", "bun"})


def _create_mcp_client(server_conf: dict, transport: str, logger: AgentLogger, config: Config | None = None):
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
        # M4: stdio command allow-list (basename match) -- blocks arbitrary-binary
        # execution from a malicious/misconfigured YAML. Extend via mcp.allowlist_commands.
        runner = os.path.basename(command)
        extra = set(config.get("mcp", "allowlist_commands", default=[])) if config is not None else set()
        if runner not in (_MCP_DEFAULT_RUNNERS | extra):
            raise ValueError(
                f"MCP stdio command {runner!r} not in allow-list. Permit it via "
                f"mcp.allowlist_commands. Default runners: {sorted(_MCP_DEFAULT_RUNNERS)}"
            )
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

        # #4: argument_patterns generalizes the legacy ``pattern`` shorthand (which
        # only matched an arg literally named "command"). Prefer an explicit
        # argument_patterns dict; fall back to {"command": pattern} for back-compat
        # so existing run_shell ``pattern:`` configs keep working unchanged.
        arg_patterns = rule_conf.get("argument_patterns") or ({"command": pattern} if pattern else None)

        engine.add_rule(
            PolicyRule(
                name=f"config_{tool}_{action_str}",
                action=action,
                tool_pattern=tool,
                argument_patterns=arg_patterns,
                description=f"From config: {tool} {pattern} -> {action_str}",
            )
        )

    return engine


def _build_sandbox(config: Config, logger: AgentLogger):
    """Build the sandbox backend from the ``sandbox:`` config section.

    Always returns a handle (defaults to ``PassthroughBackend``) so subprocess
    tools can rely on ``_deps["sandbox"]`` without None-checks. Never raises on
    bad config -- ``build_sandbox`` falls back to passthrough and warns.
    """
    from koboi.sandbox import build_sandbox

    return build_sandbox(config.sandbox, logger=logger)


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
                depends_on=ac.get("depends_on", []),
            )
        )
    return defs


def _build_router(config: Config, client: Client, agent_defs: list):
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
    assembler.build_trust_db()
    assembler.build_approval()
    assembler.build_policy()
    assembler.build_skills()
    assembler.build_mode_manager()
    assembler.build_hooks()
    assembler.build_sandbox()

    agent_defs = _parse_agent_defs(config)
    router = _build_router(config, assembler.client, agent_defs)

    parent_rag = config.rag

    # Per-agent LLM client builder. A named ``providers:`` ref (str) FULLY
    # REPLACES the top-level client; an inline dict MERGES over it (today's
    # behavior) so temperature/max_tokens/extra params take effect per agent.
    # Pool specs (W2) raise. Agents without LLM overrides keep sharing
    # assembler.client (decided inside AgentFactory via _has_client_overrides).
    def _agent_client_builder(agent_llm: dict | str) -> Client:
        if isinstance(agent_llm, str):
            return _build_client_from_dict(resolve_llm_spec(agent_llm, config), assembler.logger)
        if isinstance(agent_llm, dict) and "pool" in agent_llm:
            return _build_pool_from_spec(agent_llm["pool"], config, assembler.logger)
        overrides = {k: v for k, v in agent_llm.items() if k != "max_context_tokens"}
        return _build_client(config, assembler.logger, llm_overrides=overrides)

    agents_map = AgentFactory.create_all_configured(
        agent_defs,
        assembler.client,
        assembler.logger,
        parent_rag_config=parent_rag,
        hook_chain=assembler.hook_chain,
        sandbox=assembler.sandbox,
        embedding_config=config.get("embedding"),
        client_builder=_agent_client_builder,
    )

    orch_conf = config.orchestration
    exec_conf = orch_conf.get("execution", {})
    exec_mode = exec_conf.get("mode", "sequential")

    # Build a DagScheduler when execution.mode == "dag", seeded with the per-agent
    # depends_on edges parsed from config (deterministic, testable).
    dag_scheduler = None
    if exec_mode == "dag":
        from koboi.orchestration.dag_scheduler import DagScheduler

        deps = {ad.name: list(ad.depends_on) for ad in agent_defs}
        # Persist the graph plan when the memory backend is SQLite (durable; #3).
        dag_db_path = None
        if config.get("memory", "backend", default="sqlite") == "sqlite":
            dag_db_path = config.get("memory", "db_path", default="koboi_memory.db")
        dag_scheduler = DagScheduler(agents_map=agents_map, deps=deps, db_path=dag_db_path)

    orchestrator = Orchestrator(
        client=assembler.client,
        router=router,
        logger=assembler.logger,
        max_revisions=exec_conf.get("max_revisions", 2),
        use_revision=exec_conf.get("use_revision", False),
        enable_dynamic=orch_conf.get("router", {}).get("enable_dynamic", False),
        agents_map=agents_map,
        dag_scheduler=dag_scheduler,
        default_mode=exec_mode,
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
    client: Client,
    hook_chain: HookChain,
    logger: AgentLogger,
    memory: ConversationMemory | None = None,
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
            manager._parent_memory = memory  # type: ignore[attr-defined]  # injected attr consumed by SubAgentManager._run_single
        tools.set_dep("subagent_manager", manager)


def _setup_tasks(tools: ToolRegistry, config: Config, hook_chain: object | None = None) -> None:
    """Initialize task management if task tools are registered."""
    if "task_create" in tools:
        from koboi.task import TaskManager

        # #6: persist task state to SQLite when backend=sqlite, so it survives --resume.
        # When a session_id is set (resume), TaskManager rehydrates existing tasks.
        db_path = None
        session_id = None
        if config.get("memory", "backend", default="sqlite") == "sqlite":
            db_path = config.get("memory", "db_path", default="koboi_memory.db")
            session_id = config.get("memory", "session_id", default=None) or None
        mgr = TaskManager(db_path=db_path, session_id=session_id)
        tools.set_dep("task_manager", mgr)
        # Inject manager into TaskHook if present in the chain
        if hook_chain is not None:
            for hook in getattr(hook_chain, "_hooks", []):
                if type(hook).__name__ == "TaskHook":
                    hook.manager = mgr
                    break


async def _run_orchestrator(orchestrator, message: str | list) -> RunResult:
    """Run orchestrator and adapt OrchestratorResult to RunResult.

    The orchestrator takes a ``str`` query; multimodal (list) input is reduced
    to its text portion -- orchestration mode has no multimodal support.
    """
    import time
    from koboi.loop import _extract_text

    query = message if isinstance(message, str) else _extract_text(message)
    start = time.time()

    result = await orchestrator.run(query, mode=getattr(orchestrator, "default_mode", "sequential"))
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
