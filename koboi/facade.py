"""koboi/facade.py -- KoboiAgent: async facade pattern entry point.

Single class that hides all subsystem complexity. Creates everything from
YAML config and delegates to AgentCore.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
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
    from koboi.media.types import MediaRequest, MediaResult
    from koboi.mcp.base import BaseMCPClient
    from koboi.orchestration.orchestrator import Orchestrator
    from koboi.rag.augmentation import AugmentationStrategy
    from koboi.sandbox.base import BaseSandbox
    from koboi.server.peers import PeerRegistry
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
        replay_mode: str | None = None,
        cache_dir: str | None = None,
    ) -> KoboiAgent:
        """Factory method: create a KoboiAgent from YAML config.

        Pass ``resume_session`` to rehydrate-and-continue an interrupted session
        (P2-A): the SQLite memory reloads that session's conversation and the
        journal inherits its turn numbering. Call ``agent.resume()`` to actually
        resume the loop. ``replay_mode`` / ``cache_dir`` inject a per-run cache
        mode (see ``koboi.llm.cache``) without mutating the source config.
        """
        config = Config.from_yaml(config_path)
        return cls._from_config(
            config,
            verbose=verbose,
            resume_session=resume_session,
            replay_mode=replay_mode,
            cache_dir=cache_dir,
        )

    @classmethod
    def from_dict(
        cls,
        data: dict,
        verbose: bool = False,
        replay_mode: str | None = None,
        cache_dir: str | None = None,
        peer_registry: PeerRegistry | None = None,
    ) -> KoboiAgent:
        """Factory method: create a KoboiAgent from a Python dict.

        Usage:
            agent = KoboiAgent.from_dict({
                "agent": {"name": "my-agent", "system_prompt": "You are helpful"},
                "llm": {"provider": "openai", "model": "gpt-4o"},
            })
        """
        config = Config.from_dict(data)
        return cls._from_config(
            config, verbose=verbose, replay_mode=replay_mode, cache_dir=cache_dir, peer_registry=peer_registry
        )

    @classmethod
    def from_config_string(
        cls, yaml_string: str, verbose: bool = False, replay_mode: str | None = None, cache_dir: str | None = None
    ) -> KoboiAgent:
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
        return cls._from_config(config, verbose=verbose, replay_mode=replay_mode, cache_dir=cache_dir)

    @classmethod
    def _from_config(
        cls,
        config: Config,
        verbose: bool = False,
        resume_session: str | None = None,
        replay_mode: str | None = None,
        cache_dir: str | None = None,
        peer_registry: PeerRegistry | None = None,
    ) -> KoboiAgent:
        """Shared builder: assemble all subsystems from a Config object."""
        if resume_session:
            # Point the SQLite memory at the target session so it rehydrates that
            # conversation (and the journal inherits its turn numbering).
            config._data.setdefault("memory", {})["session_id"] = resume_session
        if replay_mode is not None or cache_dir is not None:
            # Inject a per-run cache mode on an immutable copy (the shared pooled
            # server Config must not be mutated -- AgentCore is not concurrent-safe).
            config = config.with_replay(replay_mode=replay_mode, cache_dir=cache_dir)
        # Orchestration mode: transparent to caller
        if config.orchestration.get("enabled"):
            return _build_orchestration(config, verbose=verbose, peer_registry=peer_registry)

        assembler = AgentAssembler(config, verbose=verbose)
        return assembler.build(peer_registry=peer_registry)

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
            # W5.1: deep_research can resume (rehydrate-and-finish from the journal).
            if getattr(self._orchestrator, "default_mode", None) == "deep_research":
                db_path = self._orchestrator._dag_scheduler.db_path if self._orchestrator._dag_scheduler else None
                if not db_path:
                    raise AgentError("Cannot resume deep_research: no db_path (memory.backend must be sqlite)")
                from koboi.orchestration.dag_scheduler import DagScheduler

                # Session-scoped first (avoid a cross-session leak: a shared koboi_memory.db
                # holds every session's research_context rows; the global latest would return
                # whichever session ran most recently). Fall back to the global latest only for
                # non-server callers whose rows carry no session_id tag.
                resume_session_id = getattr(self._orchestrator, "_session_id", None)
                ctx_json = (
                    DagScheduler.load_research_context_for_session(db_path, resume_session_id)
                    if resume_session_id
                    else DagScheduler.load_latest_research_context(db_path)
                )
                if not ctx_json:
                    raise AgentError("No research context found to resume (run deep_research first)")
                self._orchestrator._resume_ctx_json = ctx_json  # type: ignore[attr-defined]
                return await _run_orchestrator(self._orchestrator, "")
            # Issue #10b: clearer message. Orchestration mode runs N per-agent
            # memories with no single shared conversation/journal to resume; full
            # resume support requires orchestration-mode redesign (deferred).
            raise AgentError(
                "Resume is not supported in orchestration mode (v1): the "
                "orchestrator runs multiple per-agent memories with no single "
                "shared conversation or step-journal to rehydrate. To resume a "
                "specific agent, run its config directly with `koboi run "
                "<single-agent-config> --resume <session-id>`."
            )
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
            except Exception as e:  # noqa: BLE001
                logging.getLogger(__name__).warning("MCP client close failed: %s", e)  # 24-G
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
            # Close the augmentation's retriever chain if it owns resources (e.g. the
            # cross-encoder rerank HTTP transport). Duck-typed: only transport-bearing
            # retrievers (CrossEncoderReranker) override close().
            aug = getattr(self._core, "augmentation", None)
            retriever = getattr(aug, "retriever", None) if aug is not None else None
            if hasattr(retriever, "close"):
                try:
                    await retriever.close()
                except Exception as e:  # nosec B110 - best-effort teardown; logged for diagnostics
                    logging.getLogger(__name__).debug("Augmentation retriever close failed: %s", e, exc_info=True)
            # Close the media backend (gateway HTTP transport) if media is enabled.
            _media_tools = getattr(self._core, "tools", None)
            _media_backend = _media_tools.get_dep("media_provider") if _media_tools is not None else None
            if _media_backend is not None and hasattr(_media_backend, "close"):
                try:
                    await _media_backend.close()
                except Exception as e:  # nosec B110 - best-effort teardown
                    logging.getLogger(__name__).debug("Media backend close failed: %s", e, exc_info=True)
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
            except Exception as e:  # noqa: BLE001
                logging.getLogger(__name__).warning("MCP client close failed: %s", e)  # 24-G
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
        idempotent: bool = True,
    ) -> None:
        """Register a tool on the agent.

        ``idempotent=False`` marks a side-effecting tool that must not silently
        double-fire on crash-resume (issue #8b); the resume path skips its
        re-execution and records a synthetic result.
        """
        if self._core is not None:
            self._core.tools.register(name, description, parameters, fn, risk_level=risk_level, idempotent=idempotent)

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

    @property
    def mcp_clients(self) -> list:
        """The connected MCP clients (read-only view; used by the TUI/server layers)."""
        return list(self._mcp_clients)

    async def media_generate(self, req: MediaRequest) -> MediaResult:
        """Programmatic media generation (W5a)."""
        backend = self._media_backend()
        if backend is None:
            from koboi.media.backend import _not_configured

            return _not_configured(req, (req.modality or "image"))
        return await backend.generate(req)

    async def media_transcribe(self, audio: bytes, **opts) -> str:
        """Programmatic STT (W5a)."""
        backend = self._media_backend()
        if backend is None:
            raise RuntimeError("media not configured (enable media.transcription)")
        return await backend.transcribe(audio, **opts)

    def _media_backend(self):
        """Reach the active MediaBackend."""
        if self._orchestrator is not None:
            return getattr(self._orchestrator, "_media_backend", None)
        if self._core is not None:
            _tools = getattr(self._core, "tools", None)
            return _tools.get_dep("media_provider") if _tools is not None else None
        return None

    def mcp_status(self) -> list[dict]:
        """Per-MCP-server status for the TUI and ``/v1/.../mcp/servers`` (G6/G7).

        Returns one entry per live client (with ``connected`` from the transport's
        ``is_connected()``) PLUS one entry per configured server that failed to connect
        (``connected: False``, no client) so dead servers are visible. Each entry::

            {id, name, transport, connected, server_info, tool_names, configured}
        """
        live_endpoints: set[str] = set()
        entries: list[dict] = []
        for client in self._mcp_clients:
            endpoint = client.endpoint
            live_endpoints.add(endpoint)
            entries.append(
                {
                    "id": client.name or endpoint or client.transport,
                    "name": client.name or endpoint,
                    "transport": client.transport,
                    "connected": client.is_connected(),
                    "server_info": client.server_info,
                    "tool_names": list(client.tool_names),
                    "configured": True,
                }
            )
        # Configured-but-failed servers (connect raised -> not in _mcp_clients).
        for sc in self._config.get("mcp", "servers", default=[]) if self._config else []:
            endpoint = self._mcp_conf_endpoint(sc)
            if endpoint in live_endpoints:
                continue
            entries.append(
                {
                    "id": sc.get("group") or endpoint or sc.get("transport", "stdio"),
                    "name": sc.get("group") or endpoint or sc.get("command") or sc.get("url", ""),
                    "transport": sc.get("transport", "stdio"),
                    "connected": False,
                    "server_info": {},
                    "tool_names": [],
                    "configured": True,
                }
            )
        return entries

    @staticmethod
    def _mcp_conf_endpoint(server_conf: dict) -> str:
        """Identity endpoint string for a configured server (matches client.endpoint)."""
        if server_conf.get("transport") == "streamable-http":
            return server_conf.get("url", "")
        cmd = server_conf.get("command", "")
        args = server_conf.get("args", []) or []
        return " ".join([cmd, *args]).strip()

    def add_mcp_client(self, client, group: str | None = None, risk_level: RiskLevel = RiskLevel.SAFE) -> list[str]:
        """Register a connected MCP client's tools + retain it for cleanup (G6).

        Returns the registered tool names (so callers can revoke them later).
        Used by the ``/v1/.../mcp/servers`` POST endpoint.
        """
        from koboi.mcp.base import register_mcp_tools

        names: list[str] = []
        if self._core is not None:
            names = register_mcp_tools(client, self._core.tools, group=group, risk_level=risk_level)
        self._mcp_clients.append(client)
        return names

    def remove_mcp_client(self, client) -> None:
        """Disable a client's tools, close it, and drop it (G6 DELETE).

        29-H: close/disable failures are logged (not silently passed) so a lingering
        resource is observable."""
        if self._core is not None:
            try:
                self._core.tools.disable(list(client.tool_names))
            except Exception as e:  # noqa: BLE001
                logging.getLogger(__name__).warning("MCP tool disable failed for %r: %s", client.name, e)
        try:
            client.close()
        except Exception as e:  # noqa: BLE001
            logging.getLogger(__name__).warning("MCP client close failed for %r: %s", client.name, e)
        if client in self._mcp_clients:
            self._mcp_clients.remove(client)


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


def _resolve_replay_mode(config: Config) -> str:
    """Effective replay mode. Precedence: top-level ``replay.mode`` (incl. the
    CLI-injected value set by ``Config.with_replay``) > workflow-level
    ``orchestration.determinism.replay_mode`` (when not ``live``) > ``live``."""
    mode = config.replay.get("mode") or "live"
    if mode == "live":
        det = (config.orchestration.get("determinism") or {}).get("replay_mode")
        if det and det != "live":
            return det
    return mode


def _resolve_cache_dir(config: Config) -> str:
    return config.replay.get("cache_dir") or ".koboi/cache"


def _maybe_wrap_cache(client: Client, config: Config) -> Client:
    """Wrap a chat client in ``CachedClient`` for ``cache`` / ``replay`` modes.

    ``cache`` (STORE) memoizes live responses and replays on an identical request
    (live on a miss). ``replay`` (RAISE) is pure-offline: a miss raises
    ``CacheMissError`` (no live call, no API key for cached completions) -- the
    honest signal that the run diverged from the cached trajectory. Idempotent
    (never double-wraps). Embeddings are NOT wrapped (separate builder)."""
    mode = _resolve_replay_mode(config)
    if mode not in ("cache", "replay"):
        return client
    from koboi.llm.cache import CacheMissPolicy, CachedClient, ResponseCache

    if isinstance(client, CachedClient):
        return client
    on_miss = CacheMissPolicy.RAISE if mode == "replay" else CacheMissPolicy.STORE
    return CachedClient(client, ResponseCache(_resolve_cache_dir(config)), on_miss=on_miss)


def _build_tools(config: Config) -> ToolRegistry:
    registry = ToolRegistry()
    builtin_list = config.get("tools", "builtin", default=[])
    if builtin_list:
        from koboi.tools.builtin import register_all

        register_all(registry)
        # W0/W1: inject the registry-resolved web providers (koboi.websearch). web_search /
        # web_fetch read these from their _deps; absent web: config -> mock search and
        # httpx+readability fetch (offline-safe defaults).
        from koboi.websearch import build_fetch_provider, build_search_provider, load_custom_components

        websearch_conf = config.get("websearch", default={})
        if websearch_conf.get("custom_modules"):
            load_custom_components(websearch_conf["custom_modules"])
        registry.set_dep("search_provider", build_search_provider(websearch_conf))
        registry.set_dep("fetch_provider", build_fetch_provider(websearch_conf))
        # Inject per-agent memory store so agents don't share state
        from koboi.tools.builtin.memory import _MemoryStore

        memory_file = config.get("tools", "memory_file", default=".agent_memory.json")
        registry.set_dep("memory_store_ref", _MemoryStore(filepath=memory_file))
        if builtin_list and isinstance(builtin_list, list):
            registry.keep_only(builtin_list)

    # W0/W5b: inject the media backend whenever media is enabled -- OUTSIDE the builtin_list gate.
    media_conf = config.get("media", default={})
    if media_conf and media_conf.get("enabled"):
        from koboi.media import build_media

        media_backend = build_media(media_conf)
        if media_backend is not None:
            registry.set_dep("media_provider", media_backend)

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

    mgr = build_context(strategy, logger=logger, client=client, **kwargs)
    if mgr is not None:
        # Issue #5: optional safety margin (reserves headroom for the response).
        safety_margin = config.get("context", "safety_margin", default=0)
        if isinstance(safety_margin, int) and not isinstance(safety_margin, bool) and safety_margin > 0:
            mgr.safety_margin = safety_margin
        # Issue #3: optional real tokenizer (OpenAI + tiktoken only; else heuristic).
        if client is not None:
            from koboi.tokens import make_tokenizer

            tok = make_tokenizer(getattr(client, "provider", None), getattr(client, "model", None))
            if tok is not None:
                mgr.tokenizer = tok
    return mgr


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


def _warn_semantic_without_embeddings(config: Config, has_dedicated_embedding_client: bool) -> None:
    """Best-effort startup nudge: if semantic/hybrid retrieval is selected but no
    dedicated ``embedding:`` client was built and the (inline) chat provider has no
    embeddings endpoint (e.g. anthropic), warn so the user knows semantic will fall
    back to keyword. Named-ref/pool providers are still covered by the retriever's
    lazy runtime warning; this is the early, build-time signal.
    """
    retriever = config.get("rag", "retriever", default="keyword")
    if retriever not in ("semantic", "hybrid") or has_dedicated_embedding_client:
        return
    llm = config.get("llm", default={})
    provider = llm.get("provider", "openai") if isinstance(llm, dict) else "openai"
    if provider == "anthropic":
        logging.getLogger(__name__).warning(
            "RAG retriever %r needs embeddings, but the chat provider is 'anthropic' "
            "(no embeddings endpoint) and no top-level `embedding:` section is "
            "configured -- semantic retrieval will fall back to keyword. Add an "
            "`embedding:` section to enable it.",
            retriever,
        )


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
    _warn_semantic_without_embeddings(config, has_dedicated_embedding_client=rag_client is not client)
    # #9: pass the chat client separately so query rewriting can use a chat model
    # (rag_client above is the embedding client for the semantic leg).
    return build_rag(rag_dict, client=rag_client, chat_client=client, logger=logger)


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
        self.client = _maybe_wrap_cache(_resolve_chat_client(self.config, self.logger), self.config)
        return self.client

    def build_memory(self) -> object:
        memory_conf = self.config.get("memory", default={})
        memory_backend = memory_conf.get("backend", "sqlite")
        if memory_backend == "sqlite":
            from koboi.memory_sqlite import SQLiteMemory

            # Issue #4b: optional per-session message retention cap (default None
            # = unbounded, preserves full-transcript durability).
            retention_cap = self.config.get("memory", "retention", "max_messages", default=None)
            if isinstance(retention_cap, bool):  # guard: bool is an int subclass
                retention_cap = None
            # Issue #2: optional tenant/owner tag (schema prep for multi-tenancy).
            owner = self.config.get("memory", "owner", default=None)
            self.memory = SQLiteMemory(
                db_path=memory_conf.get("db_path", "koboi_memory.db"),
                session_id=memory_conf.get("session_id"),
                logger=self.logger,
                system_prompt=self.config.system_prompt or None,
                retention_cap=retention_cap,
                owner=owner,
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
        # Issue #4a: give the context manager a per-session metadata handle so it
        # can persist cross-restart state (e.g. sliding_window summary). Only
        # SQLiteMemory exposes get_meta/set_meta today.
        if self.context_manager is not None and self.memory is not None:
            from koboi.memory_sqlite import SQLiteMemory

            if isinstance(self.memory, SQLiteMemory):
                self.context_manager.meta_store = self.memory
        return self.context_manager

    def build_rag(self) -> object:
        self.augmentation = _build_rag(self.config, self.client, self.logger)
        # W3: opt-in live corpus -- swap the augmentation retriever for a LiveRetriever over a
        # shared LiveCorpus (seeded with the static chunks) and inject it as the live_corpus dep
        # so the ingest_url tool can grow it mid-conversation (rag.live + tools.builtin:
        # [ingest_url, ...]). add_chunks is cheap; the KeywordRetriever delegate rebuilds lazily.
        if self.augmentation is not None and self.config.get("rag", "live", default=False):
            from koboi.rag.live import LiveCorpus, LiveRetriever

            seed = getattr(self.augmentation.retriever, "_chunks", []) or []
            # W5 B2: optionally seed the live corpus from a research run's persisted findings
            # (SourceStore.to_corpus_file jsonl) -- the research->corpus convergence.
            seed_file = self.config.get("rag", "live_seed_file", default=None)
            if seed_file:
                seeded = LiveCorpus.from_corpus_file(seed_file)
                if seeded is not None:
                    seed = seeded.chunks
            corpus = LiveCorpus(seed)
            self.augmentation.retriever = LiveRetriever(corpus)
            tools = getattr(self, "tools", None)
            if tools is not None:
                tools.set_dep("live_corpus", corpus)
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

    def build_proactive_memory(self):
        """Build the proactive long-term-memory coordinator (opt-in; None unless enabled).

        Constructed only when ``memory.proactive.enabled`` is true. Reuses the
        KV ``_MemoryStore`` (creating one if the memory tool isn't registered)
        and the embedding client (dedicated if configured, else the chat client).
        """
        cfg = self.config.get("memory", "proactive", default={}) or {}
        if not cfg.get("enabled"):
            self.proactive_memory = None
            return None
        from koboi.proactive_memory import ProactiveMemory
        from koboi.tools.builtin.memory import _MemoryStore

        store = self.tools.get_dep("memory_store_ref") if self.tools else None
        if store is None:
            store = _MemoryStore(filepath=self.config.get("tools", "memory_file", default=".agent_memory.json"))
        embedding_client = _build_embedding_client(self.config, self.logger) or self.client
        self.proactive_memory = ProactiveMemory(
            client=self.client,
            embedding_client=embedding_client,
            memory=self.memory,
            store=store,
            config=cfg,
        )
        return self.proactive_memory

    def build(self, peer_registry: PeerRegistry | None = None) -> KoboiAgent:
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
        self.build_proactive_memory()
        self.build_hooks()

        # YAML-driven external command hooks (hooks: section). Gated by
        # hooks.allow_exec (default-deny); each command runs under self.sandbox.
        if self.hook_chain:
            _build_command_hooks(self.config, self.sandbox, self.hook_chain)

        _setup_subagent(self.tools, self.client, self.hook_chain, self.logger, memory=self.memory, config=self.config)
        _setup_tasks(self.tools, self.config, hook_chain=self.hook_chain)
        _setup_peer_registry(self.tools, self.config, peer_registry=peer_registry)

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

        # D: proactive long-term-memory extraction (opt-in). Runs at SESSION_END.
        if self.proactive_memory is not None and self.proactive_memory.extract_enabled and self.hook_chain:
            from koboi.hooks.proactive_extraction_hook import ProactiveExtractionHook

            self.hook_chain.add(ProactiveExtractionHook(proactive=self.proactive_memory))

        # B1.5: structural handover detection (opt-in). A3-fed (GroundingGuardrail ref).
        if self.hook_chain and self.config.get("handover", "detection", "enabled", default=False):
            from koboi.guardrails.grounding import GroundingGuardrail
            from koboi.hooks.handover_detection_hook import HandoverDetectionHook

            grounding = next(
                (g for g in (self.output_guardrails or []) if isinstance(g, GroundingGuardrail)),
                None,
            )
            if grounding is None:
                logging.getLogger(__name__).warning(
                    "handover.detection enabled without a grounding_check output guardrail "
                    "-- coverage-based handover will be inert; only explicit user-ask will trigger"
                )
            self.hook_chain.add(
                HandoverDetectionHook(
                    grounding=grounding,
                    coverage_threshold=self.config.get("handover", "detection", "coverage_threshold", default=0.5),
                    ask_patterns=self.config.get("handover", "detection", "ask_patterns", default=None),
                )
            )

        # Self-healing P1: tool-grounded reflection loop (opt-in). Verifier-fed
        # (GroundingGuardrail ref for low-grounding; P0-D errored signal for tools).
        if self.hook_chain and self.config.get("self_healing", "enabled", default=False):
            from koboi.guardrails.grounding import GroundingGuardrail
            from koboi.hooks.reflection_hook import ReflectionHook
            from koboi.llm.resolve import resolve_llm_spec

            grounding = next(
                (g for g in (self.output_guardrails or []) if isinstance(g, GroundingGuardrail)),
                None,
            )
            if grounding is None:
                logging.getLogger(__name__).warning(
                    "self_healing.enabled without a grounding_check output guardrail "
                    "-- low-grounding reflection will be inert; only tool-error critique fires"
                )
            critic_client = self.client
            critic_spec = self.config.get("self_healing", "critic_llm", default=None)
            if critic_spec:
                try:  # fail-soft: fall back to the main client on any resolve/build error
                    resolved = resolve_llm_spec(critic_spec, self.config) or {}
                    from koboi.llm.factory import create_client

                    critic_client = create_client(
                        provider=resolved.get("provider") or "openai",
                        model=resolved.get("model") or "",
                        api_key=resolved.get("api_key") or "",
                        base_url=resolved.get("base_url") or "",
                    )
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "self_healing.critic_llm resolve/build failed (%s); reusing main client", exc
                    )
                    critic_client = self.client
            self.hook_chain.add(
                ReflectionHook(
                    client=critic_client,
                    grounding=grounding,
                    max_turns=self.config.get("self_healing", "max_turns", default=3),
                    fail_soft=self.config.get("self_healing", "fail_soft", default=True),
                    tool_error_threshold=self.config.get(
                        "self_healing", "triggers", "tool_error", "repeat_threshold", default=2
                    ),
                    grounding_threshold=self.config.get(
                        "self_healing", "triggers", "low_grounding", "threshold", default=0.6
                    ),
                )
            )

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
            proactive_memory=self.proactive_memory,
        )

        return KoboiAgent(
            core=core,
            config=self.config,
            logger=self.logger,
            mcp_clients=self.mcp_clients,
            mode_manager=self.mode_manager,
            trust_db=self.trust_db,
        )


def _connect_mcp_servers(config: Config, logger: AgentLogger) -> list[tuple[BaseMCPClient, dict]]:
    """Connect all configured MCP servers with retry + fail_fast (G4/G10).

    Returns ``[(client, server_conf), ...]`` for each successfully connected server
    (failed servers are warn+skipped unless ``mcp.fail_fast``). Shared by the
    single-agent path (``_build_mcp``) and the orchestration path (``_build_orchestration``).
    """
    servers = config.get("mcp", "servers", default=[])
    if not servers:
        return []

    fail_fast = config.get("mcp", "fail_fast", default=False)
    connect_retries = int(config.get("mcp", "connect_retries", default=2))
    backoff_base = float(config.get("mcp", "connect_backoff_base", default=2.0))

    pairs: list[tuple[BaseMCPClient, dict]] = []
    for server_conf in servers:
        transport = server_conf.get("transport", "stdio")
        try:
            client = _create_mcp_client(server_conf, transport, logger, config)
            _connect_with_retry(client, connect_retries, backoff_base)
            pairs.append((client, server_conf))
        except Exception as e:
            if fail_fast:
                raise
            # 24-F: route through the AgentLogger (reaches the session log dir) when
            #       available; fall back to stdlib only if no logger was passed.
            msg = f"MCP server connection failed for '{server_conf.get('url') or server_conf.get('command', '?')}': {e}"
            if logger is not None:
                logger.log(msg)
            else:
                logging.getLogger(__name__).warning("%s", msg)
    return pairs


def _mcp_namespace_prefix(idx: int, server_conf: dict, config: Config) -> str | None:
    namespace = bool(config.get("mcp", "namespace", default=False))
    if not namespace:
        return None
    group = server_conf.get("group")
    return f"mcp__{group or idx}"


def _mcp_registrar_for_pairs(pairs: list[tuple[BaseMCPClient, dict]], config: Config):
    """Return a closure that registers shared MCP tools (group/risk/namespace) into a registry (G5).

    Used by the orchestration path so every sub-agent's per-agent ToolRegistry gets the
    SAME shared MCP clients registered (one subprocess/connection per server).
    """
    from koboi.mcp.base import default_risk_heuristic, register_mcp_tools

    def registrar(registry: ToolRegistry) -> None:
        for idx, (client, server_conf) in enumerate(pairs):
            resolver = default_risk_heuristic if server_conf.get("risk_heuristic", False) else None
            register_mcp_tools(
                client,
                registry,
                group=server_conf.get("group"),
                risk_level=_mcp_risk_level(server_conf),
                risk_resolver=resolver,
                namespace_prefix=_mcp_namespace_prefix(idx, server_conf, config),
            )

    return registrar


def _build_mcp(config: Config, tools: ToolRegistry, logger: AgentLogger) -> list:
    """Connect to MCP servers from config and register their tools."""
    from koboi.mcp.base import default_risk_heuristic, register_mcp_tools

    pairs = _connect_mcp_servers(config, logger)
    clients: list[BaseMCPClient] = []
    for idx, (client, server_conf) in enumerate(pairs):
        resolver = default_risk_heuristic if server_conf.get("risk_heuristic", False) else None
        register_mcp_tools(
            client,
            tools,
            group=server_conf.get("group"),
            risk_level=_mcp_risk_level(server_conf),
            risk_resolver=resolver,
            namespace_prefix=_mcp_namespace_prefix(idx, server_conf, config),
        )
        clients.append(client)
    return clients


def _connect_with_retry(client: BaseMCPClient, connect_retries: int, backoff_base: float) -> None:
    """Connect an MCP client with exponential backoff (G4).

    Mirrors the RetryClient backoff shape (koboi/client.py): attempt 0..connect_retries,
    ``wait = backoff_base ** attempt`` between retries. The last failure propagates so the
    caller (``_build_mcp``) can re-raise under ``fail_fast`` or log-and-skip.
    """
    import time

    last_exc: Exception | None = None
    for attempt in range(connect_retries + 1):
        try:
            client.connect()
            return
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if attempt < connect_retries:
                time.sleep(backoff_base**attempt)
    # Loop ran without returning -> connect failed every attempt; last_exc is set.
    if last_exc is None:  # pragma: no cover - unreachable: loop body always sets it
        raise RuntimeError("connect retry loop did not record an exception")
    raise last_exc


def _mcp_risk_level(server_conf: dict) -> RiskLevel:
    """Map a server's configured ``risk_level`` string to a RiskLevel (G3). Defaults to SAFE."""
    mapping = {
        "safe": RiskLevel.SAFE,
        "moderate": RiskLevel.MODERATE,
        "destructive": RiskLevel.DESTRUCTIVE,
    }
    raw = str(server_conf.get("risk_level", "safe")).lower()
    return mapping.get(raw, RiskLevel.SAFE)


# M4: allowed MCP stdio runners (basename match). Extend via mcp.allowlist_commands.
_MCP_DEFAULT_RUNNERS = frozenset({"npx", "uvx", "python", "python3", "node", "uv", "deno", "bun"})


def _create_mcp_client(
    server_conf: dict, transport: str, logger: AgentLogger, config: Config | None = None
) -> BaseMCPClient:
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


def _build_command_hooks(config: Config, sandbox: BaseSandbox, hook_chain) -> None:
    """Read the ``hooks:`` section, enforce the ``allow_exec`` gate, and wire
    :class:`~koboi.hooks.command_hook.CommandHook` entries into ``hook_chain``.

    No-op when no ``hooks.on_event`` entries are declared. Security is layered:
    ``allow_exec`` defaults to false (default-deny) -- declared hooks are skipped
    with a warning until the operator opts in. Each command runs through ``sandbox``
    (isolation + secret-hygiened env) and is offloaded off-loop inside CommandHook.
    """
    from koboi.hooks.chain import HookEvent
    from koboi.hooks.command_hook import CommandHook

    log = logging.getLogger("koboi.hooks")
    hooks_conf = config.get("hooks", default={}) or {}
    entries = hooks_conf.get("on_event", []) or []
    if not entries:
        return

    if not hooks_conf.get("allow_exec", False):
        log.warning(
            "hooks.on_event declares %d command hook(s) but hooks.allow_exec is false "
            "(default-deny); they will NOT run. Set hooks.allow_exec: true to enable.",
            len(entries),
        )
        return

    # R4: seccomp hard-blocks all egress -> messaging/forwarding hooks (uvx /
    # WhatsApp / Telegram) would fail. Warn (don't hard-fail) so non-network hooks
    # remain usable.
    if config.get("sandbox", "network_isolation", default=None) == "seccomp":
        log.warning(
            "hooks: command hooks are configured with sandbox.network_isolation='seccomp', "
            "which hard-blocks all network egress -- messaging/forwarding hooks will fail. "
            "Drop seccomp or run those hooks out-of-band."
        )

    default_timeout = hooks_conf.get("command_timeout", 10.0)
    for entry in entries:
        resolved: list[HookEvent] = []
        for ev in entry.get("events", []):
            try:
                resolved.append(HookEvent(ev))
            except ValueError:
                valid = [e.value for e in HookEvent]
                raise ValueError(
                    f"hooks.on_event: unknown event {ev!r} for hook "
                    f"{entry.get('name') or entry.get('command')!r}. Valid events: {valid}"
                ) from None
        hook = CommandHook(
            command=entry["command"],
            events=resolved,
            sandbox=sandbox,
            logger=log,
            name=entry.get("name"),
            fire_and_forget=entry.get("fire_and_forget", True),
            timeout=entry.get("timeout") or default_timeout,
            priority=entry.get("priority", 50),
            abort_on_error=entry.get("abort_on_error", False),
            pass_messages=entry.get("pass_messages", False),
            pass_metadata=entry.get("pass_metadata", False),
            env_passthrough=entry.get("env_passthrough", False),
            cwd=entry.get("cwd"),
        )
        hook_chain.add(hook)
        log.info(
            "hooks: enabled command hook %r on events %s",
            hook.name,
            [e.value for e in resolved],
        )


# ---------------------------------------------------------------------------
# Orchestration support
# ---------------------------------------------------------------------------


def _parse_agent_defs(config: Config) -> list:
    """Parse AgentDef list from orchestration.agents config.

    Applies the workflow-level determinism profile (``orchestration.determinism``)
    merged with any per-node ``determinism`` block into each node's ``llm_config``
    (via :func:`_apply_determinism`) so a dedicated pinned LLM client is built
    through the existing ``_has_client_overrides`` path -- no change to
    ``_agent_client_builder``. ``output_schema`` and
    ``force_response_format_with_tools`` are captured here for the
    response_format path (Gap A/B).
    """
    from koboi.types import AgentDef

    agents_conf = config.orchestration.get("agents", [])
    if not agents_conf:
        raise ValueError("orchestration.agents must have at least one agent")

    wf_det = config.orchestration.get("determinism") or {}
    defs = []
    for ac in agents_conf:
        if not ac.get("name", ""):
            raise ValueError("Each orchestration agent must have a 'name'")
        agent_def = AgentDef.from_dict(ac)
        _apply_determinism(agent_def, wf_det)
        defs.append(agent_def)
    return defs


def _apply_determinism(agent_def, wf_det: dict) -> None:
    """Merge workflow-level + per-node determinism into the node's ``llm_config``.

    Per-node ``determinism`` overrides the workflow-level profile; explicit node
    ``llm_config`` values are preserved (``setdefault``) so determinism only
    fills gaps. The merged knobs (temperature/seed/top_p/model) make
    ``_has_client_overrides`` return True, so the UNCHANGED ``_agent_client_builder``
    builds a dedicated pinned client for this node. ``seed`` auto-forwards via
    ``extract_extra_params`` (and is dropped on Anthropic by
    ``_filter_extra_params_for_provider``).

    A node whose ``llm_config`` is a string (a named ``providers:`` ref that fully
    replaces the client) opts out of determinism pinning -- the knobs cannot merge
    into a string, and the named ref already specifies provider/model.
    """
    from koboi.workflows import DeterminismProfile

    node_profile = DeterminismProfile.from_dict(agent_def.determinism)
    wf_profile = DeterminismProfile.from_dict(wf_det)
    base = wf_profile or node_profile
    if base is None:
        return
    effective = base.merge(node_profile) if (wf_profile and node_profile) else base
    overrides = effective.to_llm_overrides()
    if not overrides:
        return
    if isinstance(agent_def.llm_config, str):
        # Named providers: ref -- determinism knobs can't merge into a string.
        return
    llm = dict(agent_def.llm_config or {})
    for key, value in overrides.items():
        llm.setdefault(key, value)  # explicit node llm values preserved
    agent_def.llm_config = llm or None


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


def _build_orchestration(config: Config, verbose: bool = False, peer_registry: PeerRegistry | None = None):
    """Build a KoboiAgent backed by the orchestration engine.

    Reuses AgentAssembler for common subsystems (logger, client, guardrails,
    policy, hooks, skills, mode_manager, trust_db) so orchestration mode
    gets the same policy enforcement and guardrails as single-agent mode.

    Cache/replay coverage (v2/v3): EVERY chat LLM call in orchestration flows
    through ``_maybe_wrap_cache`` -- the shared ``assembler.client`` (router,
    planner, synthesis, dynamic builder) via ``build_client``, and per-node
    clients via ``_agent_client_builder``. Three call paths are currently
    UNREACHABLE from here and therefore uncached: ``QualityEvaluator`` (only
    constructed in tests), ``ProactiveExtractionHook`` (not attached in
    orchestration mode), and ``deep_research`` (docs-only on this branch). If
    any becomes reachable, route it through ``_maybe_wrap_cache`` or it
    egresses live during a cache/replay run.
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

    orch_conf = config.orchestration
    exec_conf = orch_conf.get("execution", {})
    exec_mode = exec_conf.get("mode", "sequential")

    # Dynamic / deep_research: agents are planned at runtime from the query (no config agents).
    agent_defs = [] if exec_mode in ("dynamic", "deep_research") else _parse_agent_defs(config)
    router = _build_router(config, assembler.client, agent_defs)

    parent_rag = config.rag

    if agent_defs:
        # Per-agent LLM client builder. A named ``providers:`` ref (str) FULLY
        # REPLACES the top-level client; an inline dict MERGES over it (today's
        # behavior). Pool specs (W2) raise.
        def _agent_client_builder(agent_llm: dict | str) -> Client:
            c: Client
            if isinstance(agent_llm, str):
                c = _build_client_from_dict(resolve_llm_spec(agent_llm, config), assembler.logger)
            elif isinstance(agent_llm, dict) and "pool" in agent_llm:
                c = _build_pool_from_spec(agent_llm["pool"], config, assembler.logger)
            else:
                overrides = {k: v for k, v in agent_llm.items() if k != "max_context_tokens"}
                c = _build_client(config, assembler.logger, llm_overrides=overrides)
            return _maybe_wrap_cache(c, config)

    # G5: wire shared MCP clients into orchestration (default on). The shared clients
    # are connected once; the registrar re-registers their tools into each sub-agent's
    # per-agent ToolRegistry so every agent can call them through one subprocess/server.
    share_mcp = config.get("orchestration", "share_mcp", default=True)
    mcp_pairs = _connect_mcp_servers(config, assembler.logger) if share_mcp else []
    shared_mcp_clients = [client for client, _ in mcp_pairs]
    mcp_registrar = _mcp_registrar_for_pairs(mcp_pairs, config)

    # A2A: shared peer registry for orchestrated sub-agents. Use the provided
    # (server-verified) registry; else a fresh verification-disabled one (CLI path).
    from koboi.server.peers import build_peer_registry

    peer_registry = build_peer_registry(config.get("peers", default={}), verified_registry=peer_registry)

    if agent_defs:
        agents_map = AgentFactory.create_all_configured(
            agent_defs,
            assembler.client,
            assembler.logger,
            parent_rag_config=parent_rag,
            hook_chain=assembler.hook_chain,
            sandbox=assembler.sandbox,
            embedding_config=config.get("embedding"),
            client_builder=_agent_client_builder,
            mcp_registrar=mcp_registrar,
            peer_registry=peer_registry,
        )
    else:
        agents_map = {}

    # Build a DagScheduler when execution.mode == "dag", seeded with the per-agent
    # depends_on edges parsed from config (deterministic, testable).
    dag_scheduler = None
    if exec_mode == "dag":
        from koboi.orchestration.dag_scheduler import DagScheduler

        deps = {ad.name: list(ad.depends_on) for ad in agent_defs}
        conds = {ad.name: list(ad.conditionals) for ad in agent_defs if ad.conditionals}
        # Persist the graph plan when the memory backend is SQLite (durable; #3).
        dag_db_path = None
        if config.get("memory", "backend", default="sqlite") == "sqlite":
            dag_db_path = config.get("memory", "db_path", default="koboi_memory.db")
        interrupt_nodes = {ad.name for ad in agent_defs if ad.interrupt_after}
        dag_scheduler = DagScheduler(
            agents_map=agents_map, deps=deps, db_path=dag_db_path, conditionals=conds, interrupt_nodes=interrupt_nodes
        )

    if exec_mode == "deep_research":
        # deep_research plans nodes per-query (like dynamic) but needs a DagScheduler for
        # the db_path used to journal the ResearchContext (W2).
        from koboi.orchestration.dag_scheduler import DagScheduler

        deep_db_path = None
        if config.get("memory", "backend", default="sqlite") == "sqlite":
            deep_db_path = config.get("memory", "db_path", default="koboi_memory.db")
        dag_scheduler = DagScheduler(agents_map={}, deps={}, db_path=deep_db_path)

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
        hook_chain=assembler.hook_chain,
        full_graph=exec_conf.get("full_graph", False),
        max_replans=exec_conf.get("max_replans", 0),
        sandbox=assembler.sandbox,
        research=config.get("research", default={}),
        websearch_conf=config.get("websearch", default={}),
        session_id=config.get("memory", "session_id", default=None),
    )

    return KoboiAgent(
        core=None,
        config=config,
        logger=assembler.logger,
        mcp_clients=shared_mcp_clients,
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


def _setup_peer_registry(tools: ToolRegistry, config: Config, peer_registry: PeerRegistry | None = None) -> None:
    """Inject the A2A peer registry (and ensure the ``call_peer_agent`` tool exists).

    Uses ``peer_registry`` (the server's verified registry) when provided; else builds
    a fresh verification-disabled one (CLI path). When A2A is enabled the
    ``call_peer_agent`` tool is auto-registered (it's the front door to peers -- it
    shouldn't require a separate ``tools.builtin`` opt-in); no-op if already present.
    """
    from koboi.server.peers import build_peer_registry

    registry = build_peer_registry(config.get("peers", default={}), verified_registry=peer_registry)
    if registry is None:
        return  # A2A not enabled
    if "call_peer_agent" not in tools:
        from koboi.tools.builtin import peer as _peer_tool
        from koboi.tools.registry import register_decorated

        register_decorated(tools, _peer_tool)
    tools.set_dep("peer_registry", registry)


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
            # W6 C1a: propagate orchestrator metadata (deep_research research_sources/coverage/depth)
            # so t.* eval assertions + RunResult consumers can see them.
            **result.metadata,
        },
    )
