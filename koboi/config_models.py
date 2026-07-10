"""koboi/config_models.py -- Pydantic v2 schemas for config validation."""

from __future__ import annotations

import logging
from pydantic import BaseModel, Field, field_validator, model_validator

_logger = logging.getLogger(__name__)


def _warn_unknown_keys(data: dict, model: type[BaseModel], path: str = "") -> None:
    """Log warnings for keys not defined in the model's field set."""
    known = set(model.model_fields.keys())
    for key in data:
        if key not in known:
            dotted = f"{path}.{key}" if path else key
            _logger.warning("Unknown config key '%s' will be ignored (typo?)", dotted)


class AgentConfig(BaseModel):
    model_config = {"extra": "ignore"}

    name: str = "koboi-agent"
    description: str = ""
    system_prompt: str = ""
    max_iterations: int = Field(default=10, ge=1)
    mode: str = "chat"
    theme: str = "koboi-dark"
    # JSON Schema dict; when set, the agent requests provider-enforced structured
    # output (OpenAI native response_format / Anthropic forced-tool emulation).
    # Best for final-answer / single-shot structured responses; None = unchanged.
    output_schema: dict | None = None

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("agent.name is required")
        return v


class LLMConfig(BaseModel):
    model_config = {"extra": "ignore"}

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str = ""
    base_url: str = ""
    temperature: float | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    timeout: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    retry_backoff_base: float = Field(default=2.0, gt=0)
    auth_token: str = ""
    auth_type: str = "api_key"
    embedding_model: str = "text-embedding-3-small"
    api_version: str = "2023-06-01"
    transport_retries: int = Field(default=2, ge=0)

    @field_validator("model")
    @classmethod
    def model_must_not_be_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("llm.model is required")
        return v


class EmbeddingConfig(BaseModel):
    """Optional dedicated embedding provider, decoupled from the chat ``llm``.

    When ``api_key`` is set, semantic retrieval routes here instead of the chat
    client -- useful when the chat provider has no ``/embeddings`` endpoint. If
    unset, the chat client is used (and semantic falls back to keyword)."""

    model_config = {"extra": "ignore"}

    provider: str = "openai"
    base_url: str = ""
    api_key: str = ""
    model: str = "text-embedding-3-small"


class ToolsConfig(BaseModel):
    model_config = {"extra": "ignore"}

    builtin: list[str] = Field(default_factory=list)
    custom: list[dict] = Field(default_factory=list)
    defaults: dict = Field(default_factory=dict)
    overrides: dict = Field(default_factory=dict)
    # DENYLIST: remove these tools entirely (LLM view + execution).
    disabled: list[str] = Field(default_factory=list)
    # HIDE FROM LLM: only advertise these tool groups; tools stay executable.
    groups: list[str] | None = None


class ContextConfig(BaseModel):
    model_config = {"extra": "ignore"}

    strategy: str = "noop"
    max_context_tokens: int = Field(default=8000, ge=1)
    keep_last: int | None = None
    summarization_truncation: bool | int | None = None
    custom_modules: list[str] = Field(default_factory=list)
    # Issue #5: tokens of headroom reserved inside manage() so a single large
    # response/tool result can't push an over-budget payload before the next
    # iteration trims. Default 0 preserves prior behavior.
    safety_margin: int = Field(default=0, ge=0)


class RagConfig(BaseModel):
    model_config = {"extra": "ignore"}

    enabled: bool = False
    chunker: str = "sentence"
    chunk_size: int = Field(default=500, ge=1)
    retriever: str = "keyword"
    top_k: int = Field(default=3, ge=1)
    augmentation: str = "on_the_fly"
    documents: list[str | dict] = Field(default_factory=list)
    # #11a: wrap the chosen retriever in the lightweight RerankerRetriever.
    rerank: bool = False
    # #5: opt-in on-disk embedding cache (JSON) so restarts don't re-embed the corpus.
    embedding_cache_path: str | None = None
    # #1: opt-in on-disk cache for fetched remote documents (avoids re-fetch per session).
    document_cache_path: str | None = None
    # soft per-document size cap (MB); over-cap files are skipped + warned (OOM guard).
    max_document_size_mb: int = Field(default=10, ge=0)
    # #9: opt-in query rewriting (LLM) + HyDE (semantic/hybrid) before retrieval.
    query_rewrite: bool = False
    hyde: bool = False
    # #10: opt-in metadata filter for relevance scoping (NOT ACL).
    filter: dict | None = None


class InputGuardrailConfig(BaseModel):
    model_config = {"extra": "ignore"}

    detect_injection: bool = False
    max_length: int | None = None
    custom_patterns: list | None = None


class OutputGuardrailConfig(BaseModel):
    model_config = {"extra": "ignore"}

    detect_sensitive: bool = False


class RateLimitConfig(BaseModel):
    model_config = {"extra": "ignore"}

    max_calls_per_session: int = Field(default=100, ge=1)
    max_calls_per_minute: int = Field(default=20, ge=1)
    rate_window_seconds: float = Field(default=60.0, gt=0)


class AuditConfig(BaseModel):
    model_config = {"extra": "ignore"}

    db_path: str | None = None


class GuardrailsConfig(BaseModel):
    model_config = {"extra": "ignore"}

    input: InputGuardrailConfig = Field(default_factory=InputGuardrailConfig)
    output: OutputGuardrailConfig = Field(default_factory=OutputGuardrailConfig)
    rate_limit: RateLimitConfig | None = None
    audit: AuditConfig = Field(default_factory=AuditConfig)
    approval: dict = Field(default_factory=dict)


class PolicyRuleConfig(BaseModel):
    model_config = {"extra": "ignore"}

    tool: str = "*"
    pattern: str = ""
    action: str = "allow"
    # Per-argument glob patterns {arg_name: glob}. Generalizes the legacy
    # ``pattern`` shorthand (which only matched an arg literally named "command").
    # Example: {filename: "*.env"} denies any tool whose ``filename`` arg matches.
    argument_patterns: dict[str, str] | None = None


class PolicyConfig(BaseModel):
    model_config = {"extra": "ignore"}

    rules: list[PolicyRuleConfig] = Field(default_factory=list)


class MemoryRetentionConfig(BaseModel):
    model_config = {"extra": "ignore"}

    # Issue #4b: cap the stored message rows per session (oldest pruned). None =
    # unbounded (default, preserves full-transcript durability).
    max_messages: int | None = None


class ProactiveMemoryConfig(BaseModel):
    """Opt-in proactive long-term memory (extract D / recall C / core-block B).

    Master switch ``enabled`` (default False) = zero behavior change. Sub-toggles
    select which features run. Recall embeds the user message and injects the
    top-N stored facts each turn (no tool call); extraction pulls durable facts
    after the run; the core block is a small always-in-context summary.
    """

    model_config = {"extra": "ignore"}

    enabled: bool = False
    extract: bool = False  # D: auto-extract durable facts after each run
    recall: bool = False  # C: semantic recall + inject top-N each turn
    core_block: bool = False  # B: always-in-context core-memory block
    top_k: int = Field(default=4, ge=1)  # facts injected per turn (C)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)  # cosine floor (C)
    max_facts: int = Field(default=200, ge=1)  # cap the embedded KV set (C)


class MemoryConfig(BaseModel):
    model_config = {"extra": "ignore"}

    backend: str = "sqlite"
    db_path: str = "koboi_memory.db"
    session_id: str | None = None
    retention: MemoryRetentionConfig = Field(default_factory=MemoryRetentionConfig)
    # Issue #2: optional tenant/owner tag stamped on stored rows (schema prep for
    # multi-tenancy). None = untagged (today's behavior).
    owner: str | None = None
    # Proactive long-term memory (extract/recall/core-block). Master switch is
    # `enabled`; sub-toggles select features. All default off (zero behavior
    # change unless a config opts in).
    proactive: ProactiveMemoryConfig = Field(default_factory=ProactiveMemoryConfig)


class HarnessConfig(BaseModel):
    model_config = {"extra": "ignore"}

    telemetry: bool = False
    carryover: bool = False
    doom_loop: dict | None = None
    tasks: dict | None = None
    notifications: dict | None = None


class TracingConfig(BaseModel):
    model_config = {"extra": "ignore"}

    provider: str | None = None
    public_key: str = ""
    secret_key: str = ""
    base_url: str = "http://localhost:3300"


class SkillsConfig(BaseModel):
    model_config = {"extra": "ignore"}

    search_paths: list[str] = Field(default_factory=list)
    budget_chars: int = Field(default=8000, ge=0)


class OrchestrationConfig(BaseModel):
    model_config = {"extra": "ignore"}

    enabled: bool = False
    router: dict = Field(default_factory=dict)
    execution: dict = Field(default_factory=dict)
    agents: list[dict] = Field(default_factory=list)


class MCPAuthConfig(BaseModel):
    model_config = {"extra": "ignore"}

    type: str = "none"  # "none" | "bearer"
    token: str = ""


class MCPServerConfig(BaseModel):
    model_config = {"extra": "ignore"}

    # Stdio transport (existing)
    command: str = ""
    args: list[str] = Field(default_factory=list)

    # Streamable HTTP transport (new)
    transport: str = "stdio"  # "stdio" | "streamable-http"
    url: str = ""
    auth: MCPAuthConfig | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    timeout: float = 30.0
    group: str | None = None  # Tool group namespace for filtering
    # Risk gating for tools exposed by this server. Pre-#5 behavior is SAFE for all
    # MCP tools. risk_level overrides for every tool from this server; risk_heuristic
    # infers per-tool risk from the tool name (delete/remove->DESTRUCTIVE, etc.).
    # NOTE: a non-SAFE risk only gates when guardrails.approval or policy.rules is
    # configured -- otherwise the level is informational.
    risk_level: str = "safe"
    risk_heuristic: bool = False


class MCPConfig(BaseModel):
    model_config = {"extra": "ignore"}

    servers: list[MCPServerConfig] = Field(default_factory=list)


class RlimitsConfig(BaseModel):
    """POSIX resource limits applied to restricted sandbox subprocesses.

    Applied in the child via ``preexec_fn`` (RLIMIT_*). ``as_mb`` is
    best-effort on Darwin; ``cpu`` (seconds) and ``fsize_mb`` are hard limits.
    """

    model_config = {"extra": "ignore"}

    cpu: int | None = None
    as_mb: int | None = None
    fsize_mb: int | None = None
    nofile: int | None = None


class SandboxConfig(BaseModel):
    """Top-level ``sandbox:`` section -- subprocess/filesystem isolation.

    ``passthrough`` (default) preserves pre-P0b behavior; ``restricted`` adds
    cwd/env/PATH/network/rlimit containment. Docker (P0c) is deferred.
    """

    model_config = {"extra": "ignore"}

    backend: str = "passthrough"
    workdir: str = "."
    workdir_strategy: str = "shared"  # "shared" (legacy global) | "per_session" (M1 serving)
    network: str = "deny"
    network_binaries: list[str] = Field(default_factory=list)
    safe_path: list[str] = Field(default_factory=list)
    env_passthrough: bool = False
    rlimits: RlimitsConfig | None = None
    timeout: float = Field(default=30.0, gt=0)
    max_output: int = Field(default=10000, ge=1)


class JournalConfig(BaseModel):
    """Top-level ``journal:`` section -- step journal + resume (P2-A).

    The journal records one row per loop iteration and enables crash/redeploy
    recovery via ``koboi run --resume <session>``. Auto-disabled when the memory
    backend is not SQLite (it borrows the SQLite connection).
    """

    model_config = {"extra": "ignore"}

    enabled: bool = True
    record_tool_calls: bool = True


class ServerConfig(BaseModel):
    """Top-level ``server:`` section -- REST/SSE serving (M0 skeleton; M1+ wiring).

    M0 ships the schema only; no runtime code reads it yet. Nested groups
    (``pool``/``timeouts``/``limits``/``cors``/``idempotency``) are dicts now and
    are promoted to typed sub-models as each is consumed in M1+.
    """

    model_config = {"extra": "ignore"}

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    api_keys_file: str | None = None
    api_keys: list[str] = Field(default_factory=list)
    auth_required: bool = True
    docs_enabled: bool = False  # H7: serve /docs,/redoc,/openapi.json only when true
    cors: dict = Field(default_factory=dict)
    pool: dict = Field(default_factory=dict)
    timeouts: dict = Field(default_factory=dict)
    limits: dict = Field(default_factory=dict)
    idempotency: dict = Field(default_factory=dict)
    workdir_ttl_seconds: float = Field(default=86400.0, gt=0)
    # G2: operator policy boundary for per-request mode. Unset → the safe default
    # {chat, plan, act, auto}; yolo requires explicit opt-in. limits.max_iterations_cap
    # (default 25) clamps the per-request max_iterations knob.
    allowed_modes: list[str] = Field(default_factory=list)


class JobWebhookConfig(BaseModel):
    """One outbound webhook entry under ``jobs.webhooks``.

    POSTs a JSON job payload to ``url`` when the job reaches a terminal status in
    ``events`` (one of ``completed``/``cancelled``/``timed_out``/``failed``; empty
    = all). ``secret`` HMAC-SHA256-signs the body via the ``X-Koboi-Signature``
    header so receivers can verify integrity.
    """

    model_config = {"extra": "ignore"}

    url: str
    events: list[str] = Field(default_factory=list)  # terminal statuses; empty = all
    secret: str | None = None
    timeout: float | None = None  # seconds; None = default (10)


class JobsConfig(BaseModel):
    """Top-level ``jobs:`` section -- background/autonomous job runner (M0 skeleton; M4 wiring).

    Drives long-running agent runs outside the request lifecycle with
    resume-on-startup durability. M0 ships the schema only.
    """

    model_config = {"extra": "ignore"}

    enabled: bool = False
    max_concurrent: int = Field(default=64, ge=1)
    per_tenant_max: int = Field(default=5, ge=1)
    queue_depth: int = Field(default=32, ge=1)
    default_dedicated_session: bool = True
    event_buffer: dict = Field(default_factory=dict)
    resume_on_startup: bool = True
    timeout_seconds: float = Field(default=1800.0, gt=0)
    ttl_seconds: float = Field(default=86400.0, gt=0)
    webhooks: list[JobWebhookConfig] = Field(default_factory=list)


class CommandHookConfig(BaseModel):
    """One external executable hook entry under ``hooks.on_event``.

    Spawns ``command`` as a subprocess per ``events`` lifecycle event, passing a
    JSON HookContext on stdin and (when awaited) reading JSON mutations back.
    See ``docs/custom-hooks.md`` for the protocol + security model.
    """

    model_config = {"extra": "ignore"}

    command: list[str] | str  # list -> shell=False; str -> shell=True
    events: list[str] = Field(default_factory=list)  # HookEvent names; validated at build time
    fire_and_forget: bool = True  # True: observe/side-effect, not awaited (zero latency); False: full control
    timeout: float | None = None  # per-hook override; else hooks.command_timeout
    priority: int = 50
    abort_on_error: bool = False  # crash/timeout/non-2 -> abort? default False (fail-safe continue)
    pass_messages: bool = False  # include ctx.messages (can be MB-scale) in stdin payload
    pass_metadata: bool = False  # include ctx.metadata (may be non-serializable) in stdin payload
    env_passthrough: bool = False  # forwarded into build_env
    name: str | None = None

    @field_validator("events")
    @classmethod
    def _events_nonempty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("command hook must declare at least one event in `events`")
        return v


class HooksConfig(BaseModel):
    """Top-level ``hooks:`` section -- declarative external command hooks.

    SECURITY: command execution is opt-in. ``allow_exec`` MUST be true for any
    command hook to run; otherwise entries are ignored with a warning. Commands
    run via the sandbox backend (``build_env`` secret hygiene + timeout); see
    ``docs/custom-hooks.md`` for the layered security model + known gaps.
    """

    model_config = {"extra": "ignore"}

    allow_exec: bool = False  # default-DENY gate
    command_timeout: float = Field(default=10.0, gt=0)  # default per-invocation seconds
    on_event: list[CommandHookConfig] = Field(default_factory=list)


class KoboiConfig(BaseModel):
    """Top-level config schema for koboi-agent."""

    model_config = {"extra": "ignore"}

    agent: AgentConfig = Field(default_factory=AgentConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig | None = None
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    harness: HarnessConfig = Field(default_factory=HarnessConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    journal: JournalConfig = Field(default_factory=JournalConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    jobs: JobsConfig = Field(default_factory=JobsConfig)
    hooks: HooksConfig = Field(default_factory=HooksConfig)

    @model_validator(mode="before")
    @classmethod
    def check_required_fields(cls, data: dict) -> dict:
        _warn_unknown_keys(data, cls)
        errors: list[str] = []
        agent = data.get("agent")
        if agent is not None and isinstance(agent, dict) and not agent.get("name"):
            errors.append("agent.name is required")
        elif agent is None:
            errors.append("agent.name is required")
        llm = data.get("llm")
        # Tier 2: ``llm: {pool: name}`` resolves its model at runtime from the
        # pool's members, so the static ``llm.model`` requirement doesn't apply.
        if isinstance(llm, dict) and "pool" in llm:
            pass
        elif llm is not None and isinstance(llm, dict) and not llm.get("model"):
            errors.append("llm.model is required")
        elif llm is None:
            errors.append("llm.model is required")
        if errors:
            raise ValueError("; ".join(errors))
        return data
