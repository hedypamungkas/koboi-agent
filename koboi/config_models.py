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


class ModeConfig(BaseModel):
    """Mode-behavior overrides (distinct from ``agent.mode`` which selects the enum).

    ``read_only_tools`` extends ModeHook's built-in read-only allowlist so SAFE tools
    (e.g. read-only MCP tools) are also permitted in CHAT/PLAN (mode-block nuance).
    """

    model_config = {"extra": "ignore"}

    read_only_tools: list[str] = Field(default_factory=list)


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
    # #11a: rerank stage. ``True`` (legacy) wraps the retriever in the lightweight
    # heuristic RerankerRetriever; a DICT selects a true cross-encoder backend, e.g.
    # ``{provider: jina|cohere|local, api_key, model, ...}`` (see koboi/rag/rerank.py).
    rerank: bool | dict = False
    # W3: opt-in live corpus -- the facade swaps the augmentation retriever for a LiveRetriever
    # over a shared LiveCorpus + injects it as the ``live_corpus`` dep for the ingest_url tool.
    live: bool = False
    # W5: optional jsonl of a prior research run's findings (SourceStore.to_corpus_file output)
    # to seed the live corpus -- the research->corpus convergence. Ignored unless ``live: true``.
    live_seed_file: str | None = None
    # #5: opt-in on-disk embedding cache (JSON) so restarts don't re-embed the corpus.
    embedding_cache_path: str | None = None
    # #1: opt-in on-disk cache for fetched remote documents (avoids re-fetch per session).
    document_cache_path: str | None = None
    # soft per-document size cap (MB); over-cap files are skipped + warned (OOM guard).
    max_document_size_mb: int = Field(default=10, ge=0)
    # #9: opt-in query rewriting (LLM) + HyDE (semantic/hybrid) before retrieval.
    query_rewrite: bool = False
    hyde: bool = False
    # #10: opt-in metadata filter for relevance scoping (NOT ACL). Operators:
    # scalar (equality), {"$gte"/"$lte"/"$gt"/"$lt": v}, {"$in": [...]}.
    filter: dict | None = None


class WebSearchConfig(BaseModel):
    """Top-level ``websearch:`` section -- search/fetch provider selection (koboi.websearch).

    Cosmetic validation only at the top level; runtime resolution reads
    ``websearch.search`` / ``websearch.fetch`` as plain dicts via
    ``config.get("websearch", ...)`` in ``_build_tools`` (mirrors how ``RagConfig`` is
    consumed by ``build_rag``). Defaults are offline-safe (no provider configured ->
    ``mock`` search).
    """

    model_config = {"extra": "ignore"}

    search: dict = Field(default_factory=dict)
    fetch: dict = Field(default_factory=dict)
    # Dotted module paths imported so @register_search_provider/@register_fetch_provider
    # decorators fire on import (mirrors rag.custom_modules).
    custom_modules: list[str] = Field(default_factory=list)


class MediaConfig(BaseModel):
    """Top-level ``media:`` section -- multimodal generation."""

    model_config = {"extra": "ignore"}
    enabled: bool = False
    image: dict = Field(default_factory=dict)
    video: dict = Field(default_factory=dict)
    music: dict = Field(default_factory=dict)
    speech: dict = Field(default_factory=dict)
    transcription: dict = Field(default_factory=dict)
    budget: dict = Field(default_factory=dict)
    storage: dict = Field(default_factory=dict)
    custom_modules: list[str] = Field(default_factory=list)
    profiles: list[dict] = Field(default_factory=list)


class ResearchConfig(BaseModel):
    """Top-level ``research:`` section -- ``execution.mode: deep_research`` knobs (W2).

    Cosmetic validation only; runtime reads these via ``config.get("research", ...)`` in
    ``_build_orchestration`` (passed to the Orchestrator's ``research=`` kwarg).
    """

    model_config = {"extra": "ignore"}

    max_depth: int = Field(default=3, ge=1)
    max_searches: int = Field(default=15, ge=1)
    max_fetches: int = Field(default=20, ge=1)
    max_tokens: int = Field(default=0, ge=0)  # 0 = not enforced
    coverage_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    citations: str = "numbered"  # numbered | inline | none (W2 ships numbered)
    # Optional override of the research node tool bundle (default: web_search + web_fetch).
    tools: dict | None = None
    capabilities: list[str] = Field(default_factory=list)
    search_provider: str | None = None
    fetch_provider: str | None = None
    # W3: path to write the run's gathered findings as jsonl (cross-session corpus reuse).
    persist_findings: str | None = None
    media: dict = Field(default_factory=dict)


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
    output: OutputGuardrailConfig | list[dict] = Field(default_factory=OutputGuardrailConfig)
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


class ExecutionConfig(BaseModel):
    """Typed view of ``orchestration.execution`` (self-healing P0-B).

    ``extra="allow"`` so the many untyped execution keys (mode, full_graph,
    research caps, ...) pass through unchanged; this model only validates +
    documents the knobs we care about. The facade still reads the raw dict via
    ``config.get("orchestration", "execution", ...)`` -- this is the validated
    surface for those same keys.
    """

    model_config = {"extra": "allow"}

    max_replans: int = 0  # dynamic-mode re-plan budget on node failure (0 = opt-in/off)
    max_revisions: int = 2
    use_revision: bool = False
    full_graph: bool = False
    mode: str | None = None


class OrchestrationConfig(BaseModel):
    model_config = {"extra": "ignore"}

    enabled: bool = False
    router: dict = Field(default_factory=dict)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    agents: list[dict] = Field(default_factory=list)
    share_mcp: bool = True  # G5: wire shared MCP clients into orchestration sub-agents


class MCPAuthConfig(BaseModel):
    model_config = {"extra": "ignore"}

    type: str = "none"  # "none" | "bearer" | "oauth"
    token: str = ""  # static bearer token (type=bearer)
    # OAuth2 fields (type=oauth) -- client_credentials / refresh_token grant (G1)
    token_endpoint: str = ""
    client_id: str = ""
    client_secret: str = ""
    scopes: str = ""
    refresh_token: str = ""
    access_token: str = ""  # optional pre-seeded token
    expires_in: float | None = None


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
    fail_fast: bool = False  # raise (instead of warn+skip) when an MCP server fails to connect
    connect_retries: int = 2  # number of connect attempts after the first (backoff between)
    connect_backoff_base: float = 2.0  # backoff = base ** attempt seconds
    namespace: bool = False  # register MCP tools as mcp__<group|index>__<name> to avoid collisions
    allowlist_commands: list[str] = Field(default_factory=list)  # extra stdio runners (basename)


class PeerDef(BaseModel):
    """A peer koboi instance for cross-instance agent-to-agent (A2A) calls (config shape).

    Each registered peer URL is trusted as same-org/owner (static Bearer per peer).
    The ``token`` is the OUTBOUND credential presented to the peer; inbound peer
    tokens are listed separately under ``PeersConfig.inbound_tokens``. Named ``PeerDef``
    (not ``PeerConfig``) to avoid collision with the runtime ``koboi.server.peers.PeerConfig``
    dataclass.
    """

    model_config = {"extra": "ignore"}

    name: str  # unique peer key (used by the call_peer_agent tool)
    url: str  # remote instance base URL (e.g. http://peer-y:8000)
    token: str = ""  # outbound bearer presented to the peer (plaintext; same-org trust)
    agent_name: str = ""  # optional routing hint (which named agent to target on the peer)
    org: str = ""  # documentation/audit only (same-org grouping)
    timeout: float = Field(default=30.0, gt=0)  # outbound invoke timeout (seconds)


class PeersConfig(BaseModel):
    """Cross-instance A2A configuration (opt-in; inert by default)."""

    model_config = {"extra": "ignore"}

    enabled: bool = False
    peers: list[PeerDef] = Field(default_factory=list)  # outbound peers
    inbound_tokens: list[str] = Field(default_factory=list)  # plaintext tokens accepted from peers (hashed at load)
    # Same-org peers often live on private networks / localhost (dev, internal clusters).
    # Default False = strict SSRF (reject private/loopback URLs at load). Set True when the
    # operator vouches for these URLs; strict SSRF is reserved for untrusted discovery (P3).
    allow_private_network: bool = False
    # P3 (self-observing org-claim): the instance's org label + a shared HMAC secret.
    # When org_secret is set, each declared peer's agent-card is fetched at startup
    # and its HMAC org-claim verified; only verified peers are callable (verified-only).
    org: str = ""  # human-readable org label advertised in this instance's agent-card
    org_secret: str = ""  # shared HMAC-SHA256 secret proving same-org membership
    public_base_url: str = ""  # advertised base URL for this instance's agent-card peer_invoke_url
    rate_limit_per_minute: int = 60  # max inbound /v1/peer/invoke calls per peer token per minute (0 = unlimited)
    max_concurrent_inbound: int = 10  # max simultaneous /v1/peer/invoke calls per peer token (0 = unlimited)
    card_freshness_seconds: float = (
        21600  # agent-card freshness window for verify_card (default 6h; widen for clock-skew tolerance)
    )


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
    ``events`` (one of ``completed``/``cancelled``/``timed_out``/``failed``/
    ``awaiting_human``; empty = all). ``secret`` HMAC-SHA256-signs the body via the
    ``X-Koboi-Signature`` header so receivers can verify integrity.
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
    mode: ModeConfig = Field(default_factory=ModeConfig)
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
    websearch: WebSearchConfig = Field(default_factory=WebSearchConfig)
    media: MediaConfig = Field(default_factory=MediaConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    peers: PeersConfig = Field(default_factory=PeersConfig)

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
