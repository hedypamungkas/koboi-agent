"""koboi/types -- Core data types and dataclasses for the koboi framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, TYPE_CHECKING
from collections.abc import Callable

if TYPE_CHECKING:
    from koboi.exceptions import AgentError


class RiskLevel(Enum):
    SAFE = "safe"
    MODERATE = "moderate"
    DESTRUCTIVE = "destructive"


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict
    risk_level: RiskLevel = RiskLevel.SAFE
    timeout: float | None = None
    group: str | None = None
    # Issue #8b: whether re-running this tool on resume is safe. Default True
    # (most tools are pure/idempotent). Set False for side-effecting tools that
    # must not silently double-fire on crash-resume (e.g. charge_card, send_email);
    # the resume path then skips re-execution and records a synthetic result.
    idempotent: bool = True

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("ToolDefinition.name cannot be empty")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError(f"ToolDefinition.timeout must be positive, got {self.timeout}")


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class ToolResult:
    tool_call_id: str
    content: str


@dataclass
class ToolExecOutcome:
    """Structured outcome of executing a tool (self-healing P0-D).

    ``content`` is always the LLM-facing string (so ``ToolRegistry.execute()``
    keeps its ``-> str`` contract). ``errored`` / ``error_kind`` let the pipeline
    flag a failure WITHOUT fragile string-matching of the ``"Error:"`` prefix.
    """

    content: str
    errored: bool = False
    error_kind: str | None = None


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: Reasoning/thinking tokens (reasoning models: deepseek-v4-flash, mimo-v2.5,
    #: etc.). Reported under ``completion_tokens_details.reasoning_tokens`` by
    #: OpenAI-compatible gateways. Separate from completion_tokens (the answer).
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class AgentResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage | None = None
    #: Model + provider base URL that produced this response (for telemetry/E2E).
    model: str | None = None
    base_url: str | None = None

    @property
    def is_complete(self) -> bool:
        return not self.tool_calls


@dataclass
class RoutingDecision:
    query: str
    agents: list[str]
    confidence: float
    method: Literal["keyword", "llm", "hybrid(keyword)", "hybrid(llm)", "hybrid(keyword+llm)"]
    reasoning: str
    domain_label: str | None = None

    def __post_init__(self):
        if not self.agents:
            raise ValueError("RoutingDecision.agents cannot be empty")
        if not (0 <= self.confidence <= 1):
            raise ValueError(f"RoutingDecision.confidence must be in [0,1], got {self.confidence}")


@dataclass
class AgentResult:
    agent_name: str
    answer: str
    elapsed_seconds: float
    tokens_used: int
    revision_count: int = 0
    quality_score: float | None = None
    is_dynamic: bool = False
    domain_label: str | None = None
    failed: bool = False
    # Self-healing P2b: True if the node fired any non-idempotent (side-effecting)
    # tool. Such nodes must NOT be re-run on replan (their side effect already
    # happened) -- the replan loop carries them forward instead. Default False.
    had_non_idempotent_tool: bool = False
    tool_calls: list = field(default_factory=list)


@dataclass
class OrchestratorResult:
    query: str
    routing: RoutingDecision
    agent_results: list[AgentResult] = field(default_factory=list)
    final_answer: str = ""
    total_elapsed_seconds: float = 0.0
    execution_mode: Literal[
        "sequential",
        "parallel",
        "sequential+revision",
        "parallel+revision",
        "dag",
        "dynamic",
        "deep_research",
    ] = "sequential"
    # W2: deep_research stamps research_sources / coverage / depth; empty for other modes.
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentBlueprint:
    name: str
    domain_label: str
    system_prompt: str
    chunks: list[Any]
    chunker_config: dict
    retriever_top_k: int = 3
    source: str = "dynamic_llm"
    created_at: float = 0.0


@dataclass
class AgentDef:
    """Config-level definition of a single agent for orchestration."""

    name: str
    system_prompt: str = ""
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    tools_config: dict | None = None
    rag_config: dict | None = None
    llm_config: dict | None = None
    # DAG edges: agent names this agent depends on (execution.mode: dag).
    depends_on: list[str] = field(default_factory=list)
    # Conditional edges (#1): [{to: <node>, when: <predicate>}]. The predicate `when`
    # is evaluated on THIS node's output; if it matches, `to` is enabled.
    # Predicates: {contains: "str"} | {regex: "pattern"} | {field, op, value} on JSON.
    conditionals: list[dict] = field(default_factory=list)
    # #6: if True, the scheduler surfaces a [NODE_INTERRUPT] marker after this node
    # completes (for human review / HITL at the node boundary).
    interrupt_after: bool = False
    # JSON Schema dict (provider-agnostic) for structured node output. When set,
    # the node's final answer is constrained to the schema (response_format on
    # OpenAI/Cloudflare; forced tool_use emulation on Anthropic). See
    # loop.py ``_resolve_response_format`` for the provider-aware + tools guard.
    output_schema: dict | None = None
    # Gap B: when True, apply response_format even on iterations that carry tool
    # definitions. OpenAI/Cloudflare support RF + tools together; Anthropic does
    # not (RF is emulated via a forced tool_use, incompatible with real tools) so
    # it stays suppressed there regardless of this flag.
    force_response_format_with_tools: bool = False
    # Per-node determinism profile (temperature/seed/top_p/model_pin); merged over
    # the workflow-level profile by facade._apply_determinism before the node's
    # LLM client is built. Stored as a raw dict so it round-trips through YAML.
    determinism: dict | None = None
    # P2 (A2A): when set, this node is a REMOTE agent served by a peer koboi
    # instance. The value is a peer NAME (resolved via the peers: registry). The
    # factory returns a RemoteAgentProxy instead of a local AgentCore, so the node
    # participates in sequential/parallel/dag/conditional orchestration while
    # actually running on the peer. (dynamic/deep_research rebuild local agents
    # per-query, so endpoint is ignored there.)
    endpoint: str | None = None

    def to_dict(self) -> dict:
        """Serialize to the ``orchestration.agents[*]`` YAML shape.

        Inverse of :func:`koboi.facade._parse_agent_defs`: the ``*_config``
        fields are remapped to the YAML ``tools`` / ``rag`` / ``llm`` keys and
        empty/None sections are omitted so the dump stays clean.
        """
        out: dict = {"name": self.name}
        if self.system_prompt:
            out["system_prompt"] = self.system_prompt
        if self.description:
            out["description"] = self.description
        if self.keywords:
            out["keywords"] = list(self.keywords)
        if self.tools_config:
            out["tools"] = self.tools_config
        if self.rag_config:
            out["rag"] = self.rag_config
        if self.llm_config:
            out["llm"] = self.llm_config
        if self.depends_on:
            out["depends_on"] = list(self.depends_on)
        if self.conditionals:
            out["conditionals"] = list(self.conditionals)
        if self.interrupt_after:
            out["interrupt_after"] = self.interrupt_after
        if self.output_schema is not None:
            out["output_schema"] = self.output_schema
        if self.force_response_format_with_tools:
            out["force_response_format_with_tools"] = self.force_response_format_with_tools
        if self.determinism:
            out["determinism"] = self.determinism
        if self.endpoint:
            out["endpoint"] = self.endpoint
        return out

    @classmethod
    def from_dict(cls, ac: dict) -> AgentDef:
        """Build from the ``orchestration.agents[*]`` YAML shape.

        Mirrors :func:`koboi.facade._parse_agent_defs`. Accepts both the YAML
        keys (``tools`` / ``rag`` / ``llm``) and the dataclass ``*_config`` keys
        so a dict produced by :meth:`to_dict` round-trips cleanly.
        """
        return cls(
            name=ac.get("name", ""),
            system_prompt=ac.get("system_prompt", ""),
            description=ac.get("description", ""),
            keywords=list(ac.get("keywords") or []),
            tools_config=ac.get("tools", ac.get("tools_config")),
            rag_config=ac.get("rag", ac.get("rag_config")),
            llm_config=ac.get("llm", ac.get("llm_config")),
            depends_on=list(ac.get("depends_on") or []),
            conditionals=list(ac.get("conditionals") or []),
            interrupt_after=bool(ac.get("interrupt_after", False)),
            output_schema=ac.get("output_schema"),
            force_response_format_with_tools=bool(ac.get("force_response_format_with_tools", False)),
            determinism=ac.get("determinism") or None,
            endpoint=ac.get("endpoint"),
        )


@dataclass
class GuardrailResult:
    passed: bool
    reason: str = ""
    sanitized_content: str | None = None
    action: str = "block"


@dataclass
class AuditEntry:
    timestamp: float
    event_type: str
    tool_name: str | None = None
    arguments: str | None = None
    result: str | None = None
    risk_level: RiskLevel | str | None = None
    details: str = ""


@dataclass
class RateLimitConfig:
    max_tool_calls_per_session: int = 100
    max_calls_per_tool: dict[str, int] | None = None
    max_calls_per_minute: int = 20
    rate_window_seconds: float = 60.0


@dataclass
class MCPToolInfo:
    name: str
    description: str
    input_schema: dict


@dataclass
class MCPResource:
    """An MCP ``resources/*`` resource descriptor (G2)."""

    uri: str
    name: str = ""
    description: str = ""
    mime_type: str | None = None


@dataclass
class MCPPrompt:
    """An MCP ``prompts/*`` prompt descriptor (G2)."""

    name: str
    description: str = ""
    arguments: list = None  # type: ignore[assignment]  # list[dict] per spec; None = no args


@dataclass
class SkillDefinition:
    name: str
    description: str
    skill_dir: str
    body: str | None = None
    license: str | None = None
    compatibility: str | None = None
    metadata: dict | None = None
    allowed_tools: list[str] | None = None
    disable_model_invocation: bool = False
    user_invocable: bool = True
    disallowed_tools: list[str] | None = None
    # Issue #46: per-skill opt-in for ``!`cmd` `` execution on activation. Defaults
    # False so an untrusted SKILL.md cannot run shell on the activation path; a
    # skill must declare ``allow-shell: true`` frontmatter to opt into preprocessing.
    allow_shell: bool = False


@dataclass
class EvalScore:
    name: str
    value: float
    reason: str = ""

    def __post_init__(self):
        if not (0 <= self.value <= 1):
            raise ValueError(f"EvalScore.value must be in [0,1], got {self.value}")


@dataclass
class EvalResult:
    case_name: str
    output: str
    scores: list[EvalScore] = field(default_factory=list)
    overall_score: float = 0.0
    telemetry_report: dict | str = ""
    trace_id: str | None = None
    duration_seconds: float = 0.0
    token_usage: TokenUsage | None = None
    tool_calls_made: list[ToolCall] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    passed: bool = True
    framework: str | None = None


@dataclass
class EvalCase:
    name: str
    user_message: str
    expected_tools: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    max_iterations: int = 10
    tags: list[str] = field(default_factory=list)
    expected_answer: str | None = None
    context_docs: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    verification_fn: Callable | None = None
    tool_definitions: list[dict] = field(default_factory=list)
    expected_tool_calls: list[dict] = field(default_factory=list)
    file_attachments: list[str] = field(default_factory=list)
    # Coding-harness fields (Wave 1): all optional -- None/[] means a plain
    # (non-coding) case with zero behavior change. When `repo` is set the
    # EvalRunner materializes a per-case workspace (clone/copy + checkout +
    # setup) and TestSuiteScorer gates on `test_command`'s exit code.
    repo: str | None = None
    base_commit: str | None = None
    setup_commands: list[str] = field(default_factory=list)
    test_command: str | None = None

    def __post_init__(self):
        if self.max_iterations < 1:
            raise ValueError(f"EvalCase.max_iterations must be >= 1, got {self.max_iterations}")


@dataclass
class RunResult:
    content: str
    iterations_used: int = 0
    tool_calls_made: list[ToolCall] = field(default_factory=list)
    pipeline_outcomes: list[dict] = field(default_factory=list)
    token_usage: TokenUsage | None = None
    metadata: dict = field(default_factory=dict)
    success: bool = True
    error: AgentError | None = None
    elapsed_seconds: float = 0.0

    @property
    def tools_used(self) -> list[str]:
        """Unique tool names used during this run."""
        seen: set[str] = set()
        result: list[str] = []
        for tc in self.tool_calls_made:
            if tc.name not in seen:
                seen.add(tc.name)
                result.append(tc.name)
        return result

    @property
    def model(self) -> str:
        """Model identifier from metadata, if set."""
        return self.metadata.get("model", "")

    def __str__(self) -> str:
        return self.content
