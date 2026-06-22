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
    max_tokens: int = Field(default=4096, ge=1)
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


class ToolsConfig(BaseModel):
    model_config = {"extra": "ignore"}

    builtin: list[str] = Field(default_factory=list)
    custom: list[dict] = Field(default_factory=list)
    defaults: dict = Field(default_factory=dict)
    overrides: dict = Field(default_factory=dict)


class ContextConfig(BaseModel):
    model_config = {"extra": "ignore"}

    strategy: str = "noop"
    max_context_tokens: int = Field(default=8000, ge=1)
    keep_last: int | None = None
    summarization_truncation: bool | int | None = None
    custom_modules: list[str] = Field(default_factory=list)


class RagConfig(BaseModel):
    model_config = {"extra": "ignore"}

    enabled: bool = False
    chunker: str = "sentence"
    chunk_size: int = Field(default=500, ge=1)
    retriever: str = "keyword"
    top_k: int = Field(default=3, ge=1)
    augmentation: str = "on_the_fly"
    documents: list[str | dict] = Field(default_factory=list)


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


class PolicyConfig(BaseModel):
    model_config = {"extra": "ignore"}

    rules: list[PolicyRuleConfig] = Field(default_factory=list)


class MemoryConfig(BaseModel):
    model_config = {"extra": "ignore"}

    backend: str = "sqlite"
    db_path: str = "koboi_memory.db"
    session_id: str | None = None


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


class MCPConfig(BaseModel):
    model_config = {"extra": "ignore"}

    servers: list[MCPServerConfig] = Field(default_factory=list)


class KoboiConfig(BaseModel):
    """Top-level config schema for koboi-agent."""

    model_config = {"extra": "ignore"}

    agent: AgentConfig = Field(default_factory=AgentConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
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
        if llm is not None and isinstance(llm, dict) and not llm.get("model"):
            errors.append("llm.model is required")
        elif llm is None:
            errors.append("llm.model is required")
        if errors:
            raise ValueError("; ".join(errors))
        return data
