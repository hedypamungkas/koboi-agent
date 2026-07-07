from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from koboi.config_models import KoboiConfig


_logger = logging.getLogger(__name__)


_ENV_PATTERN = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


def _resolve_env(value: str) -> str:
    def _replace(match: re.Match) -> str:
        var = match.group(1)
        default = match.group(2)
        return os.environ.get(var, default if default is not None else match.group(0))

    return _ENV_PATTERN.sub(_replace, value)


def _walk_resolve(obj: Any) -> Any:
    if isinstance(obj, str):
        return _resolve_env(obj)
    if isinstance(obj, dict):
        return {k: _walk_resolve(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_resolve(v) for v in obj]
    return obj


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# Generation-shape keys forwarded verbatim into the provider request body when
# present under ``llm:``. Covers sampling (top_p/top_k/penalties/stop/seed),
# response shaping (response_format/logit_bias/logprobs), and reasoning budgets
# (reasoning_effort/thinking/max_completion_tokens). Kept as an allowlist so
# infra keys (provider/model/api_key/base_url/temperature/max_tokens/timeout/
# retries/auth_*) are never leaked into the body.
FORWARDABLE_LLM_KEYS: frozenset[str] = frozenset(
    {
        "top_p",
        "top_k",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "seed",
        "response_format",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "max_completion_tokens",
        "reasoning_effort",
        "thinking",
        "verbosity",
    }
)


def extract_extra_params(llm: dict) -> dict | None:
    """Pick the forward-as-is generation params out of an ``llm:`` config dict.

    Returns ``None`` when none are set so callers can skip the body merge.
    """
    picked = {k: v for k, v in llm.items() if k in FORWARDABLE_LLM_KEYS and v is not None}
    return picked or None


# All recognized ``llm:`` keys = infra/connection keys (each consumed explicitly
# by a Config accessor or RetryClient) plus the forward-as-is generation keys.
# Used to warn on typos / unrecognized keys that would otherwise be silently
# dropped (LLMConfig uses extra="ignore").
_KNOWN_LLM_KEYS: frozenset[str] = FORWARDABLE_LLM_KEYS | frozenset(
    {
        "provider",
        "model",
        "api_key",
        "base_url",
        "timeout",
        "max_tokens",
        "temperature",
        "max_retries",
        "retry_backoff_base",
        "auth_token",
        "auth_type",
        "embedding_model",
        "api_version",
        "transport_retries",
        "account_id",  # cloudflare extra_env (registry.py)
    }
)


def _load_yaml_with_extends(path: Path, _seen: set[Path] | None = None) -> dict:
    if _seen is None:
        _seen = set()
    resolved = path.resolve()
    if resolved in _seen:
        raise ValueError(f"Circular config extends detected: {resolved}")
    _seen.add(resolved)

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    extends = data.pop("extends", None)
    if not extends:
        return data

    if isinstance(extends, str):
        extends = [extends]

    base: dict = {}
    for ext_path in extends:
        ext_file = (path.parent / ext_path).resolve()
        ext_data = _load_yaml_with_extends(ext_file, _seen)
        base = _deep_merge(base, ext_data)

    return _deep_merge(base, data)


class Config:
    def __init__(self, data: dict, validate: bool = False):
        self._data = _walk_resolve(data)
        self._schema: KoboiConfig | None = None
        if validate:
            self._validate()

    def _validate(self) -> None:
        from koboi.config_models import KoboiConfig

        try:
            self._schema = KoboiConfig(**self._data)
        except Exception as exc:
            raise ValueError(f"Config validation failed: {exc}") from exc
        self._warn_unknown_llm_keys()

    def _warn_unknown_llm_keys(self) -> None:
        """Warn about ``llm:`` keys that aren't recognized (likely typos).

        Unrecognized keys are silently ignored (LLMConfig uses extra="ignore"
        and only allowlisted keys are forwarded to the provider), so surface
        them to the user at config-load time.
        """
        for key in self.llm:
            if key not in _KNOWN_LLM_KEYS:
                _logger.warning(
                    "Unknown llm: key %r is not recognized and will be ignored "
                    "(not forwarded to the provider). Possible typo?",
                    key,
                )

    @property
    def schema(self):
        """The validated Pydantic model, or None if validation was skipped."""
        return self._schema

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        data = _load_yaml_with_extends(path)
        return cls(data, validate=True)

    @classmethod
    def from_string(cls, yaml_string: str, *, validate: bool = True) -> Config:
        """Create a Config from a YAML string."""
        data = yaml.safe_load(yaml_string) or {}
        return cls(data, validate=validate)

    @classmethod
    def from_dict(cls, data: dict, *, validate: bool = True) -> Config:
        """Create a Config from a Python dict (skips YAML parsing)."""
        return cls(data, validate=validate)

    @property
    def raw(self) -> dict:
        return self._data

    def to_dict(self) -> dict:
        """Return config as a plain dict. Uses validated schema if available."""
        if self._schema is not None:
            return self._schema.model_dump()
        return dict(self._data)

    def get(self, *keys: str, default: Any = None) -> Any:
        node = self._data
        for key in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(key)
            if node is None:
                return default
        return node

    # --- Convenience accessors ---

    @property
    def agent(self) -> dict:
        return self.get("agent", default={})

    @property
    def llm(self) -> dict:
        return self.get("llm", default={})

    @property
    def tools(self) -> dict:
        return self.get("tools", default={})

    @property
    def context(self) -> dict:
        return self.get("context", default={})

    @property
    def rag(self) -> dict:
        return self.get("rag", default={})

    @property
    def guardrails(self) -> dict:
        return self.get("guardrails", default={})

    @property
    def policy(self) -> dict:
        return self.get("policy", default={})

    @property
    def skills(self) -> dict:
        return self.get("skills", default={})

    @property
    def mcp(self) -> dict:
        return self.get("mcp", default={})

    @property
    def tracing(self) -> dict:
        return self.get("tracing", default={})

    @property
    def harness(self) -> dict:
        return self.get("harness", default={})

    @property
    def eval(self) -> dict:
        return self.get("eval", default={})

    @property
    def subagent(self) -> dict:
        return self.get("subagent", default={})

    @property
    def sandbox(self) -> dict:
        return self.get("sandbox", default={})

    @property
    def journal(self) -> dict:
        return self.get("journal", default={})

    @property
    def server(self) -> dict:
        return self.get("server", default={})

    @property
    def jobs(self) -> dict:
        return self.get("jobs", default={})

    @property
    def agent_name(self) -> str:
        return self.agent.get("name", "koboi-agent")

    @property
    def theme(self) -> str:
        return self.agent.get("theme", "koboi-dark")

    @property
    def system_prompt(self) -> str:
        return self.agent.get("system_prompt", "")

    @property
    def max_iterations(self) -> int:
        return self.agent.get("max_iterations", 10)

    @property
    def provider(self) -> str:
        return self.llm.get("provider", "openai")

    @property
    def model(self) -> str:
        return self.llm.get("model", "gpt-4o-mini")

    @property
    def api_key(self) -> str:
        return self.llm.get("api_key", "")

    @property
    def base_url(self) -> str:
        return self.llm.get("base_url", "")

    @property
    def llm_timeout(self) -> float:
        return self.llm.get("timeout", 120.0)

    @property
    def llm_max_tokens(self) -> int | None:
        # None = "user did not configure it": OpenAI/Cloudflare then omit
        # max_tokens (no force-cap at a default); Anthropic supplies its own
        # 4096 fallback (its API requires the field).
        return self.llm.get("max_tokens", None)

    @property
    def llm_auth_token(self) -> str:
        return self.llm.get("auth_token", "")

    @property
    def auth_type(self) -> str:
        return self.llm.get("auth_type", "api_key")

    @property
    def temperature(self) -> float | None:
        return self.llm.get("temperature", None)

    @property
    def llm_extra_params(self) -> dict:
        """Forward-as-is generation params (sampling + reasoning) under ``llm:``.

        These reach the provider HTTP body verbatim (top_p, stop, seed,
        response_format, reasoning_effort, thinking, ...). Empty when unset.
        """
        return extract_extra_params(self.llm) or {}

    @property
    def max_retries(self) -> int:
        return self.llm.get("max_retries", 3)

    @property
    def retry_backoff_base(self) -> float:
        return self.llm.get("retry_backoff_base", 2.0)

    @property
    def transport_retries(self) -> int:
        return self.llm.get("transport_retries", 2)

    @property
    def embedding_model(self) -> str:
        return self.llm.get("embedding_model", "text-embedding-3-small")

    @property
    def api_version(self) -> str:
        return self.llm.get("api_version", "2023-06-01")

    @property
    def rag_enabled(self) -> bool:
        return self.rag.get("enabled", False)

    @property
    def mode(self) -> str:
        """Default interaction mode: chat, plan, act, auto."""
        return self.agent.get("mode", "chat")

    @property
    def graduated_permissions(self) -> bool:
        """Whether to use graduated trust for auto-approval."""
        return self.get("guardrails", "approval", "graduated", default=False)

    @property
    def trust_db_path(self) -> str:
        """Path to the SQLite trust database."""
        return self.get("guardrails", "approval", "trust_db_path", default="koboi_trust.db")

    @property
    def orchestration(self) -> dict:
        return self.get("orchestration", default={})

    @property
    def keybindings(self) -> dict:
        return self.get("keybindings", default={})

    @classmethod
    def builder(cls) -> ConfigBuilder:
        """Return a new ConfigBuilder for programmatic config creation."""
        return ConfigBuilder()


class ConfigBuilder:
    """Fluent builder for creating Config without YAML files.

    Usage:
        config = (
            Config.builder()
            .agent(name="my-agent", system_prompt="You are helpful")
            .llm(provider="openai", model="gpt-4o", api_key="sk-...")
            .tools(builtin=["calculator", "web_search"])
            .build()
        )
    """

    def __init__(self):
        self._data: dict[str, Any] = {}

    def agent(
        self,
        name: str | None = None,
        system_prompt: str | None = None,
        max_iterations: int | None = None,
        description: str | None = None,
    ) -> ConfigBuilder:
        section = self._data.setdefault("agent", {})
        if name is not None:
            section["name"] = name
        if system_prompt is not None:
            section["system_prompt"] = system_prompt
        if max_iterations is not None:
            section["max_iterations"] = max_iterations
        if description is not None:
            section["description"] = description
        return self

    def llm(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        retry_backoff_base: float | None = None,
        auth_token: str | None = None,
        auth_type: str | None = None,
        embedding_model: str | None = None,
    ) -> ConfigBuilder:
        section = self._data.setdefault("llm", {})
        for key, val in {
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "max_retries": max_retries,
            "retry_backoff_base": retry_backoff_base,
            "auth_token": auth_token,
            "auth_type": auth_type,
            "embedding_model": embedding_model,
        }.items():
            if val is not None:
                section[key] = val
        return self

    def tools(
        self,
        builtin: list[str] | None = None,
        custom: list[dict] | None = None,
        defaults: dict | None = None,
        overrides: dict | None = None,
    ) -> ConfigBuilder:
        section = self._data.setdefault("tools", {})
        if builtin is not None:
            section["builtin"] = builtin
        if custom is not None:
            section["custom"] = custom
        if defaults is not None:
            section["defaults"] = defaults
        if overrides is not None:
            section["overrides"] = overrides
        return self

    def context(
        self,
        strategy: str | None = None,
        max_context_tokens: int | None = None,
        keep_last: int | None = None,
    ) -> ConfigBuilder:
        section = self._data.setdefault("context", {})
        if strategy is not None:
            section["strategy"] = strategy
        if max_context_tokens is not None:
            section["max_context_tokens"] = max_context_tokens
        if keep_last is not None:
            section["keep_last"] = keep_last
        return self

    def rag(
        self,
        enabled: bool = True,
        chunker: str | None = None,
        chunk_size: int | None = None,
        retriever: str | None = None,
        top_k: int | None = None,
        augmentation: str | None = None,
        documents: list[str | dict] | None = None,
    ) -> ConfigBuilder:
        section = self._data.setdefault("rag", {})
        section["enabled"] = enabled
        if chunker is not None:
            section["chunker"] = chunker
        if chunk_size is not None:
            section["chunk_size"] = chunk_size
        if retriever is not None:
            section["retriever"] = retriever
        if top_k is not None:
            section["top_k"] = top_k
        if augmentation is not None:
            section["augmentation"] = augmentation
        if documents is not None:
            section["documents"] = [d if isinstance(d, dict) else {"path": d} for d in documents]
        return self

    def guardrails(
        self,
        input: dict | None = None,
        output: dict | None = None,
        rate_limit: dict | None = None,
        approval: dict | None = None,
    ) -> ConfigBuilder:
        section = self._data.setdefault("guardrails", {})
        if input is not None:
            section["input"] = input
        if output is not None:
            section["output"] = output
        if rate_limit is not None:
            section["rate_limit"] = rate_limit
        if approval is not None:
            section["approval"] = approval
        return self

    def memory(
        self,
        backend: str | None = None,
        db_path: str | None = None,
        session_id: str | None = None,
    ) -> ConfigBuilder:
        section = self._data.setdefault("memory", {})
        if backend is not None:
            section["backend"] = backend
        if db_path is not None:
            section["db_path"] = db_path
        if session_id is not None:
            section["session_id"] = session_id
        return self

    def harness(
        self,
        telemetry: bool | None = None,
        carryover: bool | None = None,
        doom_loop: dict | None = None,
    ) -> ConfigBuilder:
        section = self._data.setdefault("harness", {})
        if telemetry is not None:
            section["telemetry"] = telemetry
        if carryover is not None:
            section["carryover"] = carryover
        if doom_loop is not None:
            section["doom_loop"] = doom_loop
        return self

    def tracing(
        self,
        provider: str | None = None,
        public_key: str | None = None,
        secret_key: str | None = None,
        base_url: str | None = None,
    ) -> ConfigBuilder:
        section = self._data.setdefault("tracing", {})
        if provider is not None:
            section["provider"] = provider
        if public_key is not None:
            section["public_key"] = public_key
        if secret_key is not None:
            section["secret_key"] = secret_key
        if base_url is not None:
            section["base_url"] = base_url
        return self

    def policy(self, rules: list[dict] | None = None) -> ConfigBuilder:
        if rules is not None:
            self._data.setdefault("policy", {})["rules"] = rules
        return self

    def skills(self, search_paths: list[str] | None = None, budget_chars: int | None = None) -> ConfigBuilder:
        if search_paths is not None:
            self._data.setdefault("skills", {})["search_paths"] = search_paths
        if budget_chars is not None:
            self._data.setdefault("skills", {})["budget_chars"] = budget_chars
        return self

    def mcp(self, servers: list[dict] | None = None) -> ConfigBuilder:
        if servers is not None:
            self._data.setdefault("mcp", {})["servers"] = servers
        return self

    def orchestration(
        self,
        enabled: bool = True,
        router_type: str | None = None,
        execution_mode: str | None = None,
        agents: list[dict] | None = None,
    ) -> ConfigBuilder:
        section = self._data.setdefault("orchestration", {})
        section["enabled"] = enabled
        if router_type is not None:
            section.setdefault("router", {})["type"] = router_type
        if execution_mode is not None:
            section.setdefault("execution", {})["mode"] = execution_mode
        if agents is not None:
            section["agents"] = agents
        return self

    def sandbox(
        self,
        *,
        backend: str | None = None,
        workdir: str | None = None,
        network: str | None = None,
        network_binaries: list[str] | None = None,
        safe_path: list[str] | None = None,
        env_passthrough: bool | None = None,
        rlimits: dict | None = None,
        timeout: float | None = None,
        max_output: int | None = None,
    ) -> ConfigBuilder:
        section = self._data.setdefault("sandbox", {})
        for key, val in {
            "backend": backend,
            "workdir": workdir,
            "network": network,
            "network_binaries": network_binaries,
            "safe_path": safe_path,
            "env_passthrough": env_passthrough,
            "rlimits": rlimits,
            "timeout": timeout,
            "max_output": max_output,
        }.items():
            if val is not None:
                section[key] = val
        return self

    def journal(
        self,
        *,
        enabled: bool | None = None,
        record_tool_calls: bool | None = None,
    ) -> ConfigBuilder:
        section = self._data.setdefault("journal", {})
        if enabled is not None:
            section["enabled"] = enabled
        if record_tool_calls is not None:
            section["record_tool_calls"] = record_tool_calls
        return self

    def server(
        self,
        *,
        enabled: bool | None = None,
        host: str | None = None,
        port: int | None = None,
        api_keys_file: str | None = None,
        api_keys: list[str] | None = None,
        auth_required: bool | None = None,
        cors: dict | None = None,
        pool: dict | None = None,
        timeouts: dict | None = None,
        limits: dict | None = None,
        idempotency: dict | None = None,
        workdir_ttl_seconds: float | None = None,
    ) -> ConfigBuilder:
        section = self._data.setdefault("server", {})
        for key, val in {
            "enabled": enabled,
            "host": host,
            "port": port,
            "api_keys_file": api_keys_file,
            "api_keys": api_keys,
            "auth_required": auth_required,
            "cors": cors,
            "pool": pool,
            "timeouts": timeouts,
            "limits": limits,
            "idempotency": idempotency,
            "workdir_ttl_seconds": workdir_ttl_seconds,
        }.items():
            if val is not None:
                section[key] = val
        return self

    def jobs(
        self,
        *,
        enabled: bool | None = None,
        max_concurrent: int | None = None,
        per_tenant_max: int | None = None,
        queue_depth: int | None = None,
        default_dedicated_session: bool | None = None,
        event_buffer: dict | None = None,
        resume_on_startup: bool | None = None,
        timeout_seconds: float | None = None,
        ttl_seconds: float | None = None,
    ) -> ConfigBuilder:
        section = self._data.setdefault("jobs", {})
        for key, val in {
            "enabled": enabled,
            "max_concurrent": max_concurrent,
            "per_tenant_max": per_tenant_max,
            "queue_depth": queue_depth,
            "default_dedicated_session": default_dedicated_session,
            "event_buffer": event_buffer,
            "resume_on_startup": resume_on_startup,
            "timeout_seconds": timeout_seconds,
            "ttl_seconds": ttl_seconds,
        }.items():
            if val is not None:
                section[key] = val
        return self

    def build(self) -> Config:
        return Config(self._data, validate=True)
