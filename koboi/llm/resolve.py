"""koboi/llm/resolve.py -- Normalize an llm:/embedding:/agent-llm spec to an inline provider dict.

Supports tiered multi-provider config (backward-compatible):

- Tier 0 (today): an inline dict spec ``{provider, model, api_key, ...}`` ->
  returned as-is. Existing configs are unaffected.
- Tier 1: a **string** naming an entry in the top-level ``providers:`` map ->
  resolved to that entry's dict (e.g. ``llm: primary`` -> ``providers.primary``).
- Tier 2 (W2): ``{pool: name}`` -> raises ``NotImplementedError`` until the
  ``ProviderPool`` (failover/round_robin/budget) lands.

The resolver only NORMALIZES the spec form; the actual client construction
(``RetryClient`` for chat, ``create_client`` for embeddings) stays in the facade
/ factory, unchanged for Tier 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from koboi.config import Config


def resolve_llm_spec(spec, config: "Config") -> dict | None:
    """Resolve a provider spec to an inline dict.

    Returns ``None`` for an empty/missing spec so callers can fall back to a
    default. Raises ``ValueError`` for an unknown named reference and
    ``NotImplementedError`` for the pool form (W2).
    """
    if spec is None or spec == "":
        return None
    if isinstance(spec, str):
        providers = config.get("providers", default={}) or {}
        if spec not in providers:
            raise ValueError(
                f"Unknown provider reference {spec!r}. Define it under a top-level "
                f"`providers:` section or use an inline `llm:` dict. "
                f"Available: {sorted(providers) or '(none)'}"
            )
        return dict(providers[spec])
    if isinstance(spec, dict):
        if "pool" in spec:
            raise NotImplementedError(
                "Provider pools (policy: failover/round_robin/budget) arrive in W2. "
                "Use an inline `llm:` spec or a named `providers:` ref for now."
            )
        return dict(spec)
    raise TypeError(
        f"llm/embedding spec must be a string (named ref), a dict (inline), "
        f"or {{pool: name}}; got {type(spec).__name__}"
    )
