"""koboi/server/handoff_digest.py -- warm handoff digest (Wave 3 B4).

At handover time (B1/B1.5), generate a concise REDACTED case-card summary of the
conversation so the human operator sees "customer wants X, bot tried Y" instead of
scrolling a raw transcript (B2 replay). The digest lands in ``HandoverEvent.summary``
→ B2 ``SessionEventRegistry`` → ``GET /v1/sessions/{id}/stream`` replay → operator.

Mirrors ``ProactiveMemory.extract_and_store`` (one side-LLM call over the transcript,
``tools=None``, never-raises) + ``redact_value`` (scrub secret shapes). Opt-in
(``handover.digest.enabled``); default off (no latency cost). Synchronous at the
handover site (the session lock is already released -- no deadlock); the one-LLM
round-trip delays the handover signal, acceptable since the operator wasn't driving.

NEVER RAISES -- a digest failure must NOT turn the ``HandoverEvent`` into an
``ErrorEvent`` (the handover would be lost). The call site double-wraps too.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from koboi.redact import redact_value

if TYPE_CHECKING:
    from koboi.llm.base import LLMClient

_logger = logging.getLogger(__name__)

_DIGEST_PROMPT = (
    "Summarize this customer-service conversation in 2-3 sentences for a human "
    "operator taking over. State what the customer wants and what was already "
    "attempted. Do not include secrets, credentials, or PII. Reply with only the "
    "summary, no preamble.\n\nConversation:\n{convo}"
)


class HandoffDigest:
    """Warm handoff digest (B4) -- redacted side-LLM case card. Never raises."""

    def __init__(
        self,
        *,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: str = "",
        base_url: str = "",
        timeout: float = 60.0,
        max_chars: int = 4000,
        **_unused: object,
    ) -> None:
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._max_chars = max_chars
        self._client: LLMClient | None = None  # lazily built

    def _get_client(self) -> LLMClient | None:
        if self._client is None:
            try:
                from koboi.llm.factory import create_client

                self._client = create_client(
                    provider=self._provider,
                    model=self._model,
                    api_key=self._api_key,
                    base_url=self._base_url,
                    timeout=self._timeout,
                )
            except Exception as exc:  # nosec - fail-soft
                _logger.warning("HandoffDigest client build failed: %s", exc)
                self._client = None
        return self._client

    @staticmethod
    def _flatten(messages: list[dict], max_chars: int) -> str:
        parts: list[str] = []
        total = 0
        for m in messages:
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, list):  # multimodal parts
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            line = f"{role}: {content}"
            if total + len(line) > max_chars:
                remaining = max_chars - total
                if remaining > 0:
                    parts.append(line[:remaining])
                break
            parts.append(line)
            total += len(line) + 1
        return "\n".join(parts).strip()

    async def digest(self, messages: list[dict]) -> str:
        """Return a redacted case-card summary (empty on any failure). Never raises."""
        client = self._get_client()
        if client is None:
            return ""
        try:
            convo = self._flatten(messages, self._max_chars)
            if not convo:
                return ""
            resp = await client.complete(
                messages=[{"role": "user", "content": _DIGEST_PROMPT.format(convo=convo)}],
                tools=None,
            )
            text = (getattr(resp, "content", None) or "").strip()
            if not text:
                return ""
            return redact_value(text)[:1000]  # scrub secret shapes + bound length
        except Exception as exc:  # nosec - best-effort, never break the handover
            _logger.warning("Handoff digest failed: %s", exc)
            return ""
