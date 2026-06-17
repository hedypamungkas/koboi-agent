from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from koboi.context.registry import register_context_strategy
from koboi.tokens import estimate_tokens

if TYPE_CHECKING:
    from koboi.logger import AgentLogger
    from koboi.client import Client


def ensure_tool_integrity(messages: list[dict]) -> list[dict]:
    """Remove orphaned tool results and fix assistant messages with missing results.

    Also enforces API-valid message sequences:
    - No consecutive same-role messages (except system)
    - First non-system message must be 'user'
    - No assistant message with empty tool_calls
    """
    # Pass 1: collect valid tool_call IDs from assistant messages
    valid_call_ids: set[str] = set()
    for m in messages:
        if m.get("tool_calls"):
            for tc in m["tool_calls"]:
                valid_call_ids.add(tc["id"])

    # Pass 2: remove orphaned tool results (whose parent was removed)
    result = []
    for m in messages:
        if m.get("role") == "tool":
            if m.get("tool_call_id") in valid_call_ids:
                result.append(m)
        else:
            result.append(m)

    # Pass 3: check assistant messages with tool_calls have all their results
    existing_result_ids = {
        m.get("tool_call_id") for m in result if m.get("role") == "tool"
    }
    final = []
    for m in result:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            # Empty tool_calls list is invalid — strip it
            if not m["tool_calls"]:
                clean = {"role": "assistant", "content": m.get("content") or ""}
                final.append(clean)
                continue
            call_ids = {tc["id"] for tc in m["tool_calls"]}
            missing = call_ids - existing_result_ids
            if missing:
                kept_calls = [tc for tc in m["tool_calls"] if tc["id"] not in missing]
                clean = {"role": "assistant"}
                if m.get("content"):
                    clean["content"] = m["content"]
                else:
                    names = [tc.get("function", {}).get("name", "") for tc in m["tool_calls"] if tc["id"] in missing]
                    clean["content"] = f"[Called tools: {', '.join(names)}]"
                if kept_calls:
                    clean["tool_calls"] = kept_calls
                final.append(clean)
            else:
                final.append(m)
        else:
            final.append(m)

    # Pass 4: merge consecutive same-role messages (API rejects adjacent same roles)
    merged = []
    for m in final:
        if (
            merged
            and merged[-1].get("role") == m.get("role")
            and m.get("role") != "system"
        ):
            prev = merged[-1]
            prev_content = prev.get("content", "") or ""
            curr_content = m.get("content", "") or ""
            prev["content"] = (prev_content + "\n" + curr_content).strip()
            # If either has tool_calls, keep them (shouldn't happen after Pass 3)
            if m.get("tool_calls") and not prev.get("tool_calls"):
                prev["tool_calls"] = m["tool_calls"]
        else:
            merged.append(m)

    # Pass 5: ensure first non-system message is 'user', and list has at least one user
    first_non_system_seen = False
    clean = []
    for m in merged:
        if m.get("role") == "system":
            clean.append(m)
            continue
        if not first_non_system_seen:
            first_non_system_seen = True
            if m.get("role") != "user":
                clean.append({"role": "user", "content": "[continuing analysis]"})
        clean.append(m)

    # Edge case: only system messages remain — add a synthetic user to avoid empty payload
    if not first_non_system_seen:
        clean.append({"role": "user", "content": "[continuing analysis]"})

    return clean


class ContextManager(ABC):
    def __init__(self, logger: AgentLogger | None = None):
        self.logger = logger
        self.last_actual_tokens: int = 0

    def _log(self, detail: str) -> None:
        if self.logger:
            self.logger.log_context_management(detail)

    def _effective_tokens(self, messages: list[dict]) -> int:
        """Return the best token estimate: heuristic or actual LLM-reported."""
        estimated = estimate_tokens(messages)
        return max(estimated, self.last_actual_tokens)

    @property
    @abstractmethod
    def _strategy_name(self) -> str:
        """Short uppercase name for logging (e.g. 'TRUNCATION')."""
        ...

    @abstractmethod
    async def _build_result(
        self, system_msgs: list[dict], non_system: list[dict],
    ) -> tuple[list[dict], str]:
        """Strategy-specific message selection.

        Returns (result_messages, log_detail) where log_detail is a short
        description of what the strategy did (e.g. "kept last 6").
        """
        ...

    async def manage(self, messages: list[dict], max_tokens: int) -> list[dict]:
        tokens = self._effective_tokens(messages)
        if tokens <= max_tokens:
            return messages

        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        result, log_detail = await self._build_result(system_msgs, non_system)
        result = ensure_tool_integrity(result)

        self._log(
            f"{self._strategy_name}: {len(messages)} → {len(result)} messages | "
            f"{tokens} → {estimate_tokens(result)} tokens | "
            f"{log_detail}"
        )
        return result


class NoopContextManager(ContextManager):
    """Level 1-2 behavior — pass through, no management."""

    @property
    def _strategy_name(self) -> str:
        return "NOOP"

    async def _build_result(self, system_msgs, non_system):
        return system_msgs + non_system, "passthrough"

    async def manage(self, messages: list[dict], max_tokens: int) -> list[dict]:
        return messages


@register_context_strategy("truncation", description="Keep last N messages")
class TruncationManager(ContextManager):
    """Keep last N messages. System prompt always preserved."""

    def __init__(self, logger: AgentLogger | None = None, keep_last: int = 6):
        super().__init__(logger)
        self.keep_last = keep_last

    @property
    def _strategy_name(self) -> str:
        return "TRUNCATION"

    async def _build_result(self, system_msgs, non_system):
        kept = non_system[-self.keep_last:]
        return system_msgs + kept, f"kept last {self.keep_last}"


@register_context_strategy("smart_truncation", description="System prompt + first user + last N messages")
class SmartTruncationManager(ContextManager):
    """Keep: system prompt + first user message + last N messages."""

    def __init__(self, logger: AgentLogger | None = None, keep_last: int = 6):
        super().__init__(logger)
        self.keep_last = keep_last

    @property
    def _strategy_name(self) -> str:
        return "SMART_TRUNCATION"

    async def _build_result(self, system_msgs, non_system):
        first_user = None
        rest = []
        for m in non_system:
            if first_user is None and m.get("role") == "user":
                first_user = m
            else:
                rest.append(m)

        recent = rest[-self.keep_last:]
        result = list(system_msgs)
        if first_user:
            result.append(first_user)
        result.extend(recent)
        return result, f"kept system + first_user + last {self.keep_last}"


@register_context_strategy("key_facts", description="Extract tool results into compact facts")
class KeyFactsManager(ContextManager):
    """Extract tool results into a compact facts message, discard old messages."""

    def __init__(self, logger: AgentLogger | None = None, keep_last: int = 4):
        super().__init__(logger)
        self.keep_last = keep_last

    @property
    def _strategy_name(self) -> str:
        return "KEY_FACTS"

    async def _build_result(self, system_msgs, non_system):
        split = max(0, len(non_system) - self.keep_last)
        old = non_system[:split]
        recent = non_system[split:]

        facts_lines: list[str] = []
        for m in old:
            if m.get("role") == "tool" and m.get("content"):
                facts_lines.append(f"- {m['content']}")

        facts_msg = None
        if facts_lines:
            facts_content = "Previously collected data:\n" + "\n".join(facts_lines)
            facts_msg = {"role": "system", "content": facts_content}

        result = list(system_msgs)
        if facts_msg:
            result.append(facts_msg)
        result.extend(recent)
        return result, f"extracted {len(facts_lines)} tool results into facts"


@register_context_strategy("sliding_window", description="Summarize old messages via LLM + keep recent")
class SlidingWindowManager(ContextManager):
    """Summarize old messages via LLM + keep recent + system prompt."""

    def __init__(
        self,
        logger: AgentLogger | None = None,
        client: Client | None = None,
        keep_last: int = 4,
        summarization_truncation: int = 200,
    ):
        super().__init__(logger)
        self.client = client
        self.keep_last = keep_last
        self._summarization_truncation = summarization_truncation
        self._summary: str = ""

    @property
    def _strategy_name(self) -> str:
        return "SLIDING_WINDOW"

    async def _build_result(self, system_msgs, non_system):
        split = max(0, len(non_system) - self.keep_last)
        old = non_system[:split]
        recent = non_system[split:]

        summary = self._summary
        if old and self.client:
            summary = await self._summarize(old, summary)

        summary_msg = None
        if summary:
            summary_msg = {"role": "system", "content": f"Summary of previous conversation:\n{summary}"}

        result = list(system_msgs)
        if summary_msg:
            result.append(summary_msg)
        result.extend(recent)
        return result, f"summary length: {len(summary)} chars"

    async def _summarize(self, old_messages: list[dict], prev_summary: str) -> str:
        lines = []
        if prev_summary:
            lines.append(f"Previous summary: {prev_summary}")
        lines.append("New conversation:")
        for m in old_messages:
            role = m.get("role", "?")
            content = m.get("content", "")
            if content:
                lines.append(f"  {role}: {content[:self._summarization_truncation]}")
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    lines.append(f"  {role} tool_call: {fn.get('name')}({fn.get('arguments')})")

        prompt = (
            "Summarize the following conversation in 2-3 sentences. "
            "Focus on important data (numbers, facts, tool results).\n\n"
            + "\n".join(lines)
        )

        try:
            resp = await self.client.complete(
                messages=[{"role": "user", "content": prompt}],
                tools=None,
            )
            self._summary = resp.content or ""
        except Exception as exc:
            self._log(f"Summarization failed, keeping previous summary: {exc}")

        return self._summary
