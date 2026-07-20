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
    existing_result_ids = {m.get("tool_call_id") for m in result if m.get("role") == "tool"}
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
    merged: list[dict] = []
    for m in final:
        if merged and merged[-1].get("role") == m.get("role") and m.get("role") != "system":
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
    clean_messages: list[dict] = []
    for m in merged:
        if m.get("role") == "system":
            clean_messages.append(m)
            continue
        if not first_non_system_seen:
            first_non_system_seen = True
            if m.get("role") != "user":
                clean_messages.append({"role": "user", "content": "[continuing analysis]"})
        clean_messages.append(m)

    # Edge case: only system messages remain — add a synthetic user to avoid empty payload
    if not first_non_system_seen:
        clean_messages.append({"role": "user", "content": "[continuing analysis]"})

    return clean_messages


def _flatten_text(content: object) -> str:
    """Flatten a message content (str or multimodal list/dict) to a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(p.get("text") or p.get("content") or "")
            else:
                parts.append(str(p))
        return " ".join(parts)
    if isinstance(content, dict):
        return content.get("text") or content.get("content") or str(content)
    return str(content)


class ContextManager(ABC):
    def __init__(self, logger: AgentLogger | None = None):
        self.logger = logger
        self.last_actual_tokens: int = 0
        # Optional real tokenizer (issue #3); set by the facade when an accurate
        # BPE counter is available (OpenAI + tiktoken). None -> chars/3 heuristic.
        self.tokenizer = None
        # Safety margin subtracted from the budget inside manage() (issue #5):
        # reserves headroom for the upcoming response so a single large reply or
        # tool result doesn't push an over-budget payload. Default 0 = old behavior.
        self.safety_margin: int = 0
        # Optional per-session metadata store (issue #4a); a SQLiteMemory (or any
        # object with get_meta/set_meta) set by the facade. Used by sliding_window
        # to persist its summary across restart/resume. None -> in-memory only.
        self.meta_store = None
        # Wave 3: the budget of the LAST manage() call -- strategies may use it
        # for a post-shrink fallback (see CodingContextManager).
        self._last_budget: int | None = None
        # Wave 3: True when the last manage() call ran the strategy (modified
        # content), even if the message COUNT stayed the same (body-only
        # eviction). loop.py ORs this into its count-based _last_compacted so
        # POST_COMPACT hooks (ReadBeforeWriteResetHook) see an honest signal.
        self.last_modified: bool = False

    def _log(self, detail: str) -> None:
        if self.logger:
            self.logger.log_context_management(detail)

    def _effective_tokens(self, messages: list[dict]) -> int:
        """Return the best token estimate: real tokenizer, else heuristic."""
        estimated = self.tokenizer(messages) if self.tokenizer else estimate_tokens(messages)
        return max(estimated, self.last_actual_tokens)

    @property
    @abstractmethod
    def _strategy_name(self) -> str:
        """Short uppercase name for logging (e.g. 'TRUNCATION')."""
        ...

    @abstractmethod
    async def _build_result(
        self,
        system_msgs: list[dict],
        non_system: list[dict],
    ) -> tuple[list[dict], str]:
        """Strategy-specific message selection.

        Returns (result_messages, log_detail) where log_detail is a short
        description of what the strategy did (e.g. "kept last 6").
        """
        ...

    async def manage(self, messages: list[dict], max_tokens: int) -> list[dict]:
        tokens = self._effective_tokens(messages)
        # Issue #5: reserve headroom for the upcoming response/tool result so a
        # single large reply can't push an over-budget payload (compaction only
        # re-runs at the next iteration's start). Applied here -- not at call
        # sites -- so the /compact force-path (max_tokens=0) still compacts fully.
        budget = max(0, max_tokens - self.safety_margin)
        self._last_budget = budget
        if tokens <= budget:
            self.last_modified = False
            return messages

        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        result, log_detail = await self._build_result(system_msgs, non_system)
        self.last_modified = True
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
        kept = non_system[-self.keep_last :]
        return system_msgs + kept, f"kept last {self.keep_last}"


@register_context_strategy("smart_truncation", description="System prompt + first user + last N messages")
class SmartTruncationManager(ContextManager):
    """Keep: system prompt + first user message + compact earlier-user notes + last N.

    The first user message is always anchored. Any *other* dropped user messages
    are folded into a compact 'Earlier user messages' note so mid-conversation
    facts are not silently lost (issue #6).
    """

    def __init__(
        self,
        logger: AgentLogger | None = None,
        keep_last: int = 6,
        summarization_truncation: int | None = 200,
    ):
        super().__init__(logger)
        self.keep_last = keep_last
        # Accept (and ignore-by-default-coerce) summarization_truncation so config
        # forwarding never TypeErrors; use it to cap earlier-user note line length.
        self._trunc = (
            summarization_truncation
            if isinstance(summarization_truncation, int) and not isinstance(summarization_truncation, bool)
            else 200
        )

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

        recent = rest[-self.keep_last :]
        dropped = rest[: max(0, len(rest) - self.keep_last)]

        # Fold dropped user messages into a compact note (issue #6).
        earlier_user_lines: list[str] = []
        for m in dropped:
            if m.get("role") == "user":
                text = _flatten_text(m.get("content"))
                if text:
                    earlier_user_lines.append(f"- {text[: self._trunc]}")

        result = list(system_msgs)
        if first_user:
            result.append(first_user)
        if earlier_user_lines:
            note = "Earlier user messages:\n" + "\n".join(earlier_user_lines)
            result.append({"role": "system", "content": note})
        result.extend(recent)
        detail = f"kept system + first_user + {len(earlier_user_lines)} earlier-user notes + last {self.keep_last}"
        return result, detail


@register_context_strategy("key_facts", description="Extract tool results into compact facts")
class KeyFactsManager(ContextManager):
    """Extract user/assistant/tool content from old messages into compact facts.

    Generalized (issue #7): previously only role=tool content was promoted; user
    and assistant content in the old section are now folded in too so they are
    not silently dropped. Defaults to no truncation (preserves prior full-content
    behavior for tool results); set ``summarization_truncation`` to cap length.
    """

    def __init__(
        self,
        logger: AgentLogger | None = None,
        keep_last: int = 4,
        summarization_truncation: int | None = None,
    ):
        super().__init__(logger)
        self.keep_last = keep_last
        self._trunc = (
            summarization_truncation
            if isinstance(summarization_truncation, int) and not isinstance(summarization_truncation, bool)
            else None
        )

    @property
    def _strategy_name(self) -> str:
        return "KEY_FACTS"

    async def _build_result(self, system_msgs, non_system):
        split = max(0, len(non_system) - self.keep_last)
        old = non_system[:split]
        recent = non_system[split:]

        facts_lines: list[str] = []
        for m in old:
            role = m.get("role")
            text = _flatten_text(m.get("content"))
            if not text:
                continue
            seg = text if not self._trunc else text[: self._trunc]
            if role == "tool":
                facts_lines.append(f"- {seg}")
            elif role == "user":
                facts_lines.append(f"- [user] {seg}")
            elif role == "assistant":
                facts_lines.append(f"- [assistant] {seg}")

        facts_msg = None
        if facts_lines:
            facts_content = "Previously collected data:\n" + "\n".join(facts_lines)
            facts_msg = {"role": "system", "content": facts_content}

        result = list(system_msgs)
        if facts_msg:
            result.append(facts_msg)
        result.extend(recent)
        return result, f"extracted {len(facts_lines)} facts from old messages"


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
        self._summary_loaded: bool = False  # issue #4a: lazy-load from meta_store

    @property
    def _strategy_name(self) -> str:
        return "SLIDING_WINDOW"

    def _ensure_summary_loaded(self) -> None:
        """Lazily hydrate the summary from the meta store (issue #4a).

        meta_store is attached by the facade after construction, so loading is
        deferred to first use. Idempotent; failures fall back to empty summary.
        """
        if self._summary_loaded:
            return
        self._summary_loaded = True
        store = getattr(self, "meta_store", None)
        if store is not None:
            try:
                loaded = store.get_meta("sliding_window_summary")
                if loaded:
                    self._summary = loaded
            except Exception as exc:  # nosec - best-effort hydration
                self._log(f"Summary load failed, starting empty: {exc}")

    def _persist_summary(self) -> None:
        """Persist the current summary so it survives restart/resume (issue #4a)."""
        store = getattr(self, "meta_store", None)
        if store is not None:
            try:
                store.set_meta("sliding_window_summary", self._summary)
            except Exception as exc:  # nosec - best-effort persist
                self._log(f"Summary persist failed: {exc}")

    async def _build_result(self, system_msgs, non_system):
        self._ensure_summary_loaded()
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
                lines.append(f"  {role}: {content[: self._summarization_truncation]}")
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    lines.append(f"  {role} tool_call: {fn.get('name')}({fn.get('arguments')})")

        prompt = (
            "Summarize the following conversation in 2-3 sentences. "
            "Focus on important data (numbers, facts, tool results).\n\n" + "\n".join(lines)
        )

        try:
            resp = await self.client.complete(
                messages=[{"role": "user", "content": prompt}],
                tools=None,
            )
            self._summary = resp.content or ""
            self._persist_summary()  # issue #4a
        except Exception as exc:
            self._log(f"Summarization failed, keeping previous summary: {exc}")

        return self._summary


# File tools whose results carry a per-path identity (from tool_call arguments).
_FILE_TOOLS = {"read_file", "write_file", "edit_file", "delete_file"}


@register_context_strategy(
    "coding",
    description="Stub old tool-result bodies; keep newest per-file read verbatim",
)
class CodingContextManager(ContextManager):
    """Tool-result body eviction for long coding sessions (Wave 3).

    Old tool-result BODIES (file reads, test output) collapse to one-line
    stubs while the newest result per (tool, path) identity stays verbatim --
    the working set survives, stale bulk goes. Unlike the other strategies it
    keeps the assistant/tool message PAIRS in place (only the tool ``content``
    shrinks), so ``ensure_tool_integrity`` never orphans anything.

    Correctness invariants:
    - stubs are NEW dicts -- ``get_messages()`` shares dict objects with
      ``ConversationMemory``; in-place mutation would destroy the full-fidelity
      stored history;
    - identity key is ``(tool_name, path)`` so a small ``edit_file``
      confirmation can never evict the newest ``read_file`` body for the path;
    - still over budget after stubbing -> plain-truncation fallback
      (system + last ``keep_last``), repaired by the integrity pass.
    """

    def __init__(
        self,
        logger: AgentLogger | None = None,
        keep_last: int = 20,
        keep_newest_per_key: int = 1,
        evict_min_chars: int = 200,
        summarization_truncation: int | None = None,  # accepted for facade compat; unused
    ):
        super().__init__(logger)
        self.keep_last = keep_last
        self.keep_newest_per_key = max(1, keep_newest_per_key)
        self.evict_min_chars = evict_min_chars

    @property
    def _strategy_name(self) -> str:
        return "CODING"

    @staticmethod
    def _id_map(non_system: list[dict]) -> dict[str, tuple[str, str | None]]:
        """tool_call_id -> (tool_name, path) from assistant tool_calls."""
        import json as _json

        id_map: dict[str, tuple[str, str | None]] = {}
        for m in non_system:
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                name = fn.get("name", "") if isinstance(fn, dict) else ""
                path = None
                if name in _FILE_TOOLS:
                    try:
                        path = _json.loads(fn.get("arguments") or "{}").get("path")
                    except (ValueError, AttributeError, TypeError):
                        path = None  # malformed args -> pathless key, never crash
                tc_id = tc.get("id") if isinstance(tc, dict) else None
                if tc_id:
                    id_map[tc_id] = (name, path)
        return id_map

    async def _build_result(self, system_msgs, non_system):
        from collections import Counter

        id_map = self._id_map(non_system)
        protected_from = max(0, len(non_system) - self.keep_last)
        seen: Counter[tuple[str, str | None]] = Counter()
        out = list(non_system)
        stubbed = 0
        saved_chars = 0

        for idx in range(len(non_system) - 1, -1, -1):
            m = non_system[idx]
            if m.get("role") != "tool":
                continue
            entry = id_map.get(m.get("tool_call_id", ""))
            if entry is None:
                continue  # orphan; ensure_tool_integrity handles it
            seen[entry] += 1
            if idx >= protected_from:
                continue  # recent window: counts toward seen, never stubbed
            if seen[entry] <= self.keep_newest_per_key:
                continue  # newest N per identity stay verbatim
            content = m.get("content")
            if not isinstance(content, str) or len(content) <= self.evict_min_chars:
                continue
            name, path = entry
            ident = f"({path!r})" if path else ""
            verb = "re-read" if name == "read_file" else "re-run"
            # NEW dict: never mutate the memory-owned message in place.
            out[idx] = {
                "role": "tool",
                "tool_call_id": m["tool_call_id"],
                "content": f"[evicted {name}{ident} result ({len(content)} chars); {verb} if needed]",
            }
            stubbed += 1
            saved_chars += len(content)

        result = system_msgs + out
        detail = f"stubbed {stubbed} tool results (~{saved_chars} chars)"
        # Post-shrink fallback: use the raw estimate, NOT _effective_tokens --
        # last_actual_tokens describes the PREVIOUS full prompt and would force
        # the fallback forever. The /compact force path (budget 0) always falls
        # back, which is the intent (force-compact must shrink hard).
        budget = self._last_budget
        if budget is not None and estimate_tokens(result) > budget:
            result = system_msgs + out[-self.keep_last :]
            detail += f"; still over budget -> truncated to last {self.keep_last}"
        return result, detail
