"""koboi/proactive_memory.py -- Proactive long-term memory (extract D + recall C + core-block B).

Closes the proactivity asymmetry: RAG/skills/conversation-history are all
auto-injected every turn, but the KV long-term memory layer was on-demand only
(the LLM had to choose to call ``memory_recall``). This module makes long-term
memory PROACTIVE, mirroring how the rest of the industry does it (Letta core
blocks, LangGraph recall-nodes, CrewAI before-each-task injection, ChatGPT):

* **D — extract** (write side): after each run, an LLM pulls durable facts from
  the conversation, redacts them, and stores them in the KV store (and maintains
  the core block).
* **C — recall** (read side): each turn, embed the user message, cosine-rank the
  stored facts, and return the top-N for ephemeral injection into the prompt.
* **B — core block** (read side): a small, bounded always-in-context summary.

All opt-in via ``memory.proactive`` config; inert by default.

Reuse (no new infrastructure): ``client.complete()`` for the extraction side-LLM
call (like ``SlidingWindowManager._summarize``), ``client.get_embeddings()`` +
``SemanticRetriever._cosine_similarity`` for recall (NOT the corpus-coupled
``SemanticRetriever``), ``SQLiteMemory.get_meta/set_meta`` for the core block,
``_MemoryStore.store`` for facts, and ``koboi.redact.redact_value`` for secrets.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from koboi.rag.retriever import SemanticRetriever
from koboi.redact import redact_tool_arguments

if TYPE_CHECKING:
    from koboi.client import Client
    from koboi.memory import ConversationMemory
    from koboi.tools.builtin.memory import _MemoryStore

_logger = logging.getLogger(__name__)

_CORE_META_KEY = "core_memory"  # session_meta key (non-repo-scoped path, unchanged)
# KV-store reserved key (repo_scoped path) -- double-underscore so it can never
# collide with an LLM-generated snake_case fact key.
_CORE_STORE_KEY = "__core_memory__"
_CORE_MAX_CHARS = 2000


class ProactiveMemory:
    """Coordinator for the proactive long-term-memory features (D/C/B).

    Constructed by the facade (``AgentAssembler.build_proactive_memory``) only
    when ``memory.proactive.enabled`` is true, and passed to the extraction hook
    (SESSION_END) and to ``AgentCore`` (for per-turn recall/core-block injection).
    """

    def __init__(
        self,
        *,
        client: Client,
        embedding_client: Client | None,
        memory: ConversationMemory,
        store: _MemoryStore,
        config: dict[str, Any] | None = None,
        repo_scoped: bool = False,
    ) -> None:
        self._client = client  # chat client -> complete() for extraction
        self._embedding_client = embedding_client or client  # -> get_embeddings() for recall
        self._memory = memory  # SQLiteMemory (session_meta + get_messages)
        self._store = store  # _MemoryStore (KV facts)
        cfg = config or {}
        self._extract = bool(cfg.get("extract", False))
        self._recall = bool(cfg.get("recall", False))
        self._core_block = bool(cfg.get("core_block", False))
        self._top_k = int(cfg.get("top_k", 4))
        self._min_score = float(cfg.get("min_score", 0.0))
        self._max_facts = int(cfg.get("max_facts", 200))
        # Wave 4: when True, the core-memory block lives in ``self._store`` under a
        # reserved key (survives across sessions -- the file is anchored to the repo
        # workdir) instead of session-scoped ``session_meta``.
        self._repo_scoped = repo_scoped
        # In-process state (rebuilt lazily per process; KV is small).
        self._embeddings: dict[str, list[float]] = {}  # KV key -> vector
        self._recall_cache: dict[str, str | None] = {}  # query hash -> result

    # -- feature gates -------------------------------------------------------

    @property
    def extract_enabled(self) -> bool:
        return self._extract

    @property
    def recall_enabled(self) -> bool:
        return self._recall

    @property
    def core_block_enabled(self) -> bool:
        return self._core_block

    # -- D: extraction -------------------------------------------------------

    async def extract_and_store(self) -> int:
        """Extract durable facts from the conversation via a side-LLM call, redact, store.

        Runs at SESSION_END. Returns the number of facts stored. Never raises
        (extraction is best-effort and must not break the run).
        """
        if self._client is None:
            return 0
        messages = self._memory.get_messages()
        if len(messages) < 2:
            return 0
        convo = self._format_conversation(messages)
        if not convo.strip():
            return 0
        prompt = (
            "Extract durable, reusable facts about the user and their stable preferences from "
            "the conversation below. Return ONLY a JSON object mapping a short snake_case key to "
            'a concise value, e.g. {"preferred_language": "python", "timezone": "UTC"}. '
            "Skip ephemeral/transactional content (current task steps, one-off questions, raw "
            "tool outputs, the agent's own reasoning). If there is nothing durable, return {}.\n\n"
            "Conversation:\n" + convo
        )
        try:
            resp = await self._client.complete(messages=[{"role": "user", "content": prompt}], tools=None)
        except Exception as exc:  # nosec - best-effort extraction
            _logger.warning("Proactive memory extraction call failed: %s", exc)
            return 0
        facts = self._parse_facts(getattr(resp, "content", None))
        # Redact by key name + value shape (consolidated), then drop any fact that
        # redacts away entirely so secrets never reach the KV store / core block.
        facts = self._parse_facts(redact_tool_arguments(json.dumps(facts)))
        clean = {str(k)[:256]: str(v)[:50000] for k, v in facts.items() if v and v != "***REDACTED***"}
        # Persist each fact; track which actually landed so the reported count is
        # honest and the core block never advertises facts the KV store rejected
        # (which would desync the two stores across a restart). Detect failure by
        # the store's "Error:" prefix (stable across success-message rewording) so
        # a prose change can't silently disable the feature. Wrapped to honor the
        # "never raises" contract at SESSION_END.
        persisted: dict[str, str] = {}
        try:
            for key, value in clean.items():
                result = self._store.store(key, value)
                if str(result).startswith("Error"):
                    _logger.warning("Proactive memory: KV store rejected fact '%s': %s", key, result)
                else:
                    persisted[key] = value
        except Exception as exc:  # nosec - never break the run at SESSION_END
            _logger.warning("Proactive memory: KV store loop failed: %s", exc)
        # B: maintain the always-in-context core-memory block from persisted facts.
        if self._core_block and persisted:
            self._merge_core_block(persisted)
        count = len(persisted)
        if count:
            _logger.debug("Proactive memory: stored %d fact(s)", count)
            # Invalidate caches so the next recall sees the new facts.
            self._embeddings.clear()
            self._recall_cache.clear()
        return count

    @staticmethod
    def _format_conversation(messages: list[dict], max_chars: int = 4000) -> str:
        """Flatten messages to 'role: content' text, bounded to max_chars."""
        lines: list[str] = []
        total = 0
        for m in messages:
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
            if not content:
                continue
            line = f"{role}: {content}"
            if total + len(line) > max_chars:
                lines.append(line[: max(0, max_chars - total)])
                break
            lines.append(line)
            total += len(line)
        return "\n".join(lines)

    @staticmethod
    def _parse_facts(content: str | None) -> dict[str, str]:
        """Parse a JSON object of facts from an LLM response (tolerates fences/prose)."""
        if not content:
            return {}
        text = content.strip()
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence:
            text = fence.group(1)
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return {}
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items() if v not in (None, "")}
        return {}

    # -- C: recall -----------------------------------------------------------

    async def recall(self, query: str) -> str | None:
        """Return a formatted top-N facts block relevant to ``query``, or None.

        Embeds the query once, cosine-ranks the embedded KV facts (reusing
        ``SemanticRetriever._cosine_similarity`` — NOT the corpus-coupled
        retriever), and returns the top-K above ``min_score``. Results (including
        no-match ``None``) are cached per query, so repeated iterations within a
        run never re-embed -- including turns that match nothing.
        """
        if not query or self._embedding_client is None:
            return None
        if query in self._recall_cache:
            return self._recall_cache[query]  # cached (str, or None for a no-match query)
        try:
            await self._ensure_embeddings()
            qvec = await self._embedding_client.get_embeddings(query)
        except Exception as exc:  # nosec - best-effort recall
            _logger.warning("Proactive recall embedding failed: %s", exc)
            return None
        if not qvec:
            return None
        data = self._store._data
        scored: list[tuple[float, str]] = []
        for key, vec in self._embeddings.items():
            score = SemanticRetriever._cosine_similarity(qvec, vec)
            if score >= self._min_score:
                scored.append((score, key))
        scored.sort(reverse=True)
        top = scored[: self._top_k]
        if not top:
            self._recall_cache[query] = None
            return None
        lines = [f"- {k}: {data.get(k, '')}" for _, k in top]
        block = "Relevant long-term memory:\n" + "\n".join(lines)
        self._recall_cache[query] = block
        return block

    async def _ensure_embeddings(self) -> None:
        """Embed any KV facts not yet in the in-process map (capped to max_facts).

        Double-underscore keys are reserved (e.g. the repo-scoped core-memory
        block) and must never be embedded/surfaced as if they were a recalled fact.
        """
        if self._embedding_client is None:
            return
        data = self._store._data
        keys = [k for k in data if not k.startswith("__")][: self._max_facts]
        missing = [k for k in keys if k not in self._embeddings]
        for k in missing:
            try:
                vec = await self._embedding_client.get_embeddings(f"{k}: {data[k]}")
            except Exception:  # nosec - skip a single failed embedding
                vec = None
            if vec:
                self._embeddings[k] = vec

    # -- B: core-memory block ------------------------------------------------

    def get_core_block(self) -> str | None:
        """Render the always-in-context core-memory block, or None if empty.

        Repo-scoped (``self._repo_scoped``): read from ``self._store`` under the
        reserved key -- survives across sessions since that file is anchored to
        the repo workdir, not the session id. Otherwise: JSON ``{key: value}`` map
        in ``session_meta`` (SQLiteMemory), unchanged from pre-Wave-4 behavior.
        No-op (returns None) when empty/unavailable/corrupt.
        """
        try:
            raw = self._read_core_raw()
        except Exception:  # nosec - best-effort read, display only
            return None
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(data, dict) or not data:
            return None
        return "Core memory:\n" + "\n".join(f"- {k}: {v}" for k, v in data.items())

    def _read_core_raw(self) -> str | None:
        """Read the raw core-block JSON string, or None if absent.

        Does NOT swallow errors -- callers decide (``get_core_block`` treats a
        read error as "nothing to show"; ``_merge_core_block`` must instead SKIP
        the merge entirely on a read error, never treat it as "empty").
        """
        if self._repo_scoped:
            return self._store._data.get(_CORE_STORE_KEY)
        get_meta = getattr(self._memory, "get_meta", None)
        if get_meta is None:
            return None
        return get_meta(_CORE_META_KEY)

    def _merge_core_block(self, new_facts: dict[str, str]) -> None:
        """Merge newly extracted facts into the core block (dedup by key, bounded).

        Never wipes the existing block on a read/parse error -- a single corrupt
        stored value must not destroy all accumulated facts. On error it logs and
        skips the merge (existing block left untouched).
        """
        if not self._repo_scoped:
            set_meta = getattr(self._memory, "set_meta", None)
            if set_meta is None:
                return
        existing: dict[str, str] = {}
        try:
            raw = self._read_core_raw()
        except Exception as exc:  # nosec - keep existing on read error
            _logger.warning("Proactive core block read failed; merge skipped: %s", exc)
            return
        if raw:
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                _logger.warning("Proactive core block corrupt; keeping existing, merge skipped: %s", exc)
                return
            if isinstance(parsed, dict):
                existing = {str(k): str(v) for k, v in parsed.items()}
        existing.update(new_facts)
        items = list(existing.items())[-self._max_facts :]
        rendered = json.dumps(dict(items))
        while items and len(rendered) > _CORE_MAX_CHARS:
            items.pop(0)
            rendered = json.dumps(dict(items))
        try:
            if self._repo_scoped:
                self._store.store(_CORE_STORE_KEY, rendered)
            else:
                self._memory.set_meta(_CORE_META_KEY, rendered)
        except Exception as exc:  # nosec - best-effort persist
            _logger.warning("Proactive core block persist failed: %s", exc)
