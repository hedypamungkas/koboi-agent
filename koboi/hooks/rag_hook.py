"""koboi/hooks/rag_hook.py -- Hook for RAG augmentation at POST_COMPACT.

Intercepts POST_COMPACT to set up the augmentation strategy before the LLM call.
Injects retrieved context into the message stream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from koboi.hooks.chain import Hook, HookContext, HookEvent

if TYPE_CHECKING:
    pass


class RAGHook(Hook):
    """Hook that handles RAG augmentation at POST_COMPACT (before LLM call).

    Sets up the augmentation strategy by injecting retrieved context into
    the hook context so downstream consumers can use it.
    """

    def __init__(
        self,
        strategy: str = "prepend",
        max_context_tokens: int = 2000,
    ):
        self.strategy = strategy
        self.max_context_tokens = max_context_tokens

    def handles(self) -> list[HookEvent]:
        return [HookEvent.POST_COMPACT]

    async def execute(self, ctx: HookContext) -> HookContext:
        # Store augmentation config in metadata for downstream use
        ctx.metadata["rag_strategy"] = self.strategy
        ctx.metadata["rag_max_context_tokens"] = self.max_context_tokens

        # If messages are available, check for pending RAG context
        if ctx.messages:
            # Look for RAG results in carryover or metadata
            rag_results = ctx.metadata.get("rag_results", [])
            if rag_results and self.strategy == "prepend":
                # Build augmentation text from results
                chunks_text = "\n".join(f"[{i + 1}] {r}" for i, r in enumerate(rag_results))
                augmentation = f"<retrieved-context>\n{chunks_text}\n</retrieved-context>"
                ctx.metadata["rag_augmentation"] = augmentation

        return ctx
