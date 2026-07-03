"""Tests for koboi/hooks/rag_hook.py — RAGHook (0% → >85%)."""

from __future__ import annotations


from koboi.hooks.chain import HookContext, HookEvent
from koboi.hooks.rag_hook import RAGHook


class TestRAGHook:
    def test_handles_returns_post_compact(self):
        """RAGHook should handle POST_COMPACT event."""
        hook = RAGHook()
        assert hook.handles() == [HookEvent.POST_COMPACT]

    async def test_default_initialization(self):
        """RAGHook should initialize with default values."""
        hook = RAGHook()
        assert hook.strategy == "prepend"
        assert hook.max_context_tokens == 2000

    async def test_custom_initialization(self):
        """RAGHook should accept custom strategy and max_context_tokens."""
        hook = RAGHook(strategy="append", max_context_tokens=5000)
        assert hook.strategy == "append"
        assert hook.max_context_tokens == 5000

    async def test_metadata_stores_rag_strategy(self):
        """Should store rag_strategy in metadata."""
        hook = RAGHook(strategy="prepend")
        ctx = HookContext(event=HookEvent.POST_COMPACT)
        result = await hook.execute(ctx)
        assert result.metadata["rag_strategy"] == "prepend"

    async def test_metadata_stores_max_context_tokens(self):
        """Should store rag_max_context_tokens in metadata."""
        hook = RAGHook(max_context_tokens=3000)
        ctx = HookContext(event=HookEvent.POST_COMPACT)
        result = await hook.execute(ctx)
        assert result.metadata["rag_max_context_tokens"] == 3000

    async def test_empty_messages_passthrough(self):
        """Should passthrough when messages is None or empty."""
        hook = RAGHook()
        ctx = HookContext(event=HookEvent.POST_COMPACT, messages=None)
        result = await hook.execute(ctx)
        assert result is ctx
        assert "rag_augmentation" not in result.metadata

    async def test_empty_messages_list_passthrough(self):
        """Should passthrough when messages is an empty list."""
        hook = RAGHook()
        ctx = HookContext(event=HookEvent.POST_COMPACT, messages=[])
        result = await hook.execute(ctx)
        assert result is ctx
        assert "rag_augmentation" not in result.metadata

    async def test_prepends_strategy_builds_augmentation_from_rag_results(self):
        """Prepend strategy should build augmentation text from rag_results."""
        hook = RAGHook(strategy="prepend")
        rag_results = ["Context chunk 1", "Context chunk 2", "Context chunk 3"]
        ctx = HookContext(
            event=HookEvent.POST_COMPACT,
            messages=[{"role": "user", "content": "Query"}],
        )
        ctx.metadata["rag_results"] = rag_results
        result = await hook.execute(ctx)
        assert "rag_augmentation" in result.metadata
        augmentation = result.metadata["rag_augmentation"]
        assert "<retrieved-context>" in augmentation
        assert "[1] Context chunk 1" in augmentation
        assert "[2] Context chunk 2" in augmentation
        assert "[3] Context chunk 3" in augmentation
        assert "</retrieved-context>" in augmentation

    async def test_prepend_ignores_non_prepend_strategy(self):
        """Non-prepend strategies should not build augmentation."""
        hook = RAGHook(strategy="append")
        rag_results = ["Context chunk 1"]
        ctx = HookContext(
            event=HookEvent.POST_COMPACT,
            messages=[{"role": "user", "content": "Query"}],
        )
        ctx.metadata["rag_results"] = rag_results
        result = await hook.execute(ctx)
        assert "rag_augmentation" not in result.metadata

    async def test_empty_rag_results_no_augmentation(self):
        """Empty rag_results should not produce augmentation."""
        hook = RAGHook(strategy="prepend")
        ctx = HookContext(
            event=HookEvent.POST_COMPACT,
            messages=[{"role": "user", "content": "Query"}],
        )
        ctx.metadata["rag_results"] = []
        result = await hook.execute(ctx)
        assert "rag_augmentation" not in result.metadata

    async def test_no_rag_results_in_metadata(self):
        """Missing rag_results in metadata should not produce augmentation."""
        hook = RAGHook(strategy="prepend")
        ctx = HookContext(
            event=HookEvent.POST_COMPACT,
            messages=[{"role": "user", "content": "Query"}],
        )
        result = await hook.execute(ctx)
        assert "rag_augmentation" not in result.metadata

    async def test_single_rag_result_formatting(self):
        """Single RAG result should be formatted correctly."""
        hook = RAGHook(strategy="prepend")
        rag_results = ["Only one context chunk"]
        ctx = HookContext(
            event=HookEvent.POST_COMPACT,
            messages=[{"role": "user", "content": "Query"}],
        )
        ctx.metadata["rag_results"] = rag_results
        result = await hook.execute(ctx)
        augmentation = result.metadata["rag_augmentation"]
        assert "[1] Only one context chunk" in augmentation

    async def test_multiple_rag_results_numbering(self):
        """Multiple RAG results should be numbered sequentially."""
        hook = RAGHook(strategy="prepend")
        rag_results = [f"Chunk {i}" for i in range(1, 6)]
        ctx = HookContext(
            event=HookEvent.POST_COMPACT,
            messages=[{"role": "user", "content": "Query"}],
        )
        ctx.metadata["rag_results"] = rag_results
        result = await hook.execute(ctx)
        augmentation = result.metadata["rag_augmentation"]
        for i in range(1, 6):
            assert f"[{i}] Chunk {i}" in augmentation

    async def test_metadata_preserved_across_execution(self):
        """Existing metadata should be preserved during execution."""
        hook = RAGHook(strategy="prepend")
        ctx = HookContext(
            event=HookEvent.POST_COMPACT,
            messages=[{"role": "user", "content": "Query"}],
        )
        ctx.metadata["existing_key"] = "existing_value"
        result = await hook.execute(ctx)
        assert result.metadata["existing_key"] == "existing_value"
        assert "rag_strategy" in result.metadata

    async def test_messages_unchanged_after_execution(self):
        """Messages should remain unchanged after RAG hook execution."""
        hook = RAGHook(strategy="prepend")
        original_messages = [{"role": "user", "content": "Query"}]
        ctx = HookContext(
            event=HookEvent.POST_COMPACT,
            messages=original_messages.copy(),
        )
        result = await hook.execute(ctx)
        assert result.messages == original_messages

    async def test_rag_results_with_special_characters(self):
        """RAG results with special characters should be handled."""
        hook = RAGHook(strategy="prepend")
        rag_results = ['Content with <tags> & "quotes"']
        ctx = HookContext(
            event=HookEvent.POST_COMPACT,
            messages=[{"role": "user", "content": "Query"}],
        )
        ctx.metadata["rag_results"] = rag_results
        result = await hook.execute(ctx)
        augmentation = result.metadata["rag_augmentation"]
        assert 'Content with <tags> & "quotes"' in augmentation

    async def test_strategy_lowercase_normalization(self):
        """Strategy should be stored as-is (case sensitive)."""
        hook = RAGHook(strategy="Prepend")  # Mixed case
        ctx = HookContext(event=HookEvent.POST_COMPACT)
        result = await hook.execute(ctx)
        assert result.metadata["rag_strategy"] == "Prepend"
