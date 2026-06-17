"""Shared fixtures for benchmarks."""
import pytest
from koboi.config import Config
from koboi.memory import ConversationMemory
from koboi.tools.registry import ToolRegistry
from koboi.hooks.chain import HookChain, Hook, HookContext, HookEvent
from koboi.harness.telemetry import TelemetryCollector
from koboi.context.manager import TruncationManager
from koboi.rag.chunker import FixedSizeChunker, SentenceChunker
from koboi.rag.retriever import KeywordRetriever
from koboi.rag.augmentation import InMemoryAugmentation
from koboi.rag.types import Document, Chunk


@pytest.fixture
def sample_config_dict():
    """Sample config dict for benchmarking config loading."""
    return {
        "agent": {
            "name": "benchmark-agent",
            "max_iterations": 10,
            "system_prompt": "You are a helpful assistant.",
        },
        "llm": {
            "model": "gpt-4o-mini",
            "provider": "openai",
            "api_key": "test-key-12345",
            "temperature": 0.7,
        },
        "rag": {
            "enabled": True,
            "chunk_size": 500,
        },
    }


@pytest.fixture
def sample_messages():
    """Sample message list for benchmarking."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "Tell me more about it."},
    ]


@pytest.fixture
def sample_text_50kb():
    """Generate ~50KB of sample text for chunking benchmarks."""
    base_text = "This is a sample sentence for testing text chunking. " * 10
    paragraphs = [base_text for _ in range(100)]
    return "\n\n".join(paragraphs)


@pytest.fixture
def populated_tool_registry():
    """Tool registry with sample tools."""
    registry = ToolRegistry()
    for i in range(10):
        registry.register(
            name=f"tool_{i}",
            description=f"Tool number {i}",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            fn=lambda x: f"Result: {x}",
        )
    return registry


@pytest.fixture
def sample_chunks():
    """Create sample chunks for retrieval benchmarks."""
    chunks = []
    for i in range(100):
        chunk = Chunk(
            id=f"chunk_{i}",
            doc_id="doc_1",
            content=f"Sample chunk content {i}. This contains keywords like search, retrieve, and test.",
            metadata={"index": i},
        )
        chunks.append(chunk)
    return chunks


@pytest.fixture
def mock_hook():
    """Create a mock hook for benchmarking."""

    class MockHook(Hook):
        def handles(self):
            return [HookEvent.PRE_TOOL_USE, HookEvent.POST_TOOL_USE]

        async def execute(self, ctx):
            ctx.metadata["hook_ran"] = True
            return ctx

    return MockHook


@pytest.fixture
def hook_chain_with_5_hooks(mock_hook):
    """Hook chain with 5 hooks."""
    chain = HookChain()
    for _ in range(5):
        chain.add(mock_hook())
    return chain


@pytest.fixture
def telemetry_collector():
    """Fresh telemetry collector instance."""
    return TelemetryCollector(session_id="benchmark_session")


@pytest.fixture
def truncation_manager():
    """Truncation manager instance."""
    return TruncationManager(keep_last=6)


@pytest.fixture
def fixed_size_chunker():
    """Fixed size chunker for benchmarks."""
    return FixedSizeChunker(chunk_size=500, overlap=50)


@pytest.fixture
def sentence_chunker():
    """Sentence chunker for benchmarks."""
    return SentenceChunker(max_chunk_size=800)


@pytest.fixture
def keyword_retriever(sample_chunks):
    """Keyword retriever with sample chunks."""
    return KeywordRetriever(chunks=sample_chunks)


@pytest.fixture
def sample_document(sample_text_50kb):
    """Sample document for chunking."""
    return Document(id="doc_1", title="Sample Doc", content=sample_text_50kb)
