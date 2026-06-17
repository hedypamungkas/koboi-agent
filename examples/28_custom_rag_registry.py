"""Example 28: Custom RAG Registry -- extensible RAG components.

Demonstrates:
- Registering custom chunkers and retrievers via decorators
- Using build_rag() with custom components from YAML config
- Comparing built-in vs custom RAG pipelines
- YAML-driven custom module loading (custom_modules)

Run:
    python examples/28_custom_rag_registry.py                  # automatic mode
    python examples/28_custom_rag_registry.py -m interactive   # interactive mode
"""

from __future__ import annotations

import sys
import time

import click
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from conftest import (
    console,
    ensure_path,
    load_env,
    setup_example,
    dual_mode_options,
    run_async,
)

ensure_path()
load_env()

from pathlib import Path


# ---------------------------------------------------------------------------
# Custom RAG components -- registered via decorators, no core code changes
# ---------------------------------------------------------------------------

from koboi.rag.types import Chunk, Document, RetrievalResult
from koboi.rag.chunker import BaseChunker
from koboi.rag.retriever import BaseRetriever
from koboi.rag.registry import (
    register_chunker,
    register_retriever,
    build_rag,
    chunker_registry,
    retriever_registry,
    augmentation_registry,
)


@register_chunker(
    "word_count",
    description="Chunks by word count with sentence-boundary snapping",
)
class WordCountChunker(BaseChunker):
    """Splits text into chunks of N words, snapping to sentence boundaries."""

    def __init__(self, words_per_chunk: int = 100):
        self.words_per_chunk = words_per_chunk

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content.strip()
        if not text:
            return []

        sentences = text.replace("\n", " ").split(". ")
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: list[Chunk] = []
        current_words: list[str] = []
        current_count = 0
        index = 0

        for sentence in sentences:
            words = sentence.split()
            if current_count + len(words) > self.words_per_chunk and current_words:
                content = ". ".join(current_words) + "."
                chunks.append(self._make_chunk(document.id, index, content))
                index += 1
                current_words = []
                current_count = 0

            current_words.append(sentence)
            current_count += len(words)

        if current_words:
            content = ". ".join(current_words)
            if not content.endswith("."):
                content += "."
            chunks.append(self._make_chunk(document.id, index, content))

        return chunks


@register_retriever(
    "bm25",
    description="BM25 scoring with configurable k1 and b parameters",
)
class BM25Retriever(BaseRetriever):
    """BM25 retrieval -- a ranking function used by search engines.

    Uses term frequency saturation (k1) and document length normalization (b).
    """

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self._chunks = chunks
        self._k1 = k1
        self._b = b
        self._avg_dl = 0.0
        self._doc_freqs: dict[str, int] = {}
        self._doc_lens: list[int] = []
        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        import re

        return re.findall(r"\w+", text.lower())

    def _build_index(self) -> None:
        import math

        total_len = 0
        for chunk in self._chunks:
            terms = self._tokenize(chunk.content)
            self._doc_lens.append(len(terms))
            total_len += len(terms)
            seen = set()
            for t in terms:
                if t not in seen:
                    self._doc_freqs[t] = self._doc_freqs.get(t, 0) + 1
                    seen.add(t)

        n = len(self._chunks) or 1
        self._avg_dl = total_len / n
        self._n = n

    def _score(self, query_terms: list[str], chunk_idx: int) -> float:
        import math

        score = 0.0
        dl = self._doc_lens[chunk_idx]
        content_terms = self._tokenize(self._chunks[chunk_idx].content)
        term_counts = {}
        for t in content_terms:
            term_counts[t] = term_counts.get(t, 0) + 1

        for qt in query_terms:
            if qt not in term_counts:
                continue
            tf = term_counts[qt]
            df = self._doc_freqs.get(qt, 0)
            idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1)
            tf_norm = (tf * (self._k1 + 1)) / (tf + self._k1 * (1 - self._b + self._b * dl / (self._avg_dl + 1)))
            score += idf * tf_norm

        return score

    async def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        scored = [(self._chunks[i], self._score(query_terms, i)) for i in range(len(self._chunks))]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(chunk=chunk, score=score, retrieval_method="bm25")
            for chunk, score in scored[:top_k]
            if score > 0
        ]


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------


def load_documents() -> list[Document]:
    """Load sample documents from data/sample/."""
    doc_dir = Path(__file__).resolve().parent.parent / "data" / "sample"
    files = ["company_policy.md", "product_catalog.md", "employee_handbook.md"]
    documents = []
    for fname in files:
        fpath = doc_dir / fname
        if fpath.exists():
            content = fpath.read_text(encoding="utf-8")
            documents.append(Document(id=fname, title=fname.replace(".md", ""), content=content))
        else:
            console.print(f"[yellow]Warning: {fpath} not found[/yellow]")
    return documents


QUERIES = [
    "What software products does Acme Corp offer?",
    "How many annual leave days do employees get?",
    "What is the remote work policy?",
]


def show_registry_state():
    """Display currently registered components."""
    table = Table(title="RAG Component Registry", show_lines=True)
    table.add_column("Type", style="cyan", width=14)
    table.add_column("Name", style="green")
    table.add_column("Description", style="dim")

    for name in chunker_registry.list_available():
        entry = chunker_registry.get(name)
        table.add_row("Chunker", name, entry.description if entry else "")
    for name in retriever_registry.list_available():
        entry = retriever_registry.get(name)
        table.add_row("Retriever", name, entry.description if entry else "")
    for name in augmentation_registry.list_available():
        entry = augmentation_registry.get(name)
        table.add_row("Augmentation", name, entry.description if entry else "")

    console.print(table)
    console.print()


def compare_retrievers(documents: list[Document]):
    """Compare built-in keyword retriever vs custom BM25 retriever."""
    from koboi.rag.chunker import ParagraphChunker

    chunker = ParagraphChunker(max_chunk_size=800)
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunker.chunk(doc))

    console.print(f"[dim]Total chunks: {len(all_chunks)}[/dim]\n")

    # Built-in keyword retriever via registry
    kw_config = {
        "enabled": True,
        "chunker": "paragraph",
        "retriever": "keyword",
        "top_k": 3,
        "documents": [
            {"path": str(Path(__file__).resolve().parent.parent / "data" / "sample" / f)}
            for f in ["company_policy.md", "product_catalog.md", "employee_handbook.md"]
            if (Path(__file__).resolve().parent.parent / "data" / "sample" / f).exists()
        ],
    }

    # Custom BM25 retriever via registry
    bm25_config = {
        "enabled": True,
        "chunker": "paragraph",
        "retriever": "bm25",
        "top_k": 3,
        "documents": kw_config["documents"],
    }

    # Custom word_count chunker + BM25 retriever
    custom_config = {
        "enabled": True,
        "chunker": "word_count",
        "words_per_chunk": 80,
        "retriever": "bm25",
        "top_k": 3,
        "documents": kw_config["documents"],
    }

    configs = [
        ("Keyword (built-in)", kw_config),
        ("BM25 (custom)", bm25_config),
        ("WordCount + BM25 (both custom)", custom_config),
    ]

    table = Table(title="Retriever Comparison", show_lines=True, title_style="bold magenta")
    table.add_column("Config", style="cyan", max_width=30)
    table.add_column("Query", style="white", max_width=35)
    table.add_column("Score", justify="right", style="yellow")
    table.add_column("Top Result Snippet", style="dim", max_width=50)

    for label, config in configs:
        augmentation = build_rag(config)
        if augmentation is None:
            console.print(f"[yellow]Warning: {label} -- no documents loaded[/yellow]")
            continue

        retriever = augmentation.retriever
        # If wrapped in RerankerRetriever or similar, unwrap
        if hasattr(retriever, "_base"):
            retriever = retriever._base

        for query in QUERIES:
            results = run_async(retriever.retrieve(query, top_k=3))
            score = results[0].score if results else 0.0
            snippet = results[0].chunk.content[:80].replace("\n", " ") if results else ""
            table.add_row(label, query, f"{score:.4f}", snippet)

    console.print()
    console.print(table)


def run_automatic(verbose: bool):
    """Run automatic comparison of RAG configurations."""
    documents = load_documents()
    if not documents:
        console.print("[red]No documents found. Make sure data/sample/ exists.[/red]")
        sys.exit(1)

    console.print(f"[dim]Documents loaded: {len(documents)} files[/dim]\n")

    # Show registry state
    show_registry_state()

    # Compare retrievers
    compare_retrievers(documents)

    # Demo: build_rag with YAML config
    console.print()
    console.print("[bold]Demo: build_rag() from YAML-style config dict[/bold]")
    console.print("[dim]Custom chunker 'word_count' + custom retriever 'bm25'[/dim]\n")

    doc_paths = [
        {"path": str(Path(__file__).resolve().parent.parent / "data" / "sample" / f)}
        for f in ["company_policy.md", "product_catalog.md", "employee_handbook.md"]
        if (Path(__file__).resolve().parent.parent / "data" / "sample" / f).exists()
    ]

    augmentation = build_rag(
        {
            "enabled": True,
            "chunker": "word_count",
            "words_per_chunk": 80,
            "retriever": "bm25",
            "top_k": 3,
            "augmentation": "in_memory",
            "documents": doc_paths,
        }
    )

    if augmentation:
        query = "What software products does Acme Corp offer?"
        augmented_msg = run_async(augmentation.augment_for_memory(query))

        console.print(f"[bold green]Query:[/bold green] {query}")
        console.print(
            Panel(
                augmented_msg[:600] + ("..." if len(augmented_msg) > 600 else ""),
                title="[dim]Augmented Message (with retrieved context)[/dim]",
                border_style="dim",
            )
        )
    else:
        console.print("[yellow]Could not build RAG pipeline[/yellow]")


def run_interactive(verbose: bool):
    """Interactive mode: pick a config and chat with RAG augmentation."""
    from koboi.rag.augmentation import InMemoryAugmentation
    from koboi.loop import AgentCore
    from koboi.memory import ConversationMemory
    from koboi.client import Client

    documents = load_documents()
    if not documents:
        console.print("[red]No documents found.[/red]")
        return

    show_registry_state()

    console.print("[bold]Available RAG pipelines:[/bold]")
    console.print("  1. ParagraphChunker + KeywordRetriever (built-in)")
    console.print("  2. ParagraphChunker + BM25Retriever (custom retriever)")
    console.print("  3. WordCountChunker + BM25Retriever (both custom)")
    console.print()

    from rich.prompt import Prompt

    choice = Prompt.ask("Pick a pipeline", choices=["1", "2", "3"], default="3")

    doc_paths = [
        {"path": str(Path(__file__).resolve().parent.parent / "data" / "sample" / f)}
        for f in ["company_policy.md", "product_catalog.md", "employee_handbook.md"]
        if (Path(__file__).resolve().parent.parent / "data" / "sample" / f).exists()
    ]

    pipeline_configs = {
        "1": {"chunker": "paragraph", "retriever": "keyword"},
        "2": {"chunker": "paragraph", "retriever": "bm25"},
        "3": {"chunker": "word_count", "words_per_chunk": 80, "retriever": "bm25"},
    }

    config = {
        "enabled": True,
        "top_k": 3,
        "augmentation": "in_memory",
        "documents": doc_paths,
        **pipeline_configs[choice],
    }

    augmentation = build_rag(config)
    if not augmentation:
        console.print("[red]Could not build RAG pipeline[/red]")
        return

    pipeline_label = f"{config['chunker']} + {config['retriever']}"
    console.print(f"\n[bold cyan]Using: {pipeline_label}[/bold cyan]\n")

    try:
        client = Client()
    except Exception as e:
        console.print(f"[red]Error creating LLM client: {e}[/red]")
        console.print("[dim]Make sure OPENAI_API_KEY is set in .env[/dim]")
        return

    core = AgentCore(
        client=client,
        memory=ConversationMemory(
            system_prompt="You are an assistant that answers based on document context. "
            "Always cite the source of information."
        ),
        augmentation=augmentation,
        max_iterations=8,
        verbose=verbose,
    )

    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye![/dim]")
            break

        if user_input.strip().lower() in ("quit", "exit", "q"):
            console.print("[dim]Bye![/dim]")
            break
        if not user_input.strip():
            continue

        try:
            result = run_async(core.run(user_input))
            console.print(
                Panel(
                    Markdown(str(result)),
                    title=f"Agent ({pipeline_label})",
                    border_style="green",
                )
            )
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@click.command()
@dual_mode_options
def main(mode: str, verbose: bool):
    """Example 28: Custom RAG Registry."""
    setup_example(
        "Example 28: Custom RAG Registry",
        "Extensible RAG components via decorator-based registry.\n"
        "Registers custom WordCountChunker and BM25Retriever,\n"
        "then compares built-in vs custom pipelines.\n\n"
        "[dim]Run with -m interactive to chat with custom RAG.[/dim]",
    )

    if mode == "interactive":
        run_interactive(verbose)
    else:
        run_automatic(verbose)


if __name__ == "__main__":
    main()
