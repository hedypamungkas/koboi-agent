"""Example 08: RAG Advanced -- comparison of RAG configurations.

Demonstrates:
- 4 chunker + retriever combinations
- Processes 3 sample documents from data/sample/
- Dual mode: automatic (compare all configs) and interactive (pick config, chat freely)

Run:
    python examples/07_rag_advanced.py                  # automatic mode
    python examples/07_rag_advanced.py -m interactive   # interactive mode
"""

from __future__ import annotations

import sys
import time

import click
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
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


def load_documents() -> list:
    """Load 3 sample documents from data/sample/."""
    from koboi.rag.types import Document

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


def build_config(chunker_cls, retriever_cls, documents, client=None):
    """Build one RAG configuration, return (label, chunk_count, retriever)."""
    if chunker_cls.__name__ == "FixedSizeChunker":
        chunker = chunker_cls(chunk_size=500, overlap=50)
    elif chunker_cls.__name__ == "SentenceChunker":
        chunker = chunker_cls(max_chunk_size=800)
    else:
        chunker = chunker_cls(max_chunk_size=1000)

    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk(doc)
        all_chunks.extend(chunks)

    if retriever_cls.__name__ == "SemanticRetriever":
        retriever = retriever_cls(chunks=all_chunks, client=client)
    else:
        retriever = retriever_cls(chunks=all_chunks)

    label = f"{chunker_cls.__name__} + {retriever_cls.__name__}"
    return label, len(all_chunks), retriever


def run_queries(retriever, queries, top_k=3):
    """Run queries through retriever, return results per query."""
    results = []
    for query in queries:
        retrieved = run_async(retriever.retrieve(query, top_k=top_k))
        best_score = retrieved[0].score if retrieved else 0.0
        snippet = ""
        if retrieved:
            snippet = retrieved[0].chunk.content[:80].replace("\n", " ")
        results.append({"query": query, "score": round(best_score, 4), "snippet": snippet})
    return results


QUERIES = [
    "What software products does Acme Corp offer?",
    "What is the price of AcmeERP Enterprise?",
    "What is the new employee onboarding process?",
]

CONFIG_OPTIONS = [
    ("SentenceChunker + KeywordRetriever", "SentenceChunker", "KeywordRetriever"),
    ("SentenceChunker + SemanticRetriever", "SentenceChunker", "SemanticRetriever"),
    ("FixedSizeChunker + KeywordRetriever", "FixedSizeChunker", "KeywordRetriever"),
    ("ParagraphChunker + KeywordRetriever", "ParagraphChunker", "KeywordRetriever"),
]


def run_automatic(verbose: bool):
    """Compare 4 configurations automatically."""
    from koboi.rag.chunker import FixedSizeChunker, ParagraphChunker, SentenceChunker
    from koboi.rag.retriever import KeywordRetriever, SemanticRetriever
    from koboi.client import Client

    documents = load_documents()
    if not documents:
        console.print("[red]No documents found. Make sure data/sample/ exists.[/red]")
        sys.exit(1)

    console.print(f"[dim]Documents loaded: {len(documents)} files[/dim]")

    try:
        client = Client()
    except Exception as e:
        console.print(f"[yellow]Warning: Client could not be created ({e}). SemanticRetriever will fall back.[/yellow]")
        client = None

    configs = [
        (SentenceChunker, KeywordRetriever),
        (SentenceChunker, SemanticRetriever),
        (FixedSizeChunker, KeywordRetriever),
        (ParagraphChunker, KeywordRetriever),
    ]

    all_rows = []
    for chunker_cls, retriever_cls in configs:
        label, chunk_count, retriever = build_config(chunker_cls, retriever_cls, documents, client=client)
        if verbose:
            console.print(f"\n[bold cyan]Config: {label}[/bold cyan]")
            console.print(f"  Total chunks: {chunk_count}")
        query_results = run_queries(retriever, QUERIES, top_k=3)
        for qr in query_results:
            all_rows.append(
                {
                    "config": label,
                    "query": qr["query"],
                    "chunk_count": chunk_count,
                    "score": qr["score"],
                    "snippet": qr["snippet"],
                }
            )

    # Results table
    table = Table(title="RAG Configuration Comparison", show_lines=True, title_style="bold magenta")
    table.add_column("Config", style="cyan", max_width=40)
    table.add_column("Query", style="white", max_width=35)
    table.add_column("Chunks", justify="right", style="green")
    table.add_column("Score", justify="right", style="yellow")
    table.add_column("Snippet", style="dim", max_width=45)

    for row in all_rows:
        table.add_row(row["config"], row["query"], str(row["chunk_count"]), f"{row['score']:.4f}", row["snippet"])

    console.print()
    console.print(table)

    # Demo: AgentCore with RAG
    console.print()
    console.print("[bold]Demo: AgentCore with RAG augmentation[/bold]")
    console.print("[dim]Using ParagraphChunker + KeywordRetriever[/dim]")

    from koboi.rag.augmentation import InMemoryAugmentation
    from koboi.loop import AgentCore
    from koboi.memory import ConversationMemory

    _, _, demo_retriever = build_config(ParagraphChunker, KeywordRetriever, documents)
    augmentation = InMemoryAugmentation(retriever=demo_retriever, top_k=3)

    try:
        demo_client = Client()
        demo_agent = AgentCore(
            client=demo_client,
            memory=ConversationMemory(
                system_prompt="You are an assistant that answers based on document context. Always cite the source."
            ),
            augmentation=augmentation,
            max_iterations=8,
            verbose=verbose,
        )

        demo_query = "What software products does Acme Corp offer?"
        console.print(f"\n[bold green]Query:[/bold green] {demo_query}")

        start = time.time()
        result = run_async(demo_agent.run(demo_query))
        elapsed = time.time() - start

        console.print(f"[bold blue]Answer ({elapsed:.2f}s):[/bold blue]")
        console.print(str(result))
    except Exception as e:
        console.print(f"[red]Demo failed: {e}[/red]")
        console.print("[dim]Make sure OPENAI_API_KEY is set in .env[/dim]")


def run_interactive(verbose: bool):
    """Pick a configuration, then chat freely with RAG augmentation."""
    from koboi.rag.chunker import FixedSizeChunker, ParagraphChunker, SentenceChunker
    from koboi.rag.retriever import KeywordRetriever, SemanticRetriever
    from koboi.rag.augmentation import InMemoryAugmentation
    from koboi.loop import AgentCore
    from koboi.memory import ConversationMemory
    from koboi.client import Client

    documents = load_documents()
    if not documents:
        console.print("[red]No documents found.[/red]")
        return

    console.print("[bold]Available RAG configurations:[/bold]")
    for i, (label, _, _) in enumerate(CONFIG_OPTIONS, 1):
        console.print(f"  {i}. {label}")

    choices = [str(i) for i in range(1, len(CONFIG_OPTIONS) + 1)]
    choice = Prompt.ask("Pick a configuration number", choices=choices)
    selected = CONFIG_OPTIONS[int(choice) - 1]
    selected_label = selected[0]

    console.print(f"\n[bold cyan]Using: {selected_label}[/bold cyan]\n")

    chunker_map = {
        "SentenceChunker": SentenceChunker,
        "FixedSizeChunker": FixedSizeChunker,
        "ParagraphChunker": ParagraphChunker,
    }
    retriever_map = {
        "KeywordRetriever": KeywordRetriever,
        "SemanticRetriever": SemanticRetriever,
    }

    try:
        client = Client()
    except Exception:
        client = None

    _, _, retriever = build_config(
        chunker_map[selected[1]],
        retriever_map[selected[2]],
        documents,
        client=client,
    )
    augmentation = InMemoryAugmentation(retriever=retriever, top_k=3)

    demo_client = Client() if client is None else client
    core = AgentCore(
        client=demo_client,
        memory=ConversationMemory(
            system_prompt="You are an assistant that answers based on document context. Always cite the source."
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
            console.print(Panel(Markdown(str(result)), title=f"Agent ({selected_label})", border_style="green"))
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@click.command()
@dual_mode_options
def main(mode: str, verbose: bool):
    """Example 08: RAG configuration comparison."""
    setup_example(
        "Example 08: RAG Advanced",
        "Comparing 4 Chunker + Retriever combinations\n"
        "with 3 documents and 3 queries.\n\n"
        "[dim]Run with -m interactive to pick a config and chat freely.[/dim]",
    )

    if mode == "interactive":
        run_interactive(verbose)
    else:
        run_automatic(verbose)


if __name__ == "__main__":
    main()
