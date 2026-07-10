"""Example 33: Modern RAG pipeline — BM25 + rewriting + filtering + reranking + caches.

Demonstrates every RAG capability shipped in the consolidated stack:
  - #8 BM25 retriever (saturation + length-norm, not TF-IDF)
  - #9 query rewriting (LLM rewrites verbose queries before retrieval)
  - #10 metadata filtering (source scoping; relevance, NOT ACL)
  - #11a reranking (RerankerRetriever wrapper for precision)
  - #11b dedup + #12 numbered citations (automatic in augmentation)
  - #5 embedding cache + #1 document cache (persist across restarts)
  - #3 globbing + size-cap (OOM guard)

Remote sources (R2/S3/HTTP) + table-extraction are documented in the YAML
(commented out). This script runs with local sample docs so it works without
R2 creds or optional extras.

Optional extras:
    pip install koboi-agent[rag]        # PDF/DOCX parsing + table extraction
    pip install koboi-agent[rag-cloud]  # S3/R2 via boto3

Run:
    python examples/33_rag_modern_pipeline.py
    python examples/33_rag_modern_pipeline.py -m interactive
"""

from __future__ import annotations

import click
from rich.panel import Panel

from conftest import (
    console,
    setup_example,
    dual_mode_options,
    create_agent,
    automatic_batch,
    interactive_loop,
)

QUESTIONS = [
    "How many annual leave days do permanent employees get?",
    "What SaaS packages are available and how much do they cost?",
    "What is the remote work policy?",
]


def _show_active_features(agent) -> None:
    """Print a panel showing which RAG features are active on the built agent."""
    try:
        core = getattr(agent, "_core", agent)
        aug = getattr(core, "augmentation", None)
        if not aug:
            return
        features: list[str] = []
        retriever = getattr(aug, "retriever", None)
        if retriever and hasattr(retriever, "_base"):
            features.append("rerank")
        if getattr(aug, "_query_rewrite", False):
            features.append("query_rewrite")
        if getattr(aug, "_hyde", False):
            features.append("hyde")
        if getattr(aug, "metadata_filter", None):
            features.append(f"filter={aug.metadata_filter}")
        if features:
            console.print(
                Panel(
                    ", ".join(features),
                    title="Active RAG Features",
                    border_style="cyan",
                )
            )
    except Exception:
        pass  # best-effort display; never block the example


@click.command()
@dual_mode_options
def main(mode: str, verbose: bool) -> None:
    setup_example(
        "Modern RAG Pipeline",
        "BM25 + rewriting + filtering + reranking + caches",
    )
    agent = create_agent("33_rag_modern_pipeline", verbose=verbose)
    _show_active_features(agent)

    if mode == "automatic":
        automatic_batch(agent, QUESTIONS)
    else:
        interactive_loop(agent)


if __name__ == "__main__":
    main()
