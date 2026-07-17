"""Runnable demo confirming the 6 RAG issues are FIXED (companion to tests/test_rag_issues.py).

Run:  .venv/bin/python verify_rag_issues.py   ->  expect "6/6 fixed"

Each check asserts the post-fix behaviour. tests/test_rag_issues.py is the CI source of
truth; this script is the human-runnable summary.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

from koboi.context.manager import SmartTruncationManager
from koboi.events import CompleteEvent
from koboi.llm.base import LLMClient
from koboi.loop import AgentCore
from koboi.memory import ConversationMemory
from koboi.orchestration.factory import AgentFactory
from koboi.rag.augmentation import OnTheFlyAugmentation
from koboi.rag.chunker import SemanticChunker, SentenceChunker
from koboi.rag.retriever import KeywordRetriever, SemanticRetriever, clear_embedding_cache
from koboi.rag.types import Chunk, Document
from koboi.types import AgentResponse

print(f"python {sys.version.split()[0]}  repo at {os.getcwd()}\n")


class _NoneEmbedClient(LLMClient):
    model = "mock-none-embed"

    async def complete(self, messages, tools=None):
        return AgentResponse(content="mock answer", tool_calls=[])

    async def complete_stream(self, messages, tools=None):
        yield CompleteEvent(response=AgentResponse(content="mock answer", tool_calls=[]))

    async def get_embeddings(self, text):
        return None


def _cap(logger_name):
    rec = logging.getLogger(logger_name)
    rec.setLevel(logging.DEBUG)
    records: list[logging.LogRecord] = []

    class _H(logging.Handler):
        def emit(self, r):
            records.append(r)

    h = _H()
    rec.addHandler(h)
    return records, lambda: rec.removeHandler(h)


def issue_1():
    print("=" * 64, "\nISSUE 1: per-agent rag merges with parent (not replaces)", "=" * 64, sep="\n")
    parent = {
        "enabled": True,
        "retriever": "keyword",
        "top_k": 3,
        "documents": [{"path": "data/sample/product_catalog.md"}],
    }
    a = AgentFactory.build_rag_from_config({"documents": [{"path": "data/sample/employee_handbook.md"}]}, parent)
    b = AgentFactory.build_rag_from_config(None, parent)
    c = AgentFactory.build_rag_from_config(
        {"enabled": True, "documents": [{"path": "data/sample/employee_handbook.md"}]}, None
    )
    d = AgentFactory.build_rag_from_config({"enabled": False}, parent)
    print(f"  partial agent block (no enabled) -> {type(a).__name__} (expect non-None)")
    print(f"  no agent block              -> {type(b).__name__}; own enabled -> {type(c).__name__}")
    print(f"  explicit enabled:false      -> {d!r} (expect None)")
    ok = a is not None and b is not None and c is not None and d is None
    print(f"  >>> {'FIXED' if ok else 'STILL BROKEN'}\n")
    return ok


def issue_2():
    print("=" * 64, "\nISSUE 2: dead RAGHook removed", "=" * 64, sep="\n")
    import importlib

    gone = False
    try:
        importlib.import_module("koboi.hooks.rag_hook")
    except ModuleNotFoundError:
        gone = True
    import subprocess

    # Production surface only: tests may legitimately mention the name in absence-assertions.
    res = subprocess.run(["grep", "-rln", "--include=*.py", "RAGHook", "koboi/"], capture_output=True, text=True)
    files = [f for f in res.stdout.strip().split("\n") if f]
    print(f"  import koboi.hooks.rag_hook raises ModuleNotFoundError? {gone}")
    print(f"  files still referencing RAGHook: {files or 'NONE'}")
    ok = gone and not files
    print(f"  >>> {'FIXED' if ok else 'STILL BROKEN'}\n")
    return ok


def issue_3():
    print("=" * 64, "\nISSUE 3: SemanticChunker warns on degradation", "=" * 64, sep="\n")
    records, detach = _cap("koboi.rag.chunker")
    sc = SemanticChunker()
    out = sc.chunk(Document(id="d", title="d", content="Cats are mammals. They purr. Dogs are loyal."))
    detach()
    sent = SentenceChunker(max_chunk_size=1000).chunk(
        Document(id="d", title="d", content="Cats are mammals. They purr. Dogs are loyal.")
    )
    warned = any(r.levelno >= logging.WARNING for r in records)
    print(f"  WARNING emitted on degradation? {warned}")
    print(f"  output still sentence-equivalent? {[c.content for c in out] == [c.content for c in sent]}")
    ok = warned
    print(f"  >>> {'FIXED' if ok else 'STILL BROKEN'}\n")
    return ok


async def issue_4():
    print("=" * 64, "\nISSUE 4: smart_truncation keeps all user messages", "=" * 64, sep="\n")
    mgr = SmartTruncationManager(keep_last=6)
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(1, 13):
        msgs.append({"role": "user", "content": f"turn {i} " + "word " * 120})
        msgs.append({"role": "assistant", "content": "reply " + "z " * 120})
    fact = "ZZZ_SECRET_FACT_987654321_zzz"
    msgs[3]["content"] = f"note: the {fact} must be remembered"
    out = await mgr.manage(msgs, max_tokens=50)
    joined = " ".join(m["content"] for m in out)
    print(f"  mid-conversation fact survived? {fact in joined}")
    ok = fact in joined and len(out) < len(msgs)
    print(f"  >>> {'FIXED' if ok else 'STILL BROKEN'}\n")
    return ok


async def issue_5():
    print("=" * 64, "\nISSUE 5: no-embeddings provider WARNs and names the fix", "=" * 64, sep="\n")
    clear_embedding_cache()
    records, detach = _cap("koboi.rag.retriever")
    chunks = [
        Chunk(id=f"c{i}", doc_id="kb", content=c, metadata={"source": "kb"})
        for i, c in enumerate(["Refund window 30 days.", "Invoices Net 14.", "Enterprise support."])
    ]
    sem = SemanticRetriever(chunks, client=_NoneEmbedClient())
    res = await sem.retrieve("refund policy", top_k=2)
    detach()
    warned = any(
        r.levelno >= logging.WARNING and ("embedding:" in r.getMessage() or "embeddings" in r.getMessage())
        for r in records
    )
    print(f"  WARNING (not INFO) mentioning embedding:? {warned}")
    print(f"  retrieval method(s): {[r.retrieval_method for r in res]}")
    ok = warned and any("fallback" in r.retrieval_method for r in res)
    print(f"  >>> {'FIXED' if ok else 'STILL BROKEN'}\n")
    return ok


async def issue_6():
    print("=" * 64, "\nISSUE 6: run_stream CompleteEvent carries rag_results", "=" * 64, sep="\n")

    def agent():
        chunks = [Chunk(id="c0", doc_id="kb", content="refund window is 30 days", metadata={"source": "kb"})]
        return AgentCore(
            client=_NoneEmbedClient(),
            memory=ConversationMemory(system_prompt="s"),
            augmentation=OnTheFlyAugmentation(retriever=KeywordRetriever(chunks), top_k=2),
            max_iterations=3,
        )

    run_ok = bool((await agent().run("refund window?")).metadata.get("rag_results"))
    ce = None
    async for ev in agent().run_stream("refund window?"):
        if isinstance(ev, CompleteEvent):
            ce = ev
    stream_ok = bool(ce and ce.metadata.get("rag_results"))
    print(f"  run() rag_results? {run_ok}; run_stream CompleteEvent rag_results? {stream_ok}")
    ok = run_ok and stream_ok
    print(f"  >>> {'FIXED' if ok else 'STILL BROKEN'}\n")
    return ok


async def main():
    results = [
        ("1 per-agent merge", issue_1()),
        ("2 RAGHook removed", issue_2()),
        ("3 SemanticChunker warns", issue_3()),
        ("4 smart_truncation keeps users", await issue_4()),
        ("5 no-embeddings WARNING", await issue_5()),
        ("6 streaming rag_results", await issue_6()),
    ]
    print("=" * 64, "\nSUMMARY", "=" * 64, sep="\n")
    for name, ok in results:
        print(f"  [{'FIXED' if ok else 'STILL BROKEN'}] Issue {name}")
    print(f"\n  {sum(1 for _, ok in results if ok)}/6 fixed.")


if __name__ == "__main__":
    asyncio.run(main())
