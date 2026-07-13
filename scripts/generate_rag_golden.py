#!/usr/bin/env python3
"""Generate/scale the RAG golden QA set (Tier 3) over the Acme corpus.

Reuses koboi's own ``RAGASDataGenerator`` (an LLM-based QA synthesizer, NOT the ragas
library -- so it runs with any LLM key, no ``[eval-ragas]`` needed). Run OFFLINE to
scale ``evals/fixtures/acme_qrels.json`` from the hand-authored N toward N>=100 for
tighter bootstrap CIs in ``evals/ragas_golden_suite.eval.py``; then HUMAN-SPOT-CHECK
and commit the regenerated file.

    OPENAI_API_KEY=... python scripts/generate_rag_golden.py --n 8 --out evals/fixtures/acme_qrels.json

The generated entries preserve the committed schema: {query, gold_needles, gold_doc}.
Existing hand-authored entries are kept; generated ones are appended (deduped by query).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "evals" / "fixtures" / "acme_qrels.json"
CORPUS = [
    ROOT / "data/sample/company_policy.md",
    ROOT / "data/sample/employee_handbook.md",
    ROOT / "data/sample/product_catalog.md",
]


def _build_client():
    from koboi.llm.factory import create_client

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or ""
    if not api_key or api_key == "dummy":
        raise SystemExit("Set a real OPENAI_API_KEY (or ANTHROPIC_API_KEY) to generate golden QA.")
    return create_client(
        provider=os.environ.get("OPENAI_PROVIDER", "openai"),
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL", ""),
    )


async def _generate(n_per_doc: int) -> list[dict]:
    from koboi.eval.loaders.ragas_generator import RAGASDataGenerator

    gen = RAGASDataGenerator(_build_client())
    cases = await gen.generate_from_docs([str(p) for p in CORPUS], num_questions_per_doc=n_per_doc)
    entries: list[dict] = []
    for c in cases:
        ans = (c.expected_answer or "").strip()
        if not c.user_message or not ans:
            continue
        entries.append(
            {
                "query": c.user_message,
                "gold_needles": [ans],
                "gold_doc": Path(c.metadata.get("source_doc", "")).name,
                "_synthetic": True,
            }
        )
    return entries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=8, help="questions per doc (3 docs -> ~3*N entries)")
    ap.add_argument("--out", type=Path, default=FIXTURE)
    args = ap.parse_args()

    data = json.loads(args.out.read_text()) if args.out.exists() else {"qrels": []}
    existing = {q["query"] for q in data.get("qrels", [])}
    generated = asyncio.run(_generate(args.n))
    added = [e for e in generated if e["query"] not in existing]
    data.setdefault("qrels", []).extend(added)
    args.out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"Generated {len(generated)} entries; added {len(added)} new (deduped). "
          f"Total qrels now: {len(data['qrels'])} -> {args.out}", file=sys.stderr)
    print("HUMAN-SPOT-CHECK the synthetic entries (gold_needles accuracy) before committing.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
