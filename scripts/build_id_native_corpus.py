#!/usr/bin/env python3
"""Build a NATIVE Indonesian RAG eval corpus + qrels from TyDi QA (secondary_task, id).

Closes the translated-text caveat: every prior ID measurement used machine-translated MS MARCO,
which inflates retrieval scores (translation normalization). TyDi QA is NATIVELY collected by
Indonesian speakers -- so this yields a caveat-free per-language ID claim.

Writes (idempotent; HF-cached after first run):
- ``data/id_native_corpus/p<sha>.txt`` -- one file per unique passage (gitignored; license-light).
- ``evals/fixtures/id_native_qrels.json`` -- committed: only query + answer + gold passage id +
  short snippet (no passage body). TyDi QA = Apache-2.0 (commit-safe).

Each TyDi secondary_task row's ``context`` IS its gold passage (goldp subset), so the corpus is the
union of unique contexts; each query's gold = its own passage id. Distractors are the other unique
passages; if the unique-passage count is below ``--density`` (default 3000, to match MS MARCO@2987),
top up with paragraphs from the indonesian ``primary_task`` articles (also native).

    pip install -e ".[eval-ragas]"   # needs `datasets`
    python scripts/build_id_native_corpus.py --n 128
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "id_native_corpus"
QRELS = ROOT / "evals" / "fixtures" / "id_native_qrels.json"
TYDI = "google-research-datasets/tydiqa"
ID_PREFIX = "indonesian"


def _passage_id(text: str) -> str:
    return "p" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _snippet(text: str, words: int = 10) -> str:
    return " ".join(text.split()[:words])


def _write_passage(corpus: Path, seen: dict[str, str], text: str) -> str:
    """Write a passage once (dedup by content-hash pid); return its pid."""
    text = (text or "").strip()
    pid = _passage_id(text)
    if pid not in seen:
        (corpus / f"{pid}.txt").write_text(text + "\n")
        seen[pid] = pid
    return pid


def _iter_secondary_id():
    """Yield indonesian TyDi secondary_task rows: {question, context, answer}."""
    from datasets import load_dataset

    ds = load_dataset(TYDI, "secondary_task", split="train")
    for row in ds:
        if not str(row.get("id", "")).startswith(ID_PREFIX):
            continue
        answers = row.get("answers") or {}
        texts = answers.get("text") or []
        answer = texts[0].strip() if texts and texts[0].strip() else ""
        question = (row.get("question") or "").strip()
        context = (row.get("context") or "").strip()
        if question and context and answer:
            yield {"question": question, "context": context, "answer": answer}


def _topup_from_primary(corpus: Path, seen: dict[str, str], target: int) -> int:
    """Add native indonesian paragraphs from primary_task articles until |seen| >= target."""
    from datasets import load_dataset

    added = 0
    ds = load_dataset(TYDI, "primary_task", split="train")
    for row in ds:
        if not str(row.get("id", "")).startswith(ID_PREFIX):
            continue
        doc = row.get("document_plaintext") or ""
        for para in (p.strip() for p in doc.split("\n\n")):
            if len(para) < 80 or len(seen) >= target:
                continue
            before = len(seen)
            _write_passage(corpus, seen, para)
            if len(seen) > before:
                added += 1
        if len(seen) >= target:
            break
    return added


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=128, help="number of queries to sample")
    ap.add_argument("--density", type=int, default=3000, help="target corpus size (passages)")
    ap.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    ap.add_argument("--out", type=Path, default=QRELS)
    args = ap.parse_args()

    args.corpus.mkdir(parents=True, exist_ok=True)
    try:
        rows = list(_iter_secondary_id())
    except Exception as e:  # noqa: BLE001
        print(f"Failed to load {TYDI} (secondary_task): {e}", file=sys.stderr)
        print("Tip: pip install -e '.[eval-ragas]' for the `datasets` lib; first run downloads "
              "TyDi QA (HF-cached).", file=sys.stderr)
        return 2

    if len(rows) < args.n:
        print(f"Only {len(rows)} indonesian rows with answers; need {args.n}.", file=sys.stderr)
        return 3

    import random

    random.seed(7)
    sample = random.sample(rows, args.n)

    # Write the sampled gold passages + collect qrels. Distractors = all OTHER unique passages.
    seen: dict[str, str] = {}
    gold_pids = set()
    qrels: list[dict] = []
    for r in sample:
        pid = _write_passage(args.corpus, seen, r["context"])
        gold_pids.add(pid)
        qrels.append(
            {
                "query": r["question"],
                "gold_doc": pid,
                "gold_needles": [_snippet(r["answer"])],
                "expected_answer": r["answer"],
            }
        )

    # Add distractors: every OTHER unique indonesian secondary_task context (native).
    for r in rows:
        if len(seen) >= args.density:
            break
        _write_passage(args.corpus, seen, r["context"])

    # Top up to target density with native primary_task paragraphs if still short.
    topped = 0
    if len(seen) < args.density:
        try:
            topped = _topup_from_primary(args.corpus, seen, args.density)
        except Exception as e:  # noqa: BLE001
            print(f"(top-up from primary_task skipped: {e})", file=sys.stderr)

    data = {
        "_comment": (
            "NATIVE Indonesian RAG eval qrels from TyDi QA (secondary_task, id) -- natively "
            "collected by Indonesian speakers (NOT translated). Built by "
            "scripts/build_id_native_corpus.py. Apache-2.0. License-light: query/answer/gold-id/"
            "snippet only. gold_doc is the data/id_native_corpus/<pid>.txt chunk doc_id."
        ),
        "corpus_dir": str(args.corpus.relative_to(ROOT)),
        "source": f"{TYDI}#secondary_task language={ID_PREFIX}",
        "qrels": qrels,
    }
    args.out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    density = len(seen)
    print(
        f"Wrote {len(qrels)} qrels -> {args.out}\n"
        f"Corpus: {density} unique passages (gold 1-in-{density}) -> {args.corpus}"
        + (f"\nTop-up: +{topped} primary_task paragraphs" if topped else ""),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
