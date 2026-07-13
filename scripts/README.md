# scripts/ -- operational helpers

Reproducible-eval corpus builders, a baseline runner, and a server model-reload helper.
None of these are installed with the package — run them from a source checkout.

## RAG eval corpora (PR #38)

The IR production-readiness eval (`docs/rag-production-readiness-eval.md`) needs real
corpora + qrels that are too large (or license-bound) to commit. Build them once (HF-cached
afterward), then run the live evals:

```bash
# English: MS MARCO v2.1 passage corpus (~3000 passages) + qrels
python scripts/build_ir_corpus.py --n 128
# Indonesian: native TyDi QA (secondary_task) corpus + qrels
python scripts/build_id_native_corpus.py --n 128

# then run the live suites (need a RERANK key + the built corpora)
RERANK_API_KEY=... koboi eval-test evals/ragas_ir_rerank.eval.py
RERANK_API_KEY=... koboi eval-test evals/ragas_ir_id_native.eval.py
```

Outputs:
- `data/ir_corpus/p<sha>.txt`, `data/id_native_corpus/...` — one file per passage (**gitignored**; MS MARCO / TyDi text stays out of the repo).
- `evals/fixtures/ir_qrels.json`, `evals/fixtures/id_native_qrels.json` — **committed** (license-light: query + answer + gold doc_id only).

## Scaling the golden set (Tier 3)

```bash
# Synthesize more Acme QA pairs toward N>=100 for tighter bootstrap CIs (any LLM key; no [eval-ragas] needed).
# HUMAN-SPOT-CHECK the output, then commit evals/fixtures/acme_qrels.json.
OPENAI_API_KEY=... python scripts/generate_rag_golden.py --n 8 --out evals/fixtures/acme_qrels.json
```

## Baseline + server ops

| Script | Purpose |
|--------|---------|
| `run_baseline.py` | Run the benchmark eval suite through the agent and save a regression baseline (`--config configs/benchmark_baseline.yaml`, `--max-cases N`). |
| `reload-model.sh` | Apply an `.env` `OPENAI_MODEL` change to the running server: recreates the container (a bare `restart` does NOT re-read `.env`), waits on the healthcheck, prints the active model, and optionally runs a one-call chat smoke (set `KOBOI_API_KEY`). |
