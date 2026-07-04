# E2E Integration Suite (`tests/e2e/`)

Black-box tests that drive the live REST/SSE server (`configs/e2e_full.yaml`) over
HTTP — sessions, streaming chat, jobs, RAG, skills, tools, security, and stress. The
suite is two-tiered:

- **HARD** failures (scenario raised / no content / no terminal job) always fail pytest.
- **SOFT** misses (a keyword/tool assertion on the LLM reply) are recorded per-scenario
  and fail pytest **only** under `E2E_STRICT=1`.

## Run

```bash
# Server must already be running with the e2e config:
#   koboi serve configs/e2e_full.yaml      (or docker-compose up)
KOBOI_API_KEY=koboi_xxx ./tests/e2e/run_full.sh                  # non-strict
KOBOI_API_KEY=koboi_xxx E2E_STRICT=1 ./tests/e2e/run_full.sh     # strict (content too)
KOBOI_API_KEY=koboi_xxx E2E_CATEGORY=rag ./tests/e2e/run_full.sh # one category
KOBOI_API_KEY=koboi_xxx E2E_NAME=hotel   ./tests/e2e/run_full.sh # name substring
```

Each run writes `tests/e2e/results/run_<ts>/` (timestamped, history preserved);
`latest.txt` points at the most recent run.

## RAG retrieval in this suite

The e2e corpus is `data/sample/*` + `data/e2e/*`, indexed by the **paragraph** chunker
and a **keyword** (TF-IDF cosine) retriever, `top_k: 10`, with a small `synonyms` map
as a query-side lexical bridge. Two retrieval weaknesses are addressed:

1. `top_k` raised 5 -> 10. At 5, fact-bearing chunks ranked just outside the window
   (Express $9.99, PayPal, 14-day electronics) and never reached the model. `top_k=10`
   pulls them in (verified offline against the real retriever).
2. The `synonyms` bridge closes vocabulary gaps the keyword retriever can't (e.g. a
   guest's "dog" vs the policy's "pet"); see `KeywordRetriever(synonyms=...)`.

**Known limitation (heading-pollution).** The paragraph chunker emits standalone
heading-only chunks (`## Shipping`, `## Payment`) that score high TF-IDF cosine on a
single shared term. A chunker fix to merge bare headings into their content was tried
and reverted: it fixed specific-fact queries but regressed enumerate queries
(`skill_hotel_inquiry`) by anchoring the model on one concrete item. `top_k=10`
mitigates the pollution for this suite; the principled fix (chunking that groups
sub-sections, or the hybrid retriever) is future work.

The **general** fix for vocabulary gaps is the hybrid retriever (keyword + semantic via
RRF). `e2e_full.yaml` uses `retriever: hybrid`. The semantic leg can use a **dedicated
embedding provider**, decoupled from chat, via the optional top-level `embedding:` section
(read by `facade._build_embedding_client`). Set `EMBEDDING_API_KEY` 
in `.env` to enable semantic; with it unset, the chat client is tried for embeddings and the
semantic leg falls back to keyword (both legs then carry the synonym bridge, so hybrid
degrades cleanly with no RRF demotion). The shared embedding cache
(`koboi/rag/retriever.py:_EMBEDDING_CACHE`) embeds the corpus once per process.

The upstream proxy **does** support `/embeddings` (`text-embedding-3-small`), but
each session rebuilds the ~76-chunk index (~70 s/session), which is prohibitive across
the suite. Hybrid is recommended for single-agent production deployments; enable it
there with `retriever: hybrid`.

## Known model-dependent flakiness (not code defects)

These are characteristics of the upstream model/proxy, surfaced as SOFT misses — they
are **not** infrastructure bugs (0 HARD errors / 0 provider blocks when they occur):

- **gpt-5.4-mini over-refuses echoing identifiers.** It treats a customer ID / ZIP as
  sensitive PII and won't repeat it, so `multiturn_numeric_detail` can soft-fail under
  strict mode. mimo-v2.5 passes the same scenario. Strict-grounding models also surface
  RAG retrieval gaps that more lenient models paper over with parametric recall.
- **`stress_concurrent_8_sessions` is non-deterministic.** Under 8-way concurrency one
  reply occasionally omits the expected keyword. Per-session isolation is intact (no
  cross-session leak); this is LLM output variance under load.
- **mimo-v2.5 is ~3–4× slower** than gpt-5.4-mini through the proxy (full suite ~50 min
  vs ~14 min). Latency only; no correctness impact.
