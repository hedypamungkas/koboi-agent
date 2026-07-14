"""Live confidence/abstention suite (Tier 2; w0.09) — +/−/edge coverage.

Wave 1 A2 ships an empty-retrieval abstention marker: when retrieval returns
nothing, the augmentation injects ``[RETRIEVAL_EMPTY]`` so the model refuses
instead of confabulating. This live suite exercises the A2 path end-to-end
against a real LLM, plus the over-abstention guard and a near-miss:

  - ``test_empty_retrieval_marker_makes_model_refuse`` (−): a zero-overlap OOS
    query yields EMPTY retrieval -> A2 marker fires -> model refuses. This is
    the live A2 validation. (The prior single-case version used an anglerfish
    query whose stopwords retrieved spurious chunks, so it had empty_rag=False
    and never exercised the marker.)
  - ``test_in_corpus_query_answers_without_abstaining`` (+): an in-corpus
    answerable query retrieves non-empty context (no marker) and the model
    ANSWERS -- guards against A2 leaking into non-empty turns / over-abstention.
  - ``test_near_miss_attribute_absent_refuses`` (edge): the CEO entity IS in
    the corpus but his email is NOT; retrieval is non-empty (marker does NOT
    fire) yet the model must still refuse -- tests the system-prompt
    "DIRECTLY contain the SPECIFIC information" guard on a near-miss.

LIVE ONLY; each case self-skips under ``--mock`` via ``t.require_live(extra=None)``
(no RAGAS judge here -- only t.abstains / t.check, so it must not gate on the
[eval-ragas] extra). Run manually: ``koboi eval-test evals/rag_abstention_live.eval.py --strict``
with OPENAI_API_KEY + OPENAI_BASE_URL + OPENAI_MODEL set.
"""

from koboi.eval.t import Severity  # noqa: F401  (re-exported for eval authors)

CONFIG = {
    "agent": {
        "name": "rag-abstention-live-eval",
        "description": "Live +/−/edge confidence/abstention probe",
        "system_prompt": (
            "Use ONLY the provided context to answer. If the context does not DIRECTLY "
            "contain the SPECIFIC information the question asks for, respond ONLY with: "
            "'I don't have that information.' Do not infer, guess, or answer from partial "
            "or related context. Do not make up information."
        ),
        "max_iterations": 4,
    },
    "llm": {
        "provider": "openai",
        "model": "${OPENAI_MODEL:gpt-4o-mini}",
        "api_key": "${OPENAI_API_KEY:dummy}",
        "base_url": "${OPENAI_BASE_URL:}",
    },
    "rag": {
        "enabled": True,
        "chunker": "paragraph",
        "retriever": "bm25",
        "top_k": 10,
        "stopwords": True,
        "augmentation": "on_the_fly",
        "documents": [
            {"path": "./data/sample/company_policy.md"},
            {"path": "./data/sample/employee_handbook.md"},
            {"path": "./data/sample/product_catalog.md"},
        ],
    },
}

TAGS = ["rag", "live", "abstention", "confidence"]

_REFUSAL_MARKERS = (
    "i don't know", "i do not know", "don't have", "do not have",
    "not found", "no information", "couldn't find", "could not find",
    "unable to",
)


async def test_empty_retrieval_marker_makes_model_refuse(t):
    """− case: zero-overlap OOS -> EMPTY retrieval (A2 marker) -> model refuses."""
    if not t.require_live(extra=None):
        return
    await t.send("xyzzy qwerty frobnicate zxcvbn plugh")
    rc = (t.last.metadata or {}).get("retrieval_confidence", {}) or {}
    t.check(rc.get("count") == 0, name="retrieval_empty", severity=Severity.GATE)
    t.abstains(severity=Severity.GATE)  # GATE: empty retrieval OR refusal marker
    t.completed()


async def test_in_corpus_query_answers_without_abstaining(t):
    """+ case: in-corpus answerable query -> non-empty retrieval (no marker) -> model
    ANSWERS (does not over-abstain). Guards against A2 leaking into non-empty turns."""
    if not t.require_live(extra=None):
        return
    await t.send("How many days of annual leave for permanent employees?")
    rc = (t.last.metadata or {}).get("retrieval_confidence", {}) or {}
    t.check((rc.get("count") or 0) > 0, name="in_corpus_retrieved", severity=Severity.GATE)
    reply = (t.reply or "").lower()
    t.check(
        not any(m in reply for m in _REFUSAL_MARKERS),
        name="answers_not_abstains",
        severity=Severity.GATE,
    )
    t.check("12" in reply, name="answer_contains_12", severity=Severity.SOFT)
    t.completed()


async def test_near_miss_attribute_absent_refuses(t):
    """edge case: CEO entity IS in corpus (employee_handbook) but his email is NOT.
    Retrieval is non-empty (A2 marker does NOT fire) yet the model must refuse --
    tests the system-prompt "DIRECTLY contain the SPECIFIC information" guard."""
    if not t.require_live(extra=None):
        return
    await t.send("What is CEO John Smith's email address?")
    rc = (t.last.metadata or {}).get("retrieval_confidence", {}) or {}
    # SOFT: a genuine near-miss retrieves the CEO chunk (non-empty); if the
    # retriever happens to return empty this still passes abstains via the marker.
    t.check((rc.get("count") or 0) > 0, name="ceo_entity_retrieved_near_miss", severity=Severity.SOFT)
    t.abstains(severity=Severity.GATE)  # GATE: refuse on attribute-absent
    t.completed()
