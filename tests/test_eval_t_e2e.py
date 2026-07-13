"""Golden end-to-end test: run the shipped ``evals/`` directory and assert outcomes.

Unlike the other ``test_eval_t_*`` files (which build throwaway cases in memory),
this runs the *committed* sample evals through the real discover -> run -> fold
pipeline. It locks the samples from rotting and catches regressions in the
end-to-end path against actual files (no API key -- all samples are mock-driven).
"""

from __future__ import annotations

from pathlib import Path

from koboi.eval.t import run_tests

EVALS_DIR = Path(__file__).resolve().parent.parent / "evals"


class TestShippedEvalsGolden:
    async def test_evals_directory_outcomes(self):
        results = await run_tests(EVALS_DIR, threshold=0.6, mock=True)

        # All mock-driven (R4 made RAG retrieval mock-safe via t.retrievedChunk).
        # Core samples (11): weather (2) + no_tools (1) + multi_turn (1)
        # + guardrail_block (2) + mode_blocked (1) + rag_retrieval (2)
        # + guardrail_output_warn (1) + skill_activation (1).
        # RAG production-readiness Tier-1 mock gate (24): rag_ranking (4)
        # + rag_ranking_ci (1) + rag_abstention (4) + rag_noise_robustness (2)
        # + rag_citations (3) + rag_ingestion_fidelity (5) + rag_metadata_filter (3)
        # + rag_rerank_wiring (2) -- cross-encoder rerank wiring (wrapper invoked +
        #   fail-soft preserves retrieval; zero-egress fast-fail probe).
        # Tier-2/3 live evals (17): rag_answer_correctness (5)
        # + rag_semantic_ranking (1) + rag_hybrid_ranking (1)
        # + rag_abstention_live (1) + rag_noise_faithfulness (1) + rag_hyde_recall (1)
        # + ragas_ir_suite (1) + ragas_ir_adversarial (4) + ragas_ir_rerank (1)
        # + ragas_ir_id_native (1) -- NATIVE Indonesian (TyDi QA-id), caveat-free ID claim.
        # (ragas_golden_suite + the Acme ragas_faithfulness evals were REMOVED -- superseded by
        # the real-corpus N=128/N=48 measurements; their §7a numbers were self-inflated.)
        # -- self-skip under mock via t.require_live() (live_skip), so they pass here
        # and run for real on a manual `--tags live` run.
        assert len(results) == 52

        passed = [r for r in results if r.passed]
        failed = [r for r in results if not r.passed]
        assert len(passed) == 52
        assert len(failed) == 0
        # All shipped sample evals pass. The weather file's second case demonstrates
        # GATE-vs-SOFT: a non-matching SOFT check dents the score without failing
        # the gate (so `koboi eval-test --strict` stays green).

    async def test_multi_turn_sample_recorded_two_turns(self):
        results = await run_tests(EVALS_DIR, threshold=0.6, mock=True)
        multi = next(r for r in results if "multi_turn_conversation" in r.case_name)

        assert multi.passed is True
        assert multi.metadata["turns"] == 2
        assert len(multi.tool_calls_made) == 1  # get_weather called once, in turn 1
