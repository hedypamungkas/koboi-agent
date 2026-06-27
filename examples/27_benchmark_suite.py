"""examples/27_benchmark_suite.py -- Multi-framework benchmark suite runner.

Demonstrates how to use koboi's eval system with all 5 benchmarking frameworks:
BFCL (tool calling), RAGAS (RAG quality), GAIA (general tasks),
SWE-bench (coding), and DeepEval (agentic metrics).

Usage:
    python examples/27_benchmark_suite.py
    python examples/27_benchmark_suite.py --framework bfcl
    python examples/27_benchmark_suite.py --framework all --parallel
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

from koboi import KoboiAgent
from koboi.eval import (
    EvalCase,
    EvalRunner,
    RegressionTracker,
    ScorerRegistry,
    LoaderRegistry,
)


async def run_bfcl_suite(agent_config: str, max_cases: int = 10) -> list:
    """Run BFCL tool-calling benchmark using real BFCL v4 dataset."""
    from koboi.eval.loaders.bfcl_loader import BFCLLoader

    print("\n=== BFCL: Tool Calling Accuracy ===")

    # Load real BFCL v4 data from benchmarks/bfcl/
    bfcl_dir = Path(__file__).parent.parent / "benchmarks" / "bfcl"
    if not bfcl_dir.exists():
        print(f"BFCL data not found at {bfcl_dir}")
        print("Download from: https://github.com/ShishirPatil/gorilla")
        return []

    loader = BFCLLoader()
    cases = await loader.load(
        str(bfcl_dir),
        categories=["simple_python"],  # Start with simple_python category
        max_cases=max_cases,
    )
    print(f"Loaded {len(cases)} BFCL cases")

    def harness_factory():
        return KoboiAgent.from_config(agent_config)

    scorers = ScorerRegistry.from_config(
        [
            {"name": "tool_calling_accuracy"},
            {"name": "cost"},
        ]
    )

    runner = EvalRunner(harness_factory=harness_factory, scorers=scorers, threshold=0.7)
    results = await runner.run_suite(cases, parallel=False)
    print(runner.format_results(results, threshold=0.7))
    return results


async def run_ragas_suite(agent_config: str, doc_paths: list[str]) -> list:
    """Run RAGAS RAG quality benchmark."""
    print("\n=== RAGAS: RAG Quality ===")

    def harness_factory():
        return KoboiAgent.from_config(agent_config)

    # Use faithfulness scorer (answer_relevancy needs embeddings endpoint not available on proxy)
    scorers = ScorerRegistry.from_config(
        [
            {"name": "ragas_faithfulness"},
        ]
    )

    # Multi-case RAGAS benchmark covering different RAG quality dimensions
    cases = [
        # 1. Simple factual retrieval (single doc)
        EvalCase(
            name="rag_factual_single",
            user_message="What are the working hours at Acme Corp?",
            expected_answer="Monday to Friday 08:00-17:00, with lunch break 12:00-13:00. Saturday and Sunday are off.",
            context_docs=[
                "Working Hours: Monday - Friday: 08:00 - 17:00, Lunch break: 12:00 - 13:00, Saturday - Sunday: Off.",
            ],
            tags=["ragas", "factual"],
            metadata={"framework": "ragas", "difficulty": "easy"},
        ),
        # 2. Cross-document reasoning (policy + handbook)
        EvalCase(
            name="rag_cross_doc_benefits",
            user_message="What employee benefits does Acme Corp offer and how do they relate to the remote work policy?",
            expected_answer="Acme Corp offers health insurance for employee and immediate family, 401(k) with company match, annual performance bonus, training sponsorship, 12 days PTO, 12 weeks parental leave, and employee assistance program. Remote work is allowed up to 2 days per week with supervisor approval, and employees get a $50/month internet allowance for remote work days. Core hours 10:00-15:00 apply.",
            context_docs=[
                "Employee Benefits: Health insurance (employee + immediate family), 401(k) retirement plan with company match, Annual performance bonus, Training and certification sponsorship, Flexible working hours (core hours: 10:00-15:00), Remote work: 2 days per week, Paid time off: 12 days annually, Parental leave: 12 weeks, Employee assistance program.",
                "Remote Work Policy: Employees may work remotely up to 2 days per week, Prior approval from direct supervisor required, Must be available during core hours (10:00 - 15:00), Company equipment must be used for work purposes only.",
                "Allowances: Meal allowance $150/month, Transportation allowance $100/month, Health insurance coverage for employee and immediate family, Internet allowance $50/month for remote work days.",
            ],
            tags=["ragas", "cross-doc", "synthesis"],
            metadata={"framework": "ragas", "difficulty": "hard"},
        ),
        # 3. Numeric/specific detail extraction
        EvalCase(
            name="rag_numeric_detail",
            user_message="How much does AcmeERP Enterprise cost and what is the minimum number of users?",
            expected_answer="AcmeERP Enterprise costs $15,000/year with a perpetual license and requires a minimum of 10 users.",
            context_docs=[
                "AcmeERP Enterprise: Enterprise Resource Planning system for mid-sized and large companies. Price: $15,000/year (perpetual license). Features: Accounting, HR, Inventory, Sales, Purchasing, CRM. Minimum users: 10 users. Support: 24/7 email and phone support. Deployment: Cloud or on-premise.",
            ],
            tags=["ragas", "numeric", "product"],
            metadata={"framework": "ragas", "difficulty": "medium"},
        ),
        # 4. Multi-hop: career level + performance review
        EvalCase(
            name="rag_multihop_career",
            user_message="What feedback mechanism is used for senior-level employees at Acme Corp?",
            expected_answer="Senior roles (Level 3: 5-8 years experience) use 360-degree feedback as part of their performance review, along with quarterly check-ins with manager and annual performance review.",
            context_docs=[
                "Career Levels: Level 1: Junior (0-2 years), Level 2: Mid-level (2-5 years), Level 3: Senior (5-8 years), Level 4: Lead (8+ years), Level 5: Manager (as needed).",
                "Performance Review: Quarterly check-ins with manager, Annual performance review, 360-degree feedback for senior roles, Performance-linked compensation adjustments.",
            ],
            tags=["ragas", "multi-hop", "hr"],
            metadata={"framework": "ragas", "difficulty": "hard"},
        ),
        # 5. Faithfulness test: answer should NOT include info not in context
        EvalCase(
            name="rag_faithfulness_boundary",
            user_message="What is the resignation notice period and what happens to unused annual leave?",
            expected_answer="The resignation notice period is 30 days. Regarding unused annual leave, the policy states that unused leave can be carried forward maximum 3 days to the following year. The context does not specify what happens to remaining unused leave upon resignation.",
            context_docs=[
                "Resignation Procedure: Notice period: 30 days, Document and access handover, Exit interview with HR, Company asset return, Final settlement processing.",
                "Annual Leave: Permanent employees: 12 days per year, Contract employees: 6 days per year, Unused leave can be carried forward maximum 3 days to the following year.",
            ],
            tags=["ragas", "faithfulness", "boundary"],
            metadata={"framework": "ragas", "difficulty": "hard"},
        ),
        # 6. Product comparison across multiple docs
        EvalCase(
            name="rag_product_comparison",
            user_message="Compare the pricing models of AcmePOS Professional and AcmeCRM Business.",
            expected_answer="AcmePOS Professional is $500/month as a SaaS subscription with minimum 1 outlet. AcmeCRM Business is $25/user/month with minimum 5 users. POS is outlet-based pricing while CRM is per-user pricing.",
            context_docs=[
                "AcmePOS Professional: Point of Sale system. Price: $500/month (SaaS subscription). Features: Cashier, Inventory Management, Reporting, Multi-location. Minimum: 1 outlet.",
                "AcmeCRM Business: Customer Relationship Management. Price: $25/user/month. Features: Pipeline management, Lead tracking, Email integration, Analytics. Minimum users: 5 users.",
            ],
            tags=["ragas", "comparison", "product"],
            metadata={"framework": "ragas", "difficulty": "medium"},
        ),
    ]

    runner = EvalRunner(harness_factory=harness_factory, scorers=scorers, threshold=0.7)
    results = await runner.run_suite(cases, parallel=False)
    print(runner.format_results(results, threshold=0.7))
    return results


async def run_gaia_suite(agent_config: str, max_cases: int = 5) -> list:
    """Run GAIA general task benchmark."""
    print("\n=== GAIA: General Task Completion ===")

    def harness_factory():
        return KoboiAgent.from_config(agent_config)

    scorers = ScorerRegistry.from_config(
        [
            {"name": "gaia_verification", "numeric_tolerance": 0.01},
            {"name": "cost"},
        ]
    )

    # Sample GAIA-style cases
    cases = [
        EvalCase(
            name="gaia_sample_0",
            user_message="What is 2 + 2?",
            expected_answer="4",
            tags=["gaia", "level-1"],
            metadata={"level": 1, "framework": "gaia"},
        ),
        EvalCase(
            name="gaia_sample_1",
            user_message="How many continents are there on Earth?",
            expected_answer="7",
            tags=["gaia", "level-1"],
            metadata={"level": 1, "framework": "gaia"},
        ),
    ][:max_cases]

    runner = EvalRunner(harness_factory=harness_factory, scorers=scorers, threshold=0.7)
    results = await runner.run_suite(cases, parallel=False)
    print(runner.format_results(results, threshold=0.7))
    return results


async def run_swe_bench_suite(agent_config: str, max_cases: int = 5) -> list:
    """Run SWE-bench coding benchmark."""
    print("\n=== SWE-bench: Coding Agent ===")

    # SWE-bench requires a coding agent with diff-generation prompt
    swe_config = str(Path(__file__).parent.parent / "configs" / "swe_bench.yaml")
    config_to_use = swe_config if Path(swe_config).exists() else agent_config

    def harness_factory():
        return KoboiAgent.from_config(config_to_use)

    scorers = ScorerRegistry.from_config(
        [
            {"name": "patch_generation"},
            {"name": "cost", "max_tokens": 100000},
        ]
    )

    # Sample SWE-bench-style cases
    cases = [
        EvalCase(
            name="swe_sample_0",
            user_message="Fix the bug in the calculate_total function that returns None when the list is empty.",
            expected_answer="""diff --git a/utils.py b/utils.py
--- a/utils.py
+++ b/utils.py
@@ -10,3 +10,5 @@
 def calculate_total(items):
+    if not items:
+        return 0
     return sum(items)""",
            max_iterations=30,
            tags=["swe-bench", "coding"],
            metadata={"framework": "swe-bench"},
        ),
    ][:max_cases]

    runner = EvalRunner(harness_factory=harness_factory, scorers=scorers, threshold=0.5)
    results = await runner.run_suite(cases, parallel=False)
    print(runner.format_results(results, threshold=0.5))
    return results


async def run_all_suites(agent_config: str, parallel: bool = False, max_cases: int = 5) -> list:
    """Run all benchmark suites."""
    all_results = []

    all_results.extend(await run_bfcl_suite(agent_config, max_cases))
    all_results.extend(await run_ragas_suite(agent_config, []))
    all_results.extend(await run_gaia_suite(agent_config, max_cases))
    all_results.extend(await run_swe_bench_suite(agent_config, max_cases))

    # Summary
    total = len(all_results)
    passed = sum(1 for r in all_results if r.passed)
    avg = sum(r.overall_score for r in all_results) / total if total else 0
    print(f"\n{'=' * 60}")
    print(f"TOTAL: {passed}/{total} passed — Average: {avg:.1%}")
    print(f"{'=' * 60}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Koboi Agent Benchmark Suite")
    parser.add_argument("--config", default="configs/simple_chat.yaml", help="Agent config YAML")
    parser.add_argument("--framework", default="bfcl", choices=["bfcl", "ragas", "gaia", "swe-bench", "all"])
    parser.add_argument("--parallel", action="store_true", help="Run cases in parallel")
    parser.add_argument("--max-cases", type=int, default=5, help="Max cases per suite")
    parser.add_argument("--save-baseline", action="store_true", help="Save results as baseline")
    args = parser.parse_args()

    print("Koboi Agent Benchmark Suite")
    print(f"Config: {args.config}")
    print(f"Framework: {args.framework}")
    print(f"Available scorers: {ScorerRegistry.list_available()}")
    print(f"Available loaders: {LoaderRegistry.list_available()}")

    if args.framework == "all":
        results = asyncio.run(run_all_suites(args.config, args.parallel, args.max_cases))
    elif args.framework == "bfcl":
        results = asyncio.run(run_bfcl_suite(args.config, args.max_cases))
    elif args.framework == "ragas":
        results = asyncio.run(run_ragas_suite(args.config, []))
    elif args.framework == "gaia":
        results = asyncio.run(run_gaia_suite(args.config, args.max_cases))
    elif args.framework == "swe-bench":
        results = asyncio.run(run_swe_bench_suite(args.config, args.max_cases))
    else:
        print(f"Unknown framework: {args.framework}")
        sys.exit(1)

    # Save baseline if requested
    if args.save_baseline and results:
        tracker = RegressionTracker()
        tracker.save_baseline(args.framework, results)
        print(f"\nBaseline saved to {tracker.baseline_dir}/{args.framework}.json")


if __name__ == "__main__":
    main()
