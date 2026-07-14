"""Autonomous job scenarios (category: jobs).

Each scenario submits a single autonomous job (``scenario.job=True``) and polls
to terminal status. The AutonomousApprovalHandler auto-approves SAFE/MODERATE
tools (calculator, write_file, memory_store, task_create), so these exercise the
job runner end-to-end without HITL. Assertions check the job *completes* and the
result contains the expected content.

Job-specific flows that need multiple submissions (idempotency replay, cancel,
cross-owner) live in ``test_jobs.py`` and ``test_security_edge.py``.
"""

from __future__ import annotations

from tests.e2e.framework.scenario import Scenario, Turn

SCENARIOS: list[Scenario] = [
    # --- Simple autonomous jobs (5) ---
    Scenario(
        "job_simple_greeting",
        "jobs",
        [
            Turn("Say hello and confirm you're an autonomous agent.", expect_keywords=["hello", "agent"]),
        ],
        job=True,
    ),
    Scenario(
        "job_calc_autonomous",
        "jobs",
        [
            Turn(
                "Use the calculator to compute 144 / 12 and report the result.",
                expect_tools=["calculate"],
                expect_keywords=["12"],
            ),
        ],
        job=True,
    ),
    Scenario(
        "job_write_report",
        "jobs",
        [
            Turn(
                "Write a file called summary.txt containing the text 'job complete' and read it back.",
                expect_tools=["write_file"],
                expect_keywords=["complete"],
            ),
        ],
        job=True,
    ),
    Scenario(
        "job_memory_persist",
        "jobs",
        [
            Turn(
                "Store key 'job_status' with value 'success' in memory, then recall it.",
                expect_tools=["memory_store", "memory_recall"],
                expect_keywords=["success"],
            ),
        ],
        job=True,
    ),
    Scenario(
        "job_task_create",
        "jobs",
        [
            Turn(
                "Create a task titled 'Autonomous processing' and list all tasks.",
                expect_tools=["task_create", "task_list"],
                expect_keywords=["Autonomous"],
            ),
        ],
        job=True,
    ),
    # --- Multi-tool jobs (4) ---
    Scenario(
        "job_multi_calc_file",
        "jobs",
        [
            Turn(
                "Compute 25 * 16 with the calculator, write the result to out.txt, then read it back.",
                expect_tools=["calculate", "write_file"],
                expect_keywords=["400"],
            ),
        ],
        job=True,
    ),
    Scenario(
        "job_multi_mem_task",
        "jobs",
        [
            Turn(
                "Store 'ticket'='INC-42' in memory, recall it, then create a task titled 'resolve INC-42'.",
                expect_tools=["memory_store", "memory_recall", "task_create"],
                expect_keywords=["INC-42"],
            ),
        ],
        job=True,
    ),
    Scenario(
        "job_multi_write_grep",
        "jobs",
        [
            Turn(
                "Write log.txt with the line 'WARN high latency', then use grep_search to find 'WARN'.",
                expect_tools=["write_file", "grep_search"],
                expect_keywords=["latency"],
            ),
        ],
        job=True,
    ),
    Scenario(
        "job_multi_chain",
        "jobs",
        [
            Turn(
                "Compute 7 + 8 with the calculator, store that in memory under key 'sum', then recall it.",
                expect_tools=["calculate", "memory_store", "memory_recall"],
                expect_keywords=["15"],
            ),
        ],
        job=True,
    ),
    # --- Longer/reasoning jobs (3) ---
    Scenario(
        "job_plan_summary",
        "jobs",
        [
            Turn(
                "Produce a short 3-step plan for deploying a Python web service, then summarize it in one sentence.",
                expect_keywords=["deploy"],
            ),
        ],
        job=True,
    ),
    Scenario(
        "job_rag_lookup",
        "jobs",
        [
            Turn(
                "What is the nightly rate for the Executive Suite at Grand Plaza Hotel? Report just the rate.",
                expect_keywords=["320"],
            ),
        ],
        job=True,
    ),
    Scenario(
        "job_multi_step_reasoning",
        "jobs",
        [
            Turn(
                "A shop sells 12 apples at $0.50 each and 6 oranges at $0.75 each. Use the calculator to get the total, then state it.",
                expect_tools=["calculate"],
                expect_keywords=["10.5", "10.50"],
            ),
        ],
        job=True,
    ),
    # --- Idempotency-friendly (same deterministic ask) (3) ---
    Scenario(
        "job_deterministic_calc",
        "jobs",
        [
            Turn(
                "Use the calculator to compute 2 ** 10 and report the exact result.",
                expect_tools=["calculate"],
                expect_keywords=["1024"],
            ),
        ],
        job=True,
    ),
    Scenario(
        "job_store_constant",
        "jobs",
        [
            Turn(
                "Store key 'pi' with value '3.14159' in memory and recall it.",
                expect_tools=["memory_store", "memory_recall"],
                expect_keywords=["3.14159"],
            ),
        ],
        job=True,
    ),
    Scenario(
        "job_file_with_timestamp_like_name",
        "jobs",
        [
            Turn(
                "Write a file called manifest.txt with content 'build-2026' and read it to confirm.",
                expect_tools=["write_file", "read_file"],
                expect_keywords=["build-2026"],
            ),
        ],
        job=True,
    ),
]
