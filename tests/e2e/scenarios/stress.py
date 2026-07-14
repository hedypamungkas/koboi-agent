"""Stress / concurrency scenarios (category: stress).

Two shapes:
  * ``concurrent=N`` — run the first turn on N independent sessions in parallel
    (tests session isolation + pool capacity + provider concurrency).
  * long multi-turn (10–15 turns) — exercises memory growth and, near the
    ``max_context_tokens`` window, context compaction.

Concurrency is modest (3–8) to stay under the provider's rate ceiling; the
executor's one-retry catches transient provider blips. Exotic flows that need
custom orchestration (pool-cap 429, evict/recreate, create/delete cycles,
mixed chat+job in one call) live in ``test_scenarios.py`` as dedicated tests.
"""

from __future__ import annotations

from tests.e2e.framework.scenario import Scenario, Turn

SCENARIOS: list[Scenario] = [
    # --- Concurrency ---
    Scenario(
        "stress_concurrent_5_sessions",
        "stress",
        [
            Turn(
                "In one short sentence, greet me and name a primary color.",
                expect_keywords=["blue", "red", "green", "yellow"],
            )
        ],
        concurrent=5,
        throttle_seconds=0.5,
    ),
    Scenario(
        "stress_concurrent_3_jobs",
        "stress",
        [
            Turn(
                "Use the calculator to compute 50 + 50 and report the total.",
                expect_tools=["calculate"],
                expect_keywords=["100"],
            )
        ],
        job=True,
        concurrent=3,
        throttle_seconds=0.5,
    ),
    Scenario(
        "stress_concurrent_8_sessions",
        "stress",
        [
            Turn(
                "Reply with exactly one word: a type of fruit.",
                expect_keywords=["apple", "banana", "orange", "mango", "pear", "grape"],
            )
        ],
        concurrent=8,
        throttle_seconds=0.5,
        timeout_per_turn=200,
    ),
    Scenario(
        "stress_concurrent_4_tools",
        "stress",
        [
            Turn(
                "Use the calculator to compute 6 * 7 and report the product.",
                expect_tools=["calculate"],
                expect_keywords=["42"],
            )
        ],
        concurrent=4,
        throttle_seconds=0.5,
    ),
    # --- Rapid-fire serialized (same session, low throttle) ---
    Scenario(
        "stress_rapid_3turn",
        "stress",
        [
            Turn("Use the calculator to compute 10 + 10.", expect_tools=["calculate"], expect_keywords=["20"]),
            Turn("Now compute 20 + 20.", expect_tools=["calculate"], expect_keywords=["40"]),
            Turn("Finally compute 40 + 40.", expect_tools=["calculate"], expect_keywords=["80"]),
        ],
        throttle_seconds=0.0,
    ),
    Scenario(
        "stress_rapid_5turn_memory",
        "stress",
        [
            Turn("My lucky number is 777."),
            Turn("My favorite season is autumn."),
            Turn("My pet's name is Pixel."),
            Turn("What's my lucky number?", expect_keywords=["777"]),
            Turn("And my pet's name?", expect_keywords=["Pixel"]),
        ],
        throttle_seconds=0.0,
    ),
    # --- Tool burst (one message, several tool calls) ---
    Scenario(
        "stress_tool_burst_calc",
        "stress",
        [
            Turn(
                "Use the calculator to compute each of these and list the results: 5+5, 6*6, 100-37, 81/9, 3**4.",
                expect_tools=["calculate"],
            )
        ],
    ),
    Scenario(
        "stress_tool_burst_files",
        "stress",
        [
            Turn(
                "Write three files: a.txt='1', b.txt='2', c.txt='3' using write_file, then use list_files to show them.",
                expect_tools=["write_file", "list_files"],
            )
        ],
    ),
    # --- Long conversations (memory growth / compaction) ---
    Scenario(
        "stress_long_10turn",
        "stress",
        turns=[
            Turn("Let's plan a week-long trip. Day 1 is for arrival and rest."),
            Turn("Day 2: museum day."),
            Turn("Day 3: hiking."),
            Turn("Day 4: beach."),
            Turn("Day 5: food tour."),
            Turn("Day 6: shopping."),
            Turn("Day 7: departure."),
            Turn("Which day did I assign hiking?", expect_keywords=["Day 3", "3"]),
            Turn("Which day is the food tour?", expect_keywords=["Day 5", "5"]),
            Turn("Give a one-line recap of the whole week.", min_events=2),
        ],
        throttle_seconds=0.5,
    ),
    Scenario(
        "stress_long_15turn",
        "stress",
        turns=[
            *[Turn(f"Project note {i}: task T{i} is owned by person P{i % 3}.") for i in range(1, 11)],
            # Recall the LAST notes (T8–T10) which survive keep_last=10 truncation.
            Turn("Who owns task T10?", expect_keywords=["P1"]),
            Turn("Who owns task T8?", expect_keywords=["P2"]),
            Turn("Summarize how many task notes we logged.", expect_keywords=["10"]),
            Turn("Who owns T9?", expect_keywords=["P0"]),
            Turn("List the owners mentioned for T9 and T10.", min_events=2),
        ],
        throttle_seconds=0.5,
        timeout_per_turn=200,
    ),
]
