"""Multi-tool orchestration scenarios (category: multi_tool).

Each scenario chains one or more tools. Only tools present in
``configs/e2e_full.yaml`` are referenced:
  calculator, list_files, read_file, write_file, grep_search, glob_find,
  memory_store, memory_recall, task_create, task_list, task_update,
  git_status, git_log.

Tool-usage assertions are substring matches against the ``tool_name`` field of
``tool_call`` events. When a turn explicitly requests a tool ("use the
calculator"), gpt-4o-mini in act mode calls it reliably enough to assert on.
"""

from __future__ import annotations

from tests.e2e.framework.scenario import Scenario, Turn

SCENARIOS: list[Scenario] = [
    # --- Calculator (5) ---
    Scenario("calc_basic_add", "multi_tool", [
        Turn("Use the calculator tool to compute 123 + 456.", expect_tools=["calculate"], expect_keywords=["579"]),
    ]),
    Scenario("calc_subtract", "multi_tool", [
        Turn("Use the calculator to compute 1000 - 273.", expect_tools=["calculate"], expect_keywords=["727"]),
    ]),
    Scenario("calc_multiply", "multi_tool", [
        Turn("Use the calculator to multiply 47 by 23.", expect_tools=["calculate"], expect_keywords=["1081"]),
    ]),
    Scenario("calc_compound_verify", "multi_tool", [
        Turn("First compute 15% of 8400 with the calculator, then tell me the result.", expect_tools=["calculate"], expect_keywords=["1260"]),
    ]),
    Scenario("calc_multi_step", "multi_tool", [
        Turn("Using the calculator, compute (250 * 4) + 150 - 50 and give the final number.", expect_tools=["calculate"], expect_keywords=["1100"]),
    ]),
    # --- Filesystem (6) ---
    Scenario("fs_write_read", "multi_tool", [
        Turn("Create a file called hello.txt containing the text 'koboi-e2e' using write_file, then read it back with read_file.", expect_tools=["write_file", "read_file"], expect_keywords=["koboi-e2e"]),
    ]),
    Scenario("fs_write_verify_content", "multi_tool", [
        Turn("Write a file called note.txt with the content 'meeting at noon'. Then read it and confirm the content.", expect_tools=["write_file", "read_file"], expect_keywords=["noon"]),
    ]),
    Scenario("fs_list_after_write", "multi_tool", [
        Turn("Write a file called data.txt with content 'abc123'. Then use list_files to show the files in the current directory.", expect_tools=["write_file", "list_files"]),
    ]),
    Scenario("fs_write_two_read_one", "multi_tool", [
        Turn("Write a.txt with 'alpha' and b.txt with 'beta'. Then read a.txt and tell me its content.", expect_tools=["write_file", "read_file"], expect_keywords=["alpha"]),
    ]),
    Scenario("fs_glob_find", "multi_tool", [
        Turn("Write a file called report.md with content 'q1 results'. Then use glob_find to locate files ending in .md.", expect_tools=["write_file", "glob_find"]),
    ]),
    Scenario("fs_grep_search", "multi_tool", [
        Turn("Write a file called log.txt containing the line 'ERROR disk full'. Then use grep_search to search for 'ERROR' in the current directory.", expect_tools=["write_file", "grep_search"], expect_keywords=["ERROR", "disk"]),
    ]),
    # --- Memory tool (4) ---
    Scenario("mem_store_recall", "multi_tool", [
        Turn("Use memory_store to store key 'city' with value 'Reykjavik'. Then use memory_recall to recall key 'city'.", expect_tools=["memory_store", "memory_recall"], expect_keywords=["Reykjavik"]),
    ]),
    Scenario("mem_store_two_recall", "multi_tool", [
        Turn("Store key 'color'='indigo' and key 'animal'='fennec fox' using memory_store. Then recall 'animal'.", expect_tools=["memory_recall"], expect_keywords=["fennec"]),
    ]),
    Scenario("mem_recall_missing_then_store", "multi_tool", [
        Turn("Try to memory_recall the key 'phantom_key' (it may not exist). Then store 'phantom_key'='found' and recall it again.", expect_tools=["memory_recall", "memory_store"], expect_keywords=["found"]),
    ]),
    Scenario("mem_store_numeric", "multi_tool", [
        Turn("Store key 'code' with value '7331' using memory_store, then recall it and report the value.", expect_tools=["memory_store", "memory_recall"], expect_keywords=["7331"]),
    ]),
    # --- Task tool (4) ---
    Scenario("task_create_list", "multi_tool", [
        Turn("Use task_create to create a task titled 'Prepare quarterly report'. Then use task_list to list all tasks.", expect_tools=["task_create", "task_list"], expect_keywords=["quarterly", "report"]),
    ]),
    Scenario("task_create_update", "multi_tool", [
        Turn("Create a task titled 'Review pull request' with task_create. Then use task_update to mark it as completed. Finally list tasks.", expect_tools=["task_create", "task_update", "task_list"]),
    ]),
    Scenario("task_three_then_list", "multi_tool", [
        Turn("Create three tasks: 'buy milk', 'buy eggs', 'buy flour'. Then list all tasks and count them.", expect_tools=["task_create", "task_list"], expect_keywords=["milk", "eggs"]),
    ]),
    Scenario("task_create_priority", "multi_tool", [
        Turn("Create a task titled 'Fix production bug' and another 'Write docs'. Then list tasks.", expect_tools=["task_create", "task_list"], expect_keywords=["production", "docs"]),
    ]),
    # --- Git (2) ---
    Scenario("git_status", "multi_tool", [
        Turn("Run git_status and summarize the current repository state.", expect_tools=["git_status"]),
    ]),
    Scenario("git_status_log", "multi_tool", [
        Turn("Run git_status, then run git_log and summarize recent commits.", expect_tools=["git_status", "git_log"]),
    ]),
    # --- Mixed orchestration (9) ---
    Scenario("mixed_calc_then_file", "multi_tool", [
        Turn("Use the calculator to compute 88 * 12. Then write the result to a file called result.txt and read it back.", expect_tools=["calculate", "write_file", "read_file"], expect_keywords=["1056"]),
    ]),
    Scenario("mixed_file_then_mem", "multi_tool", [
        Turn("Write a file called token.txt with content 'xyz789'. Then store that same value in memory under key 'token' and recall it.", expect_tools=["write_file", "memory_store", "memory_recall"], expect_keywords=["xyz789"]),
    ]),
    Scenario("mixed_task_calc", "multi_tool", [
        Turn("Create a task titled 'Budget calc'. Then use the calculator to compute 5000 + 1250, and tell me the total.", expect_tools=["task_create", "calculate"], expect_keywords=["6250"]),
    ]),
    Scenario("mixed_mem_task", "multi_tool", [
        Turn("Store in memory key 'task_name'='deploy v2'. Then create a task using that stored name and list tasks.", expect_tools=["memory_store", "memory_recall", "task_create", "task_list"], expect_keywords=["deploy"]),
    ]),
    Scenario("mixed_calc_mem", "multi_tool", [
        Turn("Compute 18 * 25 with the calculator. Store the result in memory under key 'product' and recall it.", expect_tools=["calculate", "memory_store", "memory_recall"], expect_keywords=["450"]),
    ]),
    Scenario("mixed_fs_task", "multi_tool", [
        Turn("Write a file called todo.txt with the line 'ship feature'. Then create a task titled 'ship feature' and list tasks.", expect_tools=["write_file", "task_create", "task_list"], expect_keywords=["feature"]),
    ]),
    Scenario("mixed_git_task", "multi_tool", [
        Turn("Run git_status to see the repo state. Then create a task titled 'review repo changes' and list tasks.", expect_tools=["git_status", "task_create", "task_list"], expect_keywords=["review"]),
    ]),
    Scenario("mixed_grep_calc", "multi_tool", [
        Turn("Write a file nums.txt with the line 'total 42'. Use grep_search to find 'total'. Then compute 42 * 100 with the calculator.", expect_tools=["write_file", "grep_search", "calculate"], expect_keywords=["4200"]),
    ]),
    Scenario("mixed_all_four", "multi_tool", [
        Turn("Compute 9 * 9 with the calculator. Write the result to square.txt. Store it in memory under key 'sq'. Then recall it.", expect_tools=["calculate", "write_file", "memory_store", "memory_recall"], expect_keywords=["81"]),
    ]),
]
