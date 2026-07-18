"""Coding-harness demo: the agent really edits a file, real tests really run.

Wave 1 end-to-end proof, fully offline (`--mock`, no API key, no network):
a tiny buggy project is generated in a temp dir at import time; the scripted
LLM reads the file then fixes the bug via ``edit_file`` (a REAL tool execution
inside a restricted sandbox anchored at the project); the ``test_suite``
scorer then runs the project's actual unittest suite in the same workspace and
GATES on exit code 0. If the edit did not genuinely fix the code on disk, the
suite fails and so does the eval -- no text-similarity involved.

Commands use ``sys.executable -m unittest`` (not bare ``pytest``): the
restricted sandbox scrubs env/PATH, so interpreter-module form is the reliable
shape for test commands.

Run:  koboi eval-test evals/coding_fix.eval.py --mock --strict
"""

import atexit
import shutil
import sys
import tempfile
from pathlib import Path

from koboi.eval.t import Severity, scripted_response, scripted_tool_call

_BUGGY = "def add(a, b):\n    return a - b  # BUG: subtracts\n"
_TEST = (
    "import unittest\n"
    "from calc import add\n\n\n"
    "class TestAdd(unittest.TestCase):\n"
    "    def test_add(self):\n"
    "        self.assertEqual(add(2, 3), 5)\n\n\n"
    'if __name__ == "__main__":\n'
    "    unittest.main()\n"
)

WS = tempfile.mkdtemp(prefix="koboi-coding-fix-eval-")
atexit.register(shutil.rmtree, WS, ignore_errors=True)
Path(WS, "calc.py").write_text(_BUGGY)
Path(WS, "test_calc.py").write_text(_TEST)

CONFIG = {
    "agent": {
        "name": "coding-fix-eval",
        "description": "Eval probe: edit_file fix gated by the real test suite",
        "system_prompt": "You are a coding agent. Fix bugs using edit_file.",
        "mode": "act",  # chat/plan would block edit_file
        "max_iterations": 6,
    },
    "llm": {
        "provider": "openai",
        "model": "gpt-4o-mini",  # required by config schema; never contacted in mock
        "api_key": "dummy",
    },
    "tools": {"builtin": ["read_file", "edit_file"]},
    "sandbox": {"backend": "restricted", "workdir": WS},
}

MOCK_RESPONSES = [
    scripted_response(None, [scripted_tool_call("read_file", {"path": "calc.py"})]),
    scripted_response(
        None,
        [
            scripted_tool_call(
                "edit_file",
                {
                    "path": "calc.py",
                    "old_string": "return a - b  # BUG: subtracts",
                    "new_string": "return a + b",
                },
            )
        ],
    ),
    scripted_response("Fixed: add() now returns a + b. The test suite should pass."),
]
TAGS = ["smoke", "coding", "harness"]


async def test_agent_fixes_bug_and_suite_passes(t):
    """The scripted fix is applied for real; the actual unittest suite gates."""
    await t.send("add() subtracts instead of adding; fix calc.py.")
    t.calledTool("edit_file")  # gate: the edit tool really executed
    await t.judge(
        "test_suite",
        severity=Severity.GATE,
        min_score=1.0,
        test_command=f"{sys.executable} -m unittest discover -q",
        workspace=WS,
    )
    t.completed()
