"""Coding-harness demo: the agent applies a unified-diff patch, real tests run.

Wave 2.4 end-to-end proof for ``apply_patch``, fully offline (``--mock``, no API
key, no network): a tiny buggy project is generated in a temp dir at import time;
the scripted LLM reads the file then fixes the bug via ``apply_patch`` (a REAL
tool execution of a unified-diff patch inside a restricted sandbox anchored at
the project); the ``test_suite`` scorer then runs the project's actual unittest
suite in the same workspace and GATES on exit code 0. If the patch did not
genuinely apply to the code on disk, the suite fails and so does the eval -- no
text-similarity involved.

Commands use ``sys.executable -m unittest`` (not bare ``pytest``): the restricted
sandbox scrubs env/PATH, so interpreter-module form is the reliable shape.

Run:  koboi eval-test evals/coding_patch.eval.py --mock --strict
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

WS = tempfile.mkdtemp(prefix="koboi-coding-patch-eval-")
atexit.register(shutil.rmtree, WS, ignore_errors=True)
Path(WS, "calc.py").write_text(_BUGGY)
Path(WS, "test_calc.py").write_text(_TEST)

# A single-hunk unified diff: replace the buggy return with the correct one.
# The context line anchors the patch; the @@ line number is advisory.
_PATCH = (
    "@@ -1,2 +1,2 @@\n"
    " def add(a, b):\n"
    "-    return a - b  # BUG: subtracts\n"
    "+    return a + b\n"
)

CONFIG = {
    "agent": {
        "name": "coding-patch-eval",
        "description": "Eval probe: apply_patch fix gated by the real test suite",
        "system_prompt": "You are a coding agent. Fix bugs by applying unified-diff patches with apply_patch.",
        "mode": "act",  # chat/plan would block apply_patch
        "max_iterations": 6,
    },
    "llm": {
        "provider": "openai",
        "model": "gpt-4o-mini",  # required by config schema; never contacted in mock
        "api_key": "dummy",
    },
    "tools": {"builtin": ["read_file", "apply_patch"]},
    "sandbox": {"backend": "restricted", "workdir": WS},
}

MOCK_RESPONSES = [
    scripted_response(None, [scripted_tool_call("read_file", {"path": "calc.py"})]),
    scripted_response(
        None,
        [scripted_tool_call("apply_patch", {"path": "calc.py", "patch": _PATCH})],
    ),
    scripted_response("Fixed: add() now returns a + b. The test suite should pass."),
]
TAGS = ["smoke", "coding", "harness"]


async def test_agent_patches_bug_and_suite_passes(t):
    """The scripted patch is applied for real; the actual unittest suite gates."""
    await t.send("add() subtracts instead of adding; fix calc.py with apply_patch.")
    t.calledTool("apply_patch")  # gate: the patch tool really executed
    await t.judge(
        "test_suite",
        severity=Severity.GATE,
        min_score=1.0,
        test_command=f"{sys.executable} -m unittest discover -q",
        workspace=WS,
    )
    t.completed()
