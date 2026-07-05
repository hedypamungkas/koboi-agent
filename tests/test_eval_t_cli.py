"""Tests for ``koboi eval-test`` exit codes (now via cli_commands.cmd_eval_test).

The click wrapper (:mod:`koboi.eval.t.cli`) was removed when eval-test moved to
the core argparse dispatcher; these tests drive the core handler directly.
"""

from __future__ import annotations

import contextlib
import io

from koboi.cli_commands import cmd_eval_test


def _write_eval(path, body):
    path.write_text(body)


def _run(path, *, strict=False, tags=None):
    """Invoke cmd_eval_test (mock mode) and return (exit_code, stdout)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cmd_eval_test(str(path), None, True, strict, 0.6, False, 5, tags)
    return code, out.getvalue()


class TestEvalTestCli:
    def test_strict_pass_exits_zero(self, tmp_path):
        _write_eval(
            tmp_path / "t.eval.py",
            "from koboi.eval.t import scripted_response, scripted_tool_call\n"
            "MOCK_RESPONSES=[scripted_response(None, [scripted_tool_call('calc')]), scripted_response('4')]\n"
            "async def test_ok(t):\n"
            "    await t.send('q')\n"
            "    t.calledTool('calc')\n",
        )
        code, _ = _run(tmp_path, strict=True)
        assert code == 0

    def test_strict_gate_failure_exits_one(self, tmp_path):
        _write_eval(
            tmp_path / "t.eval.py",
            "from koboi.eval.t import scripted_response\n"
            "MOCK_RESPONSES=[scripted_response('ok')]\n"
            "async def test_fail(t):\n"
            "    await t.send('q')\n"
            "    t.calledTool('missing')\n",
        )
        code, _ = _run(tmp_path, strict=True)
        assert code == 1

    def test_no_strict_exits_zero_on_failure(self, tmp_path):
        _write_eval(
            tmp_path / "t.eval.py",
            "from koboi.eval.t import scripted_response\n"
            "MOCK_RESPONSES=[scripted_response('ok')]\n"
            "async def test_fail(t):\n"
            "    await t.send('q')\n"
            "    t.calledTool('missing')\n",
        )
        code, _ = _run(tmp_path, strict=False)
        assert code == 0

    def test_no_tests_exits_two(self, tmp_path):
        _write_eval(tmp_path / "t.eval.py", "async def helper(t):\n    pass\n")
        code, _ = _run(tmp_path, strict=False)
        assert code == 2

    def test_tags_filter_selects_subset(self, tmp_path):
        _write_eval(
            tmp_path / "t.eval.py",
            "from koboi.eval.t import scripted_response\n"
            "TAGS=['smoke']\n"
            "MOCK_RESPONSES=[scripted_response('ok')]\n"
            "async def test_smoke(t):\n"
            "    await t.send('q')\n",
        )
        code, out = _run(tmp_path, strict=False, tags="smoke")
        assert code == 0
        assert "test_smoke" in out
