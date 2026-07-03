"""Tests for koboi.eval.t.cli -- `koboi eval-test` exit codes."""

from __future__ import annotations

from click.testing import CliRunner

from koboi.eval.t.cli import eval_test


def _write_eval(path, body):
    path.write_text(body)


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
        result = CliRunner().invoke(eval_test, [str(tmp_path), "--strict", "--mock"])
        assert result.exit_code == 0, result.output

    def test_strict_gate_failure_exits_one(self, tmp_path):
        _write_eval(
            tmp_path / "t.eval.py",
            "from koboi.eval.t import scripted_response\n"
            "MOCK_RESPONSES=[scripted_response('ok')]\n"
            "async def test_fail(t):\n"
            "    await t.send('q')\n"
            "    t.calledTool('missing')\n",
        )
        result = CliRunner().invoke(eval_test, [str(tmp_path), "--strict", "--mock"])
        assert result.exit_code == 1

    def test_no_strict_exits_zero_on_failure(self, tmp_path):
        _write_eval(
            tmp_path / "t.eval.py",
            "from koboi.eval.t import scripted_response\n"
            "MOCK_RESPONSES=[scripted_response('ok')]\n"
            "async def test_fail(t):\n"
            "    await t.send('q')\n"
            "    t.calledTool('missing')\n",
        )
        result = CliRunner().invoke(eval_test, [str(tmp_path), "--mock"])
        assert result.exit_code == 0

    def test_no_tests_exits_two(self, tmp_path):
        _write_eval(tmp_path / "t.eval.py", "async def helper(t):\n    pass\n")
        result = CliRunner().invoke(eval_test, [str(tmp_path), "--mock"])
        assert result.exit_code == 2

    def test_tags_filter_selects_subset(self, tmp_path):
        _write_eval(
            tmp_path / "t.eval.py",
            "from koboi.eval.t import scripted_response\n"
            "TAGS=['smoke']\n"
            "MOCK_RESPONSES=[scripted_response('ok')]\n"
            "async def test_smoke(t):\n"
            "    await t.send('q')\n",
        )
        result = CliRunner().invoke(eval_test, [str(tmp_path), "--mock", "--tags", "smoke"])
        assert result.exit_code == 0
        assert "test_smoke" in result.output
