"""Tests for koboi.hooks.typecheck_hook (structured-diagnostic enrichment)."""

from __future__ import annotations


from koboi.hooks.chain import HookContext, HookEvent
from koboi.hooks.typecheck_hook import TypecheckHook, _parse


def _ctx(tool_name: str, result: str) -> HookContext:
    return HookContext(
        event=HookEvent.POST_TOOL_USE,
        agent=None,
        iteration=0,
        tool_name=tool_name,
        tool_arguments="{}",
        tool_result=result,
    )


class TestParse:
    def test_ruff_concise(self):
        diags = _parse("src/calc.py:42:5: F841 Local variable 'x' is never used\n")
        assert diags == [
            {
                "file": "src/calc.py",
                "line": 42,
                "col": 5,
                "severity": "error",
                "code": "F841",
                "message": "Local variable 'x' is never used",
            }
        ]

    def test_mypy_error_with_code(self):
        diags = _parse('src/calc.py:42: error: Incompatible return  [return-value]\n')
        assert diags[0]["file"] == "src/calc.py"
        assert diags[0]["line"] == 42
        assert diags[0]["severity"] == "error"
        assert diags[0]["code"] == "return-value"
        assert "Incompatible return" in diags[0]["message"]

    def test_mypy_warning_no_code(self):
        diags = _parse("src/calc.py:7: warning: Module not imported\n")
        assert diags[0]["severity"] == "warning"
        assert "code" not in diags[0]

    def test_pyright_error(self):
        diags = _parse("src/calc.py:42:5 - error: Type int not assignable to str\n")
        assert diags[0]["line"] == 42
        assert diags[0]["col"] == 5
        assert diags[0]["severity"] == "error"

    def test_skips_exit_code_prefix_and_summaries(self):
        result = (
            "[exit code: 1]\n"
            "src/calc.py:42:5: F841 unused\n"
            "Found 1 error.\n"
        )
        diags = _parse(result)
        assert len(diags) == 1  # the [exit code] + summary lines are ignored

    def test_garbage_yields_nothing(self):
        assert _parse("totally unrelated output\nno diagnostics here\n") == []


class TestTypecheckHook:
    async def test_ruff_errors_set_typecheck_failed(self):
        hook = TypecheckHook()
        ctx = _ctx("run_typecheck", "[exit code: 1]\nsrc/calc.py:42:5: F841 unused\n")
        ctx.metadata["tool_error_kind"] = "command_failed"  # what the pipeline set
        ctx = await hook.execute(ctx)
        assert ctx.metadata["tool_error_kind"] == "typecheck_failed"
        diags = ctx.metadata["typecheck_diagnostics"]
        assert diags[0]["file"] == "src/calc.py"
        assert diags[0]["line"] == 42

    async def test_mypy_errors_set_typecheck_failed(self):
        hook = TypecheckHook()
        ctx = _ctx("run_typecheck", "src/calc.py:42: error: bad type  [return-value]\n")
        ctx.metadata["tool_error_kind"] = "command_failed"
        ctx = await hook.execute(ctx)
        assert ctx.metadata["tool_error_kind"] == "typecheck_failed"

    async def test_warnings_only_do_not_set_typecheck_failed(self):
        hook = TypecheckHook()
        ctx = _ctx("run_typecheck", "src/calc.py:7: warning: unused import\n")
        ctx.metadata["tool_error_kind"] = "command_failed"
        ctx = await hook.execute(ctx)
        # Diagnostics are attached (useful) but the kind stays command_failed --
        # a warnings-only run is not a typecheck FAILURE from recovery's view.
        assert ctx.metadata["typecheck_diagnostics"][0]["severity"] == "warning"
        assert ctx.metadata["tool_error_kind"] == "command_failed"

    async def test_non_typecheck_tool_ignored(self):
        hook = TypecheckHook()
        ctx = _ctx("run_shell", "[exit code: 1]\nsrc/calc.py:42:5: F841 unused\n")
        ctx.metadata["tool_error_kind"] = "command_failed"
        ctx = await hook.execute(ctx)
        assert ctx.metadata["tool_error_kind"] == "command_failed"
        assert "typecheck_diagnostics" not in ctx.metadata

    async def test_garbage_output_leaves_ctx_unchanged(self):
        hook = TypecheckHook()
        ctx = _ctx("run_typecheck", "nonsense\n")
        ctx.metadata["tool_error_kind"] = "command_failed"
        ctx = await hook.execute(ctx)
        assert ctx.metadata["tool_error_kind"] == "command_failed"
        assert "typecheck_diagnostics" not in ctx.metadata

    async def test_diagnostics_capped(self):
        hook = TypecheckHook(max_diags=3)
        lines = "".join(f"f.py:{i}:1: E001 msg {i}\n" for i in range(50))
        ctx = _ctx("run_typecheck", lines)
        ctx = await hook.execute(ctx)
        assert len(ctx.metadata["typecheck_diagnostics"]) == 3
