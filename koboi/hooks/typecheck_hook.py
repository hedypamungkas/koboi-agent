"""koboi/hooks/typecheck_hook.py -- Enrich run_typecheck output into structured diagnostics.

Infra-band (priority 4): runs before ``FailureClassifierHook`` (5) and the
recovery rungs (doom 50 / handover 50 / reflect 60). Parses ``run_typecheck``'s
ruff/mypy/pyright output into structured ``{file, line, col?, severity, code?,
message}`` dicts on ``ctx.metadata["typecheck_diagnostics"]`` and, when real
errors are present, refines ``ctx.metadata["tool_error_kind"]`` from the
pipeline's ``command_failed`` to ``typecheck_failed`` -- so the classifier tags
``failure_class="transient"`` and ReflectionHook can name the FIRST failing
file:line rather than the generic "try a different approach" critique.

Fail-soft: any parse error leaves ``ctx`` unchanged. A warnings-only run (no
``severity == "error"`` diagnostic) does NOT set ``typecheck_failed`` -- it is a
clean-enough run from the recovery loop's perspective.
"""

from __future__ import annotations

import logging
import re

from koboi.hooks.chain import Hook, HookContext, HookEvent

_logger = logging.getLogger(__name__)

_MAX_DIAGS = 20  # bound the metadata payload; the rest are still in tool_result

# ruff --output-format=concise:  path:line:col: CODE message
_RUFF_RE = re.compile(r"^(?P<file>\S+?):(?P<line>\d+):(?P<col>\d+):\s*(?P<code>\S+)\s+(?P<msg>.*)$")
# mypy output:  path:line: severity: message  [code]   (severity: error|warning|note)
# (N.B. comment deliberately does NOT start with "# mypy:" -- that prefix is an
# inline mypy config directive and would break type-checking.)
_MYPY_RE = re.compile(
    r"^(?P<file>\S+?):(?P<line>\d+):\s*(?P<sev>error|warning|note):\s*"
    r"(?P<msg>.*?)(?:\s+\[(?P<code>[^\]]+)\])?$"
)
# pyright:  path:line:col - severity: message   (severity in error|warning|information)
_PYRIGHT_RE = re.compile(
    r"^(?P<file>\S+?):(?P<line>\d+):(?P<col>\d+)\s*-\s*"
    r"(?P<sev>error|warning|information):\s*(?P<msg>.*)$"
)

_PARSERS = (_RUFF_RE, _MYPY_RE, _PYRIGHT_RE)


class TypecheckHook(Hook):
    """Refine run_typecheck results into structured diagnostics + typecheck_failed."""

    priority = 4

    def __init__(self, tool_names: tuple[str, ...] = ("run_typecheck",), max_diags: int = _MAX_DIAGS) -> None:
        self._tool_names = set(tool_names)
        self._max_diags = max_diags

    def handles(self) -> list[HookEvent]:
        return [HookEvent.POST_TOOL_USE]

    async def execute(self, ctx: HookContext) -> HookContext:
        try:
            if ctx.tool_name not in self._tool_names:
                return ctx
            diags = _parse(ctx.tool_result or "")
            if not diags:
                return ctx
            ctx.metadata["typecheck_diagnostics"] = diags[: self._max_diags]
            if any(d.get("severity") == "error" for d in diags):
                ctx.metadata["tool_error_kind"] = "typecheck_failed"
        except Exception as exc:  # fail-soft: enrichment is observability-only
            _logger.warning("TypecheckHook fail-soft: %s", exc)
        return ctx


def _parse(result: str) -> list[dict]:
    """Parse ruff/mypy/pyright diagnostic lines out of ``result``.

    Tolerant: skips the ``[exit code: N]`` prefix line and any non-matching
    line (summary lines like ``"Found 3 errors"`` are ignored, not errored on).
    """
    diags: list[dict] = []
    for raw in result.splitlines():
        line = raw.rstrip()
        if not line or line.startswith("[exit code"):
            continue
        m = next((p.match(line) for p in _PARSERS if p.match(line)), None)
        if m is None:
            continue
        d = m.groupdict()
        sev = (d.get("sev") or "").lower()
        if not sev:
            # ruff emits a rule CODE but no severity. Derive one so a warnings-only
            # run (W291/W503 pycodestyle warnings) does NOT trip typecheck_failed:
            # ``W*`` -> warning, everything else (E/F/B/UP/...) -> error.
            code = (d.get("code") or "").upper()
            sev = "warning" if code.startswith("W") else "error"
        diag: dict = {
            "file": d.get("file"),
            "line": int(d["line"]) if d.get("line") else None,
            "severity": sev,
            "message": (d.get("msg") or "").strip(),
        }
        if d.get("col"):
            diag["col"] = int(d["col"])
        if d.get("code"):
            diag["code"] = d["code"]
        diags.append(diag)
    return diags
