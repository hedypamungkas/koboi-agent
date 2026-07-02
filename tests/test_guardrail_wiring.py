"""tests/test_guardrail_wiring -- guardrail slot routing must not cross wires.

Regression guard for the bug where a bare ``guardrails.output`` config block
(e.g. ``{detect_sensitive: true}``, no ``name`` key) was normalized with the
input slot's default name ``injection_detector`` -- building an
``InputGuardrail`` into the *output* list. ``InputGuardrail.check`` blocks on
empty input, so on tool-call-only turns (empty assistant output) it clobbered
the response with ``[GUARDRAIL WARNING (InputGuardrail): Input is empty]``.

The output slot must default to ``content_filter`` (``OutputGuardrail``), which
passes on empty output.
"""

from __future__ import annotations

import asyncio

import pytest

from koboi.config import Config
from koboi.facade import _build_guardrails
from koboi.guardrails.input import InputGuardrail
from koboi.guardrails.output import OutputGuardrail


def _build(conf: dict):
    """Return (input_grds, output_grds) for a guardrails config block."""
    input_grds, output_grds, _rl, _audit = _build_guardrails(Config.from_dict(conf, validate=False))
    return input_grds, output_grds


class TestOutputSlotRouting:
    def test_bare_output_block_uses_output_guardrail(self):
        # e2e_full.yaml: guardrails.output: {detect_sensitive: true}
        in_g, out_g = _build({"guardrails": {"output": {"detect_sensitive": True}}})
        assert out_g, "output guardrails should not be empty"
        assert all(isinstance(g, OutputGuardrail) for g in out_g)
        assert not any(isinstance(g, InputGuardrail) for g in out_g)

    def test_bare_input_block_uses_input_guardrail(self):
        # e2e_full.yaml: guardrails.input: {detect_injection: true, max_length: 20000}
        in_g, out_g = _build({"guardrails": {"input": {"detect_injection": True, "max_length": 20000}}})
        assert in_g, "input guardrails should not be empty"
        assert all(isinstance(g, InputGuardrail) for g in in_g)

    def test_output_guardrail_passes_on_empty_output(self):
        # The behaviour that eliminates the clobber on tool-call-only turns.
        grd = OutputGuardrail()
        result = asyncio.get_event_loop().run_until_complete(grd.check(""))
        assert result.passed

    def test_input_guardrail_blocks_on_empty_input(self):
        # Confirms the asymmetry -- InputGuardrail MUST stay out of the output slot.
        grd = InputGuardrail()
        result = asyncio.get_event_loop().run_until_complete(grd.check(""))
        assert not result.passed
        assert "empty" in result.reason.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
