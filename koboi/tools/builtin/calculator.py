"""koboi/tools/builtin/calculator -- Mathematical expression evaluator."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from koboi.tools.registry import tool

# Cap the bit-length of any ``**``/``pow()`` integer result so a pathological
# exponent (e.g. ``9 ** 9 ** 8``) cannot hang/OOM the process. 32768 bits is
# ~9864 decimal digits -- generous for real math, far below the DoS threshold.
_MAX_POW_RESULT_BITS = 1 << 15


def _safe_pow(base, exp, mod=None):
    """``operator.pow`` with a bound on integer-result size (issue #47).

    ``**`` is right-associative, so ``9 ** 9 ** 8`` expands the exponent first;
    we therefore reject up-front when ``base.bit_length() * exp`` would exceed
    the cap. ``abs(base) <= 1`` (result is -1/0/1), 3-arg ``pow(b, e, mod)``
    (modular exponentiation, bounded by ``mod``), floats, and negative
    exponents are always cheap and left to ``pow``/``operator.pow``.
    """
    if mod is not None:
        return pow(base, exp, mod)
    if isinstance(base, int) and isinstance(exp, int) and exp >= 0 and abs(base) > 1:
        if base.bit_length() * exp > _MAX_POW_RESULT_BITS:
            raise ValueError("exponentiation result too large")
    return operator.pow(base, exp)


_BINOPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: _safe_pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARYOPS: dict[type, Callable[[Any], Any]] = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval_node(node, names: dict):
    """Recursively evaluate a whitelisted math AST node.

    No ``eval``/``compile``: only arithmetic, constants, whitelisted names, and
    calls to whitelisted functions are handled; anything else raises. This makes
    the calculator safe by construction rather than by post-filtering.
    """
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, names)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left, names), _eval_node(node.right, names))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_node(node.operand, names))
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        raise ValueError(f"Unknown name: {node.id}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in names:
            raise ValueError("Only whitelisted math functions may be called")
        args = [_eval_node(a, names) for a in node.args]
        return names[node.func.id](*args)
    raise ValueError(f"Disallowed expression element: {type(node).__name__}")


def _safe_eval(expression: str, names: dict) -> float:
    """Parse and evaluate a math expression without eval()."""
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree, names)


@tool(
    name="calculate",
    group="math",
    description="Calculate mathematical expressions. Supports basic operations (+, -, *, /, **), math functions (sqrt, sin, cos, log, etc), and constants (pi, e).",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression, e.g. '2 + 3 * 4' or 'sqrt(144)'",
            },
        },
        "required": ["expression"],
    },
)
def calculate(expression: str) -> str:
    safe_names = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "pi": math.pi,
        "e": math.e,
        "pow": _safe_pow,
        "ceil": math.ceil,
        "floor": math.floor,
    }
    try:
        result = _safe_eval(expression, safe_names)
        return f"{expression} = {result}"
    except Exception as e:
        # MemoryError must not be masked as a normal "Error" string; the
        # _safe_pow bound is the primary protection, but re-raise defensively
        # so a future unbounded path can never silently swallow an OOM.
        if isinstance(e, MemoryError):
            raise
        return f"Error calculating '{expression}': {e}"
