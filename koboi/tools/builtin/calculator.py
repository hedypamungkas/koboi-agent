"""koboi/tools/builtin/calculator -- Mathematical expression evaluator."""
from __future__ import annotations

import ast
import math

from koboi.tools.registry import tool

_ALLOWED_AST_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp,
    ast.Num, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
    ast.Mod, ast.FloorDiv, ast.USub, ast.UAdd,
    ast.Call, ast.Name, ast.Load,
)


def _safe_eval(expression: str, names: dict) -> float:
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(f"Disallowed expression element: {type(node).__name__}")
    return eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, names)


@tool(
    name="calculate",
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
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "log": math.log, "log10": math.log10,
        "pi": math.pi, "e": math.e, "pow": pow,
        "ceil": math.ceil, "floor": math.floor,
    }
    try:
        result = _safe_eval(expression, safe_names)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {e}"
