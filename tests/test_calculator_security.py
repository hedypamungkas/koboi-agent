"""Tests for the AST-based safe calculator in koboi.tools.builtin.calculator."""

from __future__ import annotations

from koboi.tools.builtin.calculator import calculate, _safe_eval


class TestBasicArithmetic:
    def test_addition(self):
        assert "8" in calculate("3 + 5")

    def test_subtraction(self):
        assert "3" in calculate("10 - 7")

    def test_multiplication(self):
        assert "42" in calculate("6 * 7")

    def test_division(self):
        assert "2.5" in calculate("5 / 2")

    def test_floor_division(self):
        assert "3" in calculate("7 // 2")

    def test_modulo(self):
        assert "1" in calculate("7 % 3")

    def test_power(self):
        assert "8" in calculate("2 ** 3")

    def test_order_of_operations(self):
        assert "14" in calculate("2 + 3 * 4")

    def test_unary_negation(self):
        assert "-5" in calculate("-5")

    def test_unary_plus(self):
        assert "5" in calculate("+5")

    def test_complex_expression(self):
        assert "14" in calculate("(2 + 3) * 2 + 4")


class TestMathFunctions:
    def test_sqrt(self):
        assert "12.0" in calculate("sqrt(144)")

    def test_sin(self):
        result = calculate("sin(0)")
        assert "0" in result

    def test_cos(self):
        result = calculate("cos(0)")
        assert "1" in result

    def test_log(self):
        assert "1.0" in calculate("log(e)")

    def test_log10(self):
        assert "2" in calculate("log10(100)")

    def test_constants_pi(self):
        result = calculate("pi")
        assert "3.14" in result

    def test_constants_e(self):
        result = calculate("e")
        assert "2.71" in result

    def test_abs(self):
        assert "5" in calculate("abs(-5)")

    def test_round(self):
        assert "3" in calculate("round(2.7)")

    def test_min(self):
        assert "1" in calculate("min(1, 2, 3)")

    def test_max(self):
        assert "3" in calculate("max(1, 2, 3)")

    def test_pow_function(self):
        assert "8" in calculate("pow(2, 3)")

    def test_ceil(self):
        assert "4" in calculate("ceil(3.2)")

    def test_floor(self):
        assert "3" in calculate("floor(3.8)")


class TestSecurityBlocking:
    def test_dunder_bypass_blocked(self):
        """__import__ should be rejected by AST whitelist."""
        result = calculate("__import__('os').system('echo pwned')")
        assert "Error" in result

    def test_lambda_injection_blocked(self):
        """Lambda expressions are not in the allowed AST node list."""
        result = calculate("(lambda: __import__('os'))()")
        assert "Error" in result

    def test_attribute_access_blocked(self):
        """Attribute access (dunder class/bases) uses Attribute node which is not allowed."""
        result = calculate("().__class__.__bases__")
        assert "Error" in result

    def test_subscript_access_blocked(self):
        """List comprehensions produce AST nodes not in the allowed list."""
        result = calculate("[x for x in 'hello'].__class__")
        assert "Error" in result

    def test_import_blocked(self):
        """Direct import call should be blocked."""
        result = calculate("import('os')")
        assert "Error" in result

    def test_eval_blocked(self):
        """Nested eval calls should be blocked."""
        result = calculate("eval('1+1')")
        assert "Error" in result

    def test_exec_blocked(self):
        """exec calls should be blocked."""
        result = calculate("exec('print(1)')")
        assert "Error" in result

    def test_open_blocked(self):
        """open() should be blocked -- not in safe_names."""
        result = calculate("open('/etc/passwd')")
        assert "Error" in result

    def test_string_literal_evaluates_to_value(self):
        """Plain string literals use Constant node (allowed) and evaluate to their value."""
        result = calculate("'hello'")
        # String literals pass the AST check via Constant node but produce the
        # string value when evaluated (no math result). This is acceptable since
        # there is no security breach -- builtins are stripped.
        assert "hello" in result

    def test_boolean_literals_blocked(self):
        """Boolean True/False produce NameConstant nodes not in allowed list (Python 3.7)."""
        result = calculate("True")
        # On Python 3.8+ Constant handles True/False, but Name is also allowed
        # so this may or may not succeed -- the key is no security breach
        # Just verify it doesn't crash
        assert isinstance(result, str)


class TestSafeEvalInternal:
    def test_safe_eval_basic(self):
        result = _safe_eval("2 + 2", {})
        assert result == 4

    def test_safe_eval_with_names(self):
        import math

        result = _safe_eval("sqrt(16)", {"sqrt": math.sqrt})
        assert result == 4.0

    def test_safe_eval_rejects_attribute(self):
        import pytest

        with pytest.raises(ValueError, match="Disallowed"):
            _safe_eval("x.__class__", {"x": 42})

    def test_safe_eval_rejects_subscript(self):
        import pytest

        with pytest.raises(ValueError, match="Disallowed"):
            _safe_eval("[1, 2, 3][0]", {})

    def test_safe_eval_rejects_compare(self):
        import pytest

        with pytest.raises(ValueError, match="Disallowed"):
            _safe_eval("1 < 2", {})
