"""Independent test cases for builtin tools (WebSearch and Calculator).

This module contains independent verification tests for the builtin tool
implementations, following the detailed design specification in section 3.5.

Test categories:
1. BaseTool interface compliance
2. WebSearch tool functionality
3. Calculator tool - basic arithmetic
4. Calculator tool - advanced operations
5. Calculator tool - safe functions
6. Calculator tool - security (AST sandbox)
7. Calculator tool - error handling
8. Boundary conditions
"""
import pytest
import inspect

from agent_framework.interfaces.base_tool import BaseTool
from agent_framework.tools.builtin.web_search import WebSearch
from agent_framework.tools.builtin.calculator import Calculator, _safe_eval


# ============================================================================
# 1. BaseTool Interface Compliance
# ============================================================================


class TestBaseToolCompliance:
    """Verify both tools correctly implement BaseTool."""

    def test_web_search_inherits_base_tool(self):
        """WebSearch must be a subclass of BaseTool."""
        assert issubclass(WebSearch, BaseTool)

    def test_calculator_inherits_base_tool(self):
        """Calculator must be a subclass of BaseTool."""
        assert issubclass(Calculator, BaseTool)

    def test_web_search_is_instantiable(self):
        """WebSearch can be instantiated."""
        tool = WebSearch()
        assert isinstance(tool, BaseTool)

    def test_calculator_is_instantiable(self):
        """Calculator can be instantiated."""
        tool = Calculator()
        assert isinstance(tool, BaseTool)

    def test_web_search_has_name(self):
        """WebSearch must have name attribute."""
        tool = WebSearch()
        assert hasattr(tool, "name")
        assert isinstance(tool.name, str)

    def test_web_search_has_description(self):
        """WebSearch must have description attribute."""
        tool = WebSearch()
        assert hasattr(tool, "description")
        assert isinstance(tool.description, str)

    def test_calculator_has_name(self):
        """Calculator must have name attribute."""
        tool = Calculator()
        assert hasattr(tool, "name")
        assert isinstance(tool.name, str)

    def test_calculator_has_description(self):
        """Calculator must have description attribute."""
        tool = Calculator()
        assert hasattr(tool, "description")
        assert isinstance(tool.description, str)

    def test_web_search_has_run_method(self):
        """WebSearch must have run method."""
        assert hasattr(WebSearch, "run")

    def test_calculator_has_run_method(self):
        """Calculator must have run method."""
        assert hasattr(Calculator, "run")

    def test_web_search_run_is_async(self):
        """WebSearch.run must be async."""
        assert inspect.iscoroutinefunction(WebSearch.run)

    def test_calculator_run_is_async(self):
        """Calculator.run must be async."""
        assert inspect.iscoroutinefunction(Calculator.run)

    def test_web_search_run_signature(self):
        """WebSearch.run must accept input, session_id, and kwargs."""
        sig = inspect.signature(WebSearch.run)
        params = list(sig.parameters.keys())
        assert "input" in params
        assert "session_id" in params

    def test_calculator_run_signature(self):
        """Calculator.run must accept input, session_id, and kwargs."""
        sig = inspect.signature(Calculator.run)
        params = list(sig.parameters.keys())
        assert "input" in params
        assert "session_id" in params


# ============================================================================
# 2. WebSearch Tool Functionality
# ============================================================================


class TestWebSearch:
    """Test WebSearch tool functionality."""

    def test_web_search_name(self):
        """WebSearch name should be 'web_search'."""
        tool = WebSearch()
        assert tool.name == "web_search"

    def test_web_search_description_not_empty(self):
        """WebSearch description should not be empty."""
        tool = WebSearch()
        assert len(tool.description) > 0

    @pytest.mark.asyncio
    async def test_web_search_returns_string(self):
        """WebSearch.run must return a string."""
        tool = WebSearch()
        result = await tool.run("test query")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_web_search_contains_query(self):
        """WebSearch result must contain the original query."""
        tool = WebSearch()
        result = await tool.run("Python programming")
        assert "Python programming" in result

    @pytest.mark.asyncio
    async def test_web_search_returns_results(self):
        """WebSearch result must contain numbered results."""
        tool = WebSearch()
        result = await tool.run("test")
        assert "1." in result
        assert "2." in result
        assert "3." in result

    @pytest.mark.asyncio
    async def test_web_search_with_session_id(self):
        """WebSearch accepts session_id parameter."""
        tool = WebSearch()
        result = await tool.run("test", session_id="session-123")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_web_search_with_kwargs(self):
        """WebSearch accepts additional kwargs."""
        tool = WebSearch()
        result = await tool.run("test", session_id="s1", extra="value")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_web_search_empty_query_returns_error(self):
        """WebSearch returns error for empty query."""
        tool = WebSearch()
        result = await tool.run("")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_web_search_whitespace_query_returns_error(self):
        """WebSearch returns error for whitespace-only query."""
        tool = WebSearch()
        result = await tool.run("   ")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_web_search_strips_whitespace(self):
        """WebSearch strips leading/trailing whitespace from query."""
        tool = WebSearch()
        result = await tool.run("  Python  ")
        assert "Python" in result

    @pytest.mark.asyncio
    async def test_web_search_unicode_query(self):
        """WebSearch handles Unicode query."""
        tool = WebSearch()
        result = await tool.run("Python tutorial")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_web_search_long_query(self):
        """WebSearch handles long query."""
        tool = WebSearch()
        long_query = "test " * 100
        result = await tool.run(long_query)
        assert isinstance(result, str)


# ============================================================================
# 3. Calculator - Basic Arithmetic
# ============================================================================


class TestCalculatorBasicArithmetic:
    """Test Calculator with basic arithmetic operations."""

    @pytest.mark.asyncio
    async def test_addition(self):
        """Calculator evaluates addition."""
        calc = Calculator()
        result = await calc.run("2 + 3")
        assert result == "5"

    @pytest.mark.asyncio
    async def test_subtraction(self):
        """Calculator evaluates subtraction."""
        calc = Calculator()
        result = await calc.run("10 - 4")
        assert result == "6"

    @pytest.mark.asyncio
    async def test_multiplication(self):
        """Calculator evaluates multiplication."""
        calc = Calculator()
        result = await calc.run("6 * 7")
        assert result == "42"

    @pytest.mark.asyncio
    async def test_division(self):
        """Calculator evaluates division."""
        calc = Calculator()
        result = await calc.run("15 / 3")
        assert result == "5"

    @pytest.mark.asyncio
    async def test_floor_division(self):
        """Calculator evaluates floor division."""
        calc = Calculator()
        result = await calc.run("17 // 5")
        assert result == "3"

    @pytest.mark.asyncio
    async def test_modulo(self):
        """Calculator evaluates modulo."""
        calc = Calculator()
        result = await calc.run("17 % 5")
        assert result == "2"

    @pytest.mark.asyncio
    async def test_power(self):
        """Calculator evaluates exponentiation."""
        calc = Calculator()
        result = await calc.run("2 ** 10")
        assert result == "1024"

    @pytest.mark.asyncio
    async def test_negative_numbers(self):
        """Calculator handles negative numbers."""
        calc = Calculator()
        result = await calc.run("-5 + 3")
        assert result == "-2"

    @pytest.mark.asyncio
    async def test_unary_positive(self):
        """Calculator handles unary positive."""
        calc = Calculator()
        result = await calc.run("+5")
        assert result == "5"

    @pytest.mark.asyncio
    async def test_float_arithmetic(self):
        """Calculator handles float arithmetic."""
        calc = Calculator()
        result = await calc.run("1.5 + 2.5")
        assert result == "4"

    @pytest.mark.asyncio
    async def test_integer_result_format(self):
        """Calculator formats integer results without decimal point."""
        calc = Calculator()
        result = await calc.run("10 / 2")
        assert result == "5"  # Not "5.0"

    @pytest.mark.asyncio
    async def test_float_result_format(self):
        """Calculator keeps decimal point for non-integer results."""
        calc = Calculator()
        result = await calc.run("10 / 3")
        assert "." in result


# ============================================================================
# 4. Calculator - Advanced Operations
# ============================================================================


class TestCalculatorAdvanced:
    """Test Calculator with advanced operations."""

    @pytest.mark.asyncio
    async def test_parentheses(self):
        """Calculator evaluates expressions with parentheses."""
        calc = Calculator()
        result = await calc.run("(2 + 3) * 4")
        assert result == "20"

    @pytest.mark.asyncio
    async def test_nested_parentheses(self):
        """Calculator evaluates nested parentheses."""
        calc = Calculator()
        result = await calc.run("((2 + 3) * (4 - 1))")
        assert result == "15"

    @pytest.mark.asyncio
    async def test_complex_expression(self):
        """Calculator evaluates complex expression."""
        calc = Calculator()
        result = await calc.run("2 + 3 * 4 - 1")
        assert result == "13"  # 2 + 12 - 1

    @pytest.mark.asyncio
    async def test_operator_precedence(self):
        """Calculator respects operator precedence."""
        calc = Calculator()
        result = await calc.run("2 + 3 * 4")
        assert result == "14"  # Not 20

    @pytest.mark.asyncio
    async def test_mixed_types(self):
        """Calculator handles int and float mixed expressions."""
        calc = Calculator()
        result = await calc.run("1 + 2.5")
        assert result == "3.5"

    @pytest.mark.asyncio
    async def test_large_numbers(self):
        """Calculator handles large numbers."""
        calc = Calculator()
        result = await calc.run("999999999 * 999999999")
        assert result == "999999998000000001"

    @pytest.mark.asyncio
    async def test_decimal_numbers(self):
        """Calculator handles decimal numbers."""
        calc = Calculator()
        result = await calc.run("0.1 + 0.2")
        # Float precision may vary
        assert float(result) == pytest.approx(0.3, abs=1e-10)


# ============================================================================
# 5. Calculator - Safe Functions
# ============================================================================


class TestCalculatorFunctions:
    """Test Calculator with safe built-in functions."""

    @pytest.mark.asyncio
    async def test_abs_function(self):
        """Calculator evaluates abs()."""
        calc = Calculator()
        result = await calc.run("abs(-5)")
        assert result == "5"

    @pytest.mark.asyncio
    async def test_round_function(self):
        """Calculator evaluates round()."""
        calc = Calculator()
        result = await calc.run("round(3.7)")
        assert result == "4"

    @pytest.mark.asyncio
    async def test_min_function(self):
        """Calculator evaluates min()."""
        calc = Calculator()
        result = await calc.run("min(3, 1, 2)")
        assert result == "1"

    @pytest.mark.asyncio
    async def test_max_function(self):
        """Calculator evaluates max()."""
        calc = Calculator()
        result = await calc.run("max(3, 1, 2)")
        assert result == "3"

    @pytest.mark.asyncio
    async def test_pow_function(self):
        """Calculator evaluates pow()."""
        calc = Calculator()
        result = await calc.run("pow(2, 10)")
        assert result == "1024"

    @pytest.mark.asyncio
    async def test_int_function(self):
        """Calculator evaluates int()."""
        calc = Calculator()
        result = await calc.run("int(3.9)")
        assert result == "3"

    @pytest.mark.asyncio
    async def test_float_function(self):
        """Calculator evaluates float()."""
        calc = Calculator()
        result = await calc.run("float(5)")
        assert result == "5"

    @pytest.mark.asyncio
    async def test_nested_functions(self):
        """Calculator evaluates nested safe functions."""
        calc = Calculator()
        result = await calc.run("abs(-max(3, 5, 1))")
        assert result == "5"

    @pytest.mark.asyncio
    async def test_function_with_expression(self):
        """Calculator evaluates function with expression argument."""
        calc = Calculator()
        result = await calc.run("abs(2 - 5)")
        assert result == "3"


# ============================================================================
# 6. Calculator - Security (AST Sandbox)
# ============================================================================


class TestCalculatorSecurity:
    """Test Calculator security against code injection."""

    @pytest.mark.asyncio
    async def test_rejects_import(self):
        """Calculator rejects import statements."""
        calc = Calculator()
        result = await calc.run("import os")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_rejects_function_def(self):
        """Calculator rejects function definitions."""
        calc = Calculator()
        result = await calc.run("def f(): return 1")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_rejects_attribute_access(self):
        """Calculator rejects attribute access."""
        calc = Calculator()
        result = await calc.run("__import__('os').system('ls')")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_rejects_string_constant(self):
        """Calculator rejects string constants."""
        calc = Calculator()
        result = await calc.run("'hello'")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_rejects_list_literal(self):
        """Calculator rejects list literals."""
        calc = Calculator()
        result = await calc.run("[1, 2, 3]")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_rejects_dict_literal(self):
        """Calculator rejects dict literals."""
        calc = Calculator()
        result = await calc.run("{'a': 1}")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_rejects_lambda(self):
        """Calculator rejects lambda expressions."""
        calc = Calculator()
        result = await calc.run("lambda x: x + 1")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_rejects_unsafe_function(self):
        """Calculator rejects calls to unsafe functions."""
        calc = Calculator()
        result = await calc.run("exec('print(1)')")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_rejects_print(self):
        """Calculator rejects print() calls."""
        calc = Calculator()
        result = await calc.run("print(42)")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_rejects_open(self):
        """Calculator rejects open() calls."""
        calc = Calculator()
        result = await calc.run("open('/etc/passwd')")
        assert "Error" in result


# ============================================================================
# 7. Calculator - Error Handling
# ============================================================================


class TestCalculatorErrors:
    """Test Calculator error handling."""

    @pytest.mark.asyncio
    async def test_empty_expression(self):
        """Calculator returns error for empty expression."""
        calc = Calculator()
        result = await calc.run("")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_whitespace_expression(self):
        """Calculator returns error for whitespace-only expression."""
        calc = Calculator()
        result = await calc.run("   ")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_division_by_zero(self):
        """Calculator returns error for division by zero."""
        calc = Calculator()
        result = await calc.run("1 / 0")
        assert "Error" in result
        assert "zero" in result.lower()

    @pytest.mark.asyncio
    async def test_invalid_syntax(self):
        """Calculator returns error for invalid syntax."""
        calc = Calculator()
        result = await calc.run("2 +* 3")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_incomplete_expression(self):
        """Calculator returns error for incomplete expression."""
        calc = Calculator()
        result = await calc.run("2 +")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_floor_division_by_zero(self):
        """Calculator returns error for floor division by zero."""
        calc = Calculator()
        result = await calc.run("1 // 0")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_modulo_by_zero(self):
        """Calculator returns error for modulo by zero."""
        calc = Calculator()
        result = await calc.run("1 % 0")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_error_returns_string(self):
        """Calculator always returns string, even on error."""
        calc = Calculator()
        result = await calc.run("invalid")
        assert isinstance(result, str)


# ============================================================================
# 8. Boundary Conditions
# ============================================================================


class TestBoundaryConditions:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_calculator_with_session_id(self):
        """Calculator accepts session_id parameter."""
        calc = Calculator()
        result = await calc.run("1 + 1", session_id="s1")
        assert result == "2"

    @pytest.mark.asyncio
    async def test_calculator_with_kwargs(self):
        """Calculator accepts additional kwargs."""
        calc = Calculator()
        result = await calc.run("1 + 1", session_id="s1", extra="value")
        assert result == "2"

    @pytest.mark.asyncio
    async def test_web_search_returns_string_type(self):
        """WebSearch always returns string."""
        tool = WebSearch()
        result = await tool.run("test")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_calculator_returns_string_type(self):
        """Calculator always returns string."""
        calc = Calculator()
        result = await calc.run("42")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_calculator_zero_result(self):
        """Calculator handles zero result."""
        calc = Calculator()
        result = await calc.run("0 * 999")
        assert result == "0"

    @pytest.mark.asyncio
    async def test_calculator_negative_zero(self):
        """Calculator handles negative zero."""
        calc = Calculator()
        result = await calc.run("-0")
        assert result == "0"

    @pytest.mark.asyncio
    async def test_calculator_very_large_result(self):
        """Calculator handles very large results."""
        calc = Calculator()
        result = await calc.run("2 ** 100")
        assert result == str(2 ** 100)

    @pytest.mark.asyncio
    async def test_multiple_calculator_instances(self):
        """Multiple Calculator instances are independent."""
        calc1 = Calculator()
        calc2 = Calculator()
        r1 = await calc1.run("1 + 1")
        r2 = await calc2.run("2 + 2")
        assert r1 == "2"
        assert r2 == "4"

    @pytest.mark.asyncio
    async def test_multiple_web_search_instances(self):
        """Multiple WebSearch instances are independent."""
        tool1 = WebSearch()
        tool2 = WebSearch()
        r1 = await tool1.run("query1")
        r2 = await tool2.run("query2")
        assert "query1" in r1
        assert "query2" in r2

    def test_safe_eval_helper_is_function(self):
        """_safe_eval helper must be callable."""
        assert callable(_safe_eval)
