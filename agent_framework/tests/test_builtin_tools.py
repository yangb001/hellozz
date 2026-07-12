"""Tests for built-in tools: WebSearch and Calculator."""
import pytest
from agent_framework.tools.builtin.web_search import WebSearch
from agent_framework.tools.builtin.calculator import Calculator


class TestWebSearchInit:
    """Test WebSearch initialization."""

    def test_name(self):
        tool = WebSearch()
        assert tool.name == "web_search"

    def test_description(self):
        tool = WebSearch()
        assert len(tool.description) > 0

    def test_is_subclass_of_base_tool(self):
        from agent_framework.interfaces.base_tool import BaseTool
        assert issubclass(WebSearch, BaseTool)


class TestWebSearchRun:
    """Test WebSearch.run method."""

    @pytest.mark.asyncio
    async def test_returns_string(self):
        tool = WebSearch()
        result = await tool.run("test query")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_returns_results_for_query(self):
        tool = WebSearch()
        result = await tool.run("Python programming")
        assert len(result) > 0
        assert "Python programming" in result or "search" in result.lower() or "result" in result.lower()

    @pytest.mark.asyncio
    async def test_empty_query(self):
        tool = WebSearch()
        result = await tool.run("")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_with_session_id(self):
        tool = WebSearch()
        result = await tool.run("test", session_id="s1")
        assert isinstance(result, str)


class TestCalculatorInit:
    """Test Calculator initialization."""

    def test_name(self):
        calc = Calculator()
        assert calc.name == "calculator"

    def test_description(self):
        calc = Calculator()
        assert len(calc.description) > 0

    def test_is_subclass_of_base_tool(self):
        from agent_framework.interfaces.base_tool import BaseTool
        assert issubclass(Calculator, BaseTool)


class TestCalculatorBasicArithmetic:
    """Test basic arithmetic operations."""

    @pytest.mark.asyncio
    async def test_addition(self):
        calc = Calculator()
        result = await calc.run("2 + 3")
        assert "5" in result

    @pytest.mark.asyncio
    async def test_subtraction(self):
        calc = Calculator()
        result = await calc.run("10 - 4")
        assert "6" in result

    @pytest.mark.asyncio
    async def test_multiplication(self):
        calc = Calculator()
        result = await calc.run("3 * 7")
        assert "21" in result

    @pytest.mark.asyncio
    async def test_division(self):
        calc = Calculator()
        result = await calc.run("10 / 2")
        assert "5" in result

    @pytest.mark.asyncio
    async def test_floor_division(self):
        calc = Calculator()
        result = await calc.run("10 // 3")
        assert "3" in result

    @pytest.mark.asyncio
    async def test_modulo(self):
        calc = Calculator()
        result = await calc.run("10 % 3")
        assert "1" in result

    @pytest.mark.asyncio
    async def test_power(self):
        calc = Calculator()
        result = await calc.run("2 ** 3")
        assert "8" in result


class TestCalculatorComplexExpressions:
    """Test more complex mathematical expressions."""

    @pytest.mark.asyncio
    async def test_parentheses(self):
        calc = Calculator()
        result = await calc.run("(2 + 3) * 4")
        assert "20" in result

    @pytest.mark.asyncio
    async def test_nested_parentheses(self):
        calc = Calculator()
        result = await calc.run("((1 + 2) * (3 + 4))")
        assert "21" in result

    @pytest.mark.asyncio
    async def test_float_result(self):
        calc = Calculator()
        result = await calc.run("10 / 3")
        assert "3.333" in result

    @pytest.mark.asyncio
    async def test_negative_numbers(self):
        calc = Calculator()
        result = await calc.run("-5 + 3")
        assert "-2" in result

    @pytest.mark.asyncio
    async def test_decimal_numbers(self):
        calc = Calculator()
        result = await calc.run("1.5 + 2.5")
        assert "4" in result


class TestCalculatorEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_division_by_zero(self):
        calc = Calculator()
        result = await calc.run("1 / 0")
        assert "error" in result.lower() or "zero" in result.lower()

    @pytest.mark.asyncio
    async def test_invalid_expression(self):
        calc = Calculator()
        result = await calc.run("hello world")
        assert "error" in result.lower() or "invalid" in result.lower()

    @pytest.mark.asyncio
    async def test_empty_input(self):
        calc = Calculator()
        result = await calc.run("")
        assert "error" in result.lower() or "empty" in result.lower() or "invalid" in result.lower()

    @pytest.mark.asyncio
    async def test_with_session_id(self):
        calc = Calculator()
        result = await calc.run("1 + 1", session_id="s1")
        assert "2" in result


class TestCalculatorBuiltinFunctions:
    """Test math built-in functions if supported."""

    @pytest.mark.asyncio
    async def test_abs_function(self):
        calc = Calculator()
        result = await calc.run("abs(-5)")
        assert "5" in result
