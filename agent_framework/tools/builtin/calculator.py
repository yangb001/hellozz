"""Calculator - Mathematical expression evaluation tool for agent use.

This tool evaluates mathematical expressions safely using Python's AST module
to parse and validate expressions before evaluation. It supports basic
arithmetic, parentheses, and built-in math functions like abs().
"""
import ast
import operator
import math
from agent_framework.interfaces.base_tool import BaseTool


# Supported binary operators
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Supported unary operators
_UNARY_OPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Supported built-in functions (safe subset)
_SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "int": int,
    "float": float,
    "pow": pow,
}


def _safe_eval(node: ast.AST) -> float:
    """Safely evaluate an AST node representing a math expression.

    Only allows numeric literals, basic operators, and a safe subset
    of built-in functions. Rejects any attribute access, imports,
    or function calls to unsafe names.

    Args:
        node: An AST node to evaluate.

    Returns:
        The numeric result of the expression.

    Raises:
        ValueError: If the expression contains unsupported operations.
    """
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _BIN_OPS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _safe_eval(node.operand)
        return _UNARY_OPS[op_type](operand)
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCTIONS:
            func = _SAFE_FUNCTIONS[node.func.id]
            args = [_safe_eval(arg) for arg in node.args]
            return func(*args)
        raise ValueError(f"Unsupported function call: {ast.dump(node.func)}")
    else:
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


class Calculator(BaseTool):
    """Mathematical expression evaluation tool.

    Safely evaluates mathematical expressions using AST parsing.
    Supports basic arithmetic (+, -, *, /, //, %, **), parentheses,
    and a safe subset of built-in functions (abs, round, min, max).

    Attributes:
        name: Tool identifier ("calculator").
        description: Human-readable description of the tool.
    """

    name: str = "calculator"
    description: str = (
        "Evaluate mathematical expressions. "
        "Input should be a math expression (e.g., '2 + 3 * 4', '(10 - 2) / 3'). "
        "Supports +, -, *, /, //, %, **, parentheses, and functions like abs(), round()."
    )

    async def run(self, input: str, session_id: str = None, **kwargs) -> str:
        """Evaluate a mathematical expression.

        Args:
            input: The math expression string to evaluate.
            session_id: Optional session identifier (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            The result of the expression as a string, or an error message.
        """
        if not input or not input.strip():
            return "Error: Empty expression provided."

        expression = input.strip()

        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree)
            # Format: remove trailing .0 for integer results
            if isinstance(result, float) and result == int(result) and not isinstance(result, bool):
                return str(int(result))
            return str(result)
        except ZeroDivisionError:
            return "Error: Division by zero."
        except (ValueError, SyntaxError, TypeError) as e:
            return f"Error: Invalid expression - {e}"
