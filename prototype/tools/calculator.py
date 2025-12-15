# prototype/tools/calculator.py

import ast
import operator as op

# Supported operators for safe evaluation
allowed_operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos
}


def _eval_expr(node):
    if isinstance(node, ast.Num):  # <number>
        return node.n
    elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
        op_type = type(node.op)
        if op_type in allowed_operators:
            return allowed_operators[op_type](_eval_expr(node.left), _eval_expr(node.right))
        else:
            raise ValueError(f"Operator {op_type} not supported.")
    elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
        op_type = type(node.op)
        if op_type in allowed_operators:
            return allowed_operators[op_type](_eval_expr(node.operand))
        else:
            raise ValueError(f"Unary operator {op_type} not supported.")
    else:
        raise ValueError("Invalid expression.")


def calculate(expr):
    """
    Safely evaluate a numeric expression from a string.
    Supports +, -, *, /, and ** (exponentiation).
    """
    try:
        node = ast.parse(expr.strip(), mode='eval').body
        result = _eval_expr(node)
        return str(result)
    except Exception as e:
        return f"Error: {e}"
