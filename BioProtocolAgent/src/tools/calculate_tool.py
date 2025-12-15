# src/tools/calculate_tool.py
import ast
import operator
from typing import Any, Dict

# 허용할 연산자만 정의
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Num):
        return float(node.n)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Only numeric constants allowed.")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _ALLOWED_OPERATORS[type(node.op)](left, right)
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def _safe_arith(expr: str) -> Dict[str, Any]:
    expr = expr.strip()
    if not expr:
        return {"error": "Empty expression."}
    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval_node(tree.body)
        return {"expression": expr, "result": result}
    except Exception as e:
        return {"expression": expr, "error": str(e)}


import math
from typing import Optional


def _parse_g_value(text: str) -> Optional[float]:
    """
    '1100×g', '1100 x g', '1100 g' 같은 표현에서 1100 추출
    """
    t = text.lower().replace("×", "x")
    for token in t.replace("x g", "xg").split():
        if "xg" in token:
            try:
                return float(token.split("xg")[0])
            except ValueError:
                continue
    # 숫자 + g 패턴
    for token in t.split():
        if token.endswith("g"):
            try:
                return float(token[:-1])
            except ValueError:
                continue
    return None


def _g_to_rpm(g: float, radius_cm: float) -> float:
    return math.sqrt(g / (1.118e-5 * radius_cm))


def calculate(expr: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    - 일반 산술식: "2 * 3.5 / 0.7"
    - g->rpm 힌트가 있을 때: "convert 1100×g to rpm"
    """
    expr = expr.strip()
    lowered = expr.lower()

    # 1) g -> rpm 변환 모드
    if "rpm" in lowered and "g" in lowered:
        g_val = _parse_g_value(expr)
        rotor_radius = context.get("rotor_radius_cm")  # 필요하면 protocol_view에 추가

        if g_val is None:
            return {
                "mode": "g_to_rpm",
                "input": expr,
                "error": "Could not parse g value.",
            }

        if rotor_radius is None:
            # 반경이 없으면 공식을 설명만
            return {
                "mode": "g_to_rpm",
                "g": g_val,
                "rotor_radius_cm": None,
                "formula": "rpm = sqrt(g / (1.118e-5 * r_cm))",
                "note": "rotor_radius_cm not provided; cannot compute exact rpm.",
            }

        rpm = _g_to_rpm(g_val, rotor_radius)
        return {
            "mode": "g_to_rpm",
            "g": g_val,
            "rotor_radius_cm": rotor_radius,
            "rpm": rpm,
        }

    # 2) 나머지는 일반 산술
    return _safe_arith(expr)
