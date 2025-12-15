# src/tools/verifier_tools.py
from typing import Any, Dict

from src.tools.calculate_tool import calculate
# TODO: 실제 구현/경로에 맞게 import 수정
from src.tools.doc_search import doc_search_fulltext
from src.tools.rag_search import rag_tool
from src.tools.reorder_tool import reorder_tool


# def doc_search(query: str, methods_text: str) -> Dict[str, Any]:
#     # 실제 구현 있는 경우 그대로 사용
#     # 여기서는 dummy 형태만 예시
#     return {
#         "query": query,
#         "matches": [methods_text[:300]]  # 앞부분 잘라서 예시로
#     }
#
#
# def rag_tool(query: str) -> Dict[str, Any]:
#     return {
#         "query": query,
#         "hits": []
#     }
#
#
# def calculate(expr: str) -> Dict[str, Any]:
#     try:
#         val = eval(expr, {})
#     except Exception:
#         val = None
#     return {"expression": expr, "value": val}
#
#
# def reorder_tool(arg: str, current_action: Dict, all_actions: List[Dict]) -> Dict:
#     """
#     Placeholder reorder tool.
#     arg: reasoning text from LLM (e.g. "This action should be before T3::S1")
#     current_action: dict of the action being verified
#     all_actions: list of all actions in the protocol
#     """
#     # 여기서는 단순히 "변경 없음" 리턴
#     return {
#         "suggested_order": "no_change",
#         "reason": "Reorder tool not fully implemented yet."
#     }


def run_tool(tool_name: str, arg: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    ReAct에서 호출되는 공용 Tool 래퍼.
    context 예시:
      {
        "protocol_id": ...,
        "pmcid": ...,
        "methods_text": ...,
        "current_action": {...},
        "actions": [...],
      }
    """
    if tool_name == "doc_search":
        query = arg.strip()
        pmcid = context.get("pmcid")

        if not pmcid:
            return {"hits": []}

        hits = doc_search_fulltext(pmcid, query, top_k=5)
        return {"hits": hits}

    elif tool_name == "rag_tool":
        # 외부 Bio-protocol 코퍼스 기반 RAG
        query = arg.strip()
        return rag_tool(query)

    elif tool_name == "calculator":
        expr = arg.strip()
        return calculate(expr, context)

    elif tool_name == "reorder_tool":
        action = context.get("current_action") or {}
        all_actions = context.get("actions") or []
        return reorder_tool(arg, action, all_actions)

    else:
        return {"error": f"Unknown tool: {tool_name}"}
