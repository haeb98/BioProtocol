# src/tools/reorder_tool.py
from typing import Dict, Any, List, Set


def _collect_produced_before(actions: List[Dict[str, Any]],
                             idx: int) -> Set[str]:
    """0..idx-1 까지의 action들이 만들어낸 produces 집합."""
    produced: Set[str] = set()
    for i in range(idx):
        a = actions[i]
        for p in a.get("produces", []) or []:
            if isinstance(p, str):
                produced.add(p.lower())
            elif isinstance(p, dict) and "name" in p:
                produced.add(p["name"].lower())
    return produced


def _collect_materials(action: Dict[str, Any]) -> Set[str]:
    mats: Set[str] = set()
    for m in action.get("materials", []) or []:
        if isinstance(m, str):
            mats.add(m.lower())
        elif isinstance(m, dict) and "name" in m:
            mats.add(m["name"].lower())
    return mats


def reorder_tool(arg: str,
                 current_action: Dict[str, Any],
                 all_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Rule-based reorder 점검:
    - current_action이 사용하는 material 중 '앞선 step에서 생산된 적 없는 것'이 있으면
      'manual_check_required'로 플래그.

    TODO: rag 기반 외부 프로토콜 순서 비교는 이후 확장.
    """
    cur_id = current_action.get("action_id")
    try:
        idx = next(i for i, a in enumerate(all_actions)
                   if a.get("action_id") == cur_id)
    except StopIteration:
        idx = None

    if idx is None:
        return {
            "order_suggestion": "unknown",
            "message": "current_action not found in actions list.",
        }

    produced_before = _collect_produced_before(all_actions, idx)
    needed = _collect_materials(current_action)

    # 앞에서 생산된 적 없는 materials
    unmet = sorted(list(needed - produced_before))

    if unmet:
        return {
            "order_suggestion": "manual_check_required",
            "unmet_materials": unmet,
            "message": (
                f"These materials are not produced in prior steps: {unmet}. "
                f"Please verify ordering around {cur_id}."
            ),
        }

    return {
        "order_suggestion": "no_change",
        "message": "All materials appear to be available from previous steps.",
    }
