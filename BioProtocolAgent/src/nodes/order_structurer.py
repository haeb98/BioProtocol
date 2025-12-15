# src/nodes/order_structurer.py
import json
from typing import List, Dict, Any

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

client = OpenAI()  # 너가 쓰는 클라이언트 방식에 맞게 수정 (기존 action_extractor랑 통일)


def call_llm_order_structurer_for_actions(
        protocol_id: str,
        methods_text: str,
        actions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    gen_actions_ir_10.jsonl 에 있는 actions 리스트를 받아
    LLM에게 '가장 그럴듯한 순서'로 재배열하도록 요청하는 함수.
    반환값은 같은 스키마의 actions 리스트(단, order만 변경)
    """

    # LLM에 보여줄 요약용 텍스트 구성
    actions_brief = [
        {
            "id": a.get("action_id") or a.get("id") or f"{protocol_id}::A{i + 1}",
            "text": a.get("action_text") or a.get("description") or a.get("step_text", ""),
        }
        for i, a in enumerate(actions)
    ]

    system_prompt = (
        "You are an expert in wet-lab experimental protocols. "
        "Given a list of unordered experimental actions, "
        "reconstruct the most plausible chronological execution order."
    )

    user_prompt = (
        "You are given the Methods section of a paper and a list of experimental actions.\n"
        "Reorder the actions into the most plausible execution order.\n\n"
        "Methods text:\n"
        f"{methods_text}\n\n"
        "Unordered actions (each with an id and short text):\n"
        f"{json.dumps(actions_brief, ensure_ascii=False, indent=2)}\n\n"
        "Return ONLY a JSON list of action ids in the new order, e.g.:\n"
        '["Bio-protocol-2302::A1", "Bio-protocol-2302::A3", ...]'
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # 너가 쓰는 모델로 교체
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    content = resp.choices[0].message.content.strip()
    try:
        ordered_ids = json.loads(content)
    except json.JSONDecodeError:
        # 혹시 JSON 실패하면 그냥 기존 순서 그대로 반환
        return actions

    # id → action dict 매핑
    action_map = {}
    for i, a in enumerate(actions):
        aid = a.get("action_id") or a.get("id") or f"{protocol_id}::A{i + 1}"
        a = dict(a)
        a["action_id"] = aid
        action_map[aid] = a

    ordered_actions = []
    used = set()
    for aid in ordered_ids:
        if aid in action_map and aid not in used:
            ordered_actions.append(action_map[aid])
            used.add(aid)

    # 빠진 액션 있으면 뒤에 붙이기
    for aid, a in action_map.items():
        if aid not in used:
            ordered_actions.append(a)

    return ordered_actions
