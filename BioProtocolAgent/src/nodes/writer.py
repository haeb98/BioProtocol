# src/nodes/writer.py
import json
from typing import Dict, Any, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


def _parse_json_list_from_text(text: str) -> List[Dict[str, Any]]:
    """
    LLM이 JSON 리스트만 딱 주면 그대로 파싱하고,
    앞뒤에 설명이 섞여 있어도 [ ... ] 부분만 잘라서 파싱을 시도한다.
    실패하면 [] 를 돌려준다.
    """
    text = text.strip()
    # 1차 시도: 전체를 그대로
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    # 2차 시도: 첫 '[' 부터 마지막 ']' 까지를 잘라서 시도
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = text[start: end + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass

    # 3차: 실패하면 빈 리스트
    return []


SYSTEM_WRITER = """You are an expert biology protocol writer. 
You receive (optionally) tasks, steps, or action skeletons, and must output
a list of normalized Action IR objects in JSON with fields:
[action_id, action_type, materials, conditions, produces, step_text]."""


def _call_llm_for_actions(prompt: str) -> List[Dict[str, Any]]:
    """
    Methods / Tasks / Steps 를 넣고 Action IR 리스트를 JSON으로 생성시키는 공통 함수.
    """
    resp = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert in wet-lab protocols. "
                    "You MUST return only a JSON array of action objects. "
                    "Do not include any explanation outside JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    content = resp.choices[0].message.content or ""
    actions = _parse_json_list_from_text(content)

    # 디버깅용 로그 (원하면 주석 처리)
    if not actions:
        print("[writer] WARNING: LLM did not return valid JSON, fallback to empty list.")
        # 최소 한 개 dummy라도 넣고 싶으면 여기서 생성해도 됨.

    return actions


def writer_node_simple(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    P1: Methods -> Writer (단일 LLM)
    - Methods 텍스트만 보고 바로 Action IR 리스트를 생성
    - protocol_text(또는 protocol_draft)는 비워두고, 나중에 원하면 따로 재사용
    """
    methods = state["methods_text"]

    prompt = (
        "You are a single LLM that directly converts Methods text into "
        "atomic Action IRs.\n\n"
        "Methods:\n"
        f"{methods}\n\n"
        "Return ONLY a JSON array of action objects."
    )
    actions = _call_llm_for_actions(prompt)

    state["actions"] = actions
    state["protocol_draft"] = None
    return state


def writer_node_prompted(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    P2: Methods -> Writer (with richer prompt)
    - Methods에서 바로 Action IR을 만들되, 액션/재료/조건 필드를 좀 더 상세히 요구
    """
    methods = state["methods_text"]

    prompt = (
        "You are an expert protocol planner. From the following Methods text, "
        "extract a SEQUENCE of atomic actions.\n"
        "Each action object must have:\n"
        '  - "action": verb (Add, Incubate, Centrifuge, Mix, ...)\n'
        '  - "description": short natural language description\n'
        '  - "materials": list of {name, role, volume, concentration, state, source}\n'
        '  - "conditions": list of {type, value, unit, source}\n'
        '  - "produces": list of product names\n'
        '  - "evidence_span": short span copied from the Methods text\n\n'
        "Methods:\n"
        f"{methods}\n\n"
        "Return ONLY a JSON array of action objects."
    )

    actions = _call_llm_for_actions(prompt)
    state["actions"] = actions
    state["protocol_draft"] = None
    return state


def writer_node_skeleton(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    P3~P6 공통 writer
    - actions 가 이미 있는 경우: IR을 기반으로 자연어 프로토콜만 생성
    - actions 가 없는 경우: Methods + Tasks + Steps 를 기반으로 Action IR 생성
    """
    methods = state["methods_text"]
    tasks = state.get("tasks", [])
    steps = state.get("steps", [])
    existing_actions = state.get("actions")

    # (1) 이미 Action IR이 만들어진 상태 (P5, P6 등)
    if existing_actions:
        prompt = (
            "You are given a verified sequence of Action IRs for a wet-lab protocol. "
            "Rewrite them into a clear, step-by-step experimental protocol text. "
            "Mask safety-critical parameters (exact time/temperature/volume) if needed "
            "by replacing them with placeholders like '*미공개*' or '[X]'.\n\n"
            f"Action IRs:\n{json.dumps(existing_actions, ensure_ascii=False, indent=2)}"
        )
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You rewrite action sequences into protocol text."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        state["protocol_draft"] = resp.choices[0].message.content
        # actions 는 그대로 유지
        return state

    # (2) 아직 Action IR이 없다 → skeleton을 기반으로 처음부터 액션 생성
    prompt = (
        "You are given Methods, high-level Tasks, and Step skeletons. "
        "Convert them into a SEQUENCE of atomic Action IRs.\n\n"
        f"Methods:\n{methods}\n\n"
        f"Tasks:\n{json.dumps(tasks, ensure_ascii=False, indent=2)}\n\n"
        f"Steps:\n{json.dumps(steps, ensure_ascii=False, indent=2)}\n\n"
        "Return ONLY a JSON array of action objects."
    )
    actions = _call_llm_for_actions(prompt)
    state["actions"] = actions
    state["protocol_draft"] = None
    return state
