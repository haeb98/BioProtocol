# src/nodes/verifier.py
import json
from typing import Dict, Any

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

client = OpenAI()  # 기존에 쓰는 클라이언트 방식대로 맞춰줘


def call_llm_verify_action(
        protocol_id: str,
        methods_text: str,
        action: Dict[str, Any],
) -> Dict[str, Any]:
    """
    하나의 action dict 에 대해 CoVe + ReAct 스타일 검증을 수행하고
    검증 결과를 포함한 새로운 action dict 를 반환.
    """

    action_text = action.get("action_text") or action.get("description") or action.get("step_text", "")
    materials = action.get("materials", [])
    conditions = action.get("conditions", [])
    produces = action.get("produces", [])

    system_prompt = (
        "You are an expert wet-lab protocol reviewer. "
        "Your job is to VERIFY whether a generated experimental action "
        "is faithful to the original Methods section of a paper. "
        "Use a Chain-of-Verification style: first plan what to check, "
        "then look for evidence in the Methods text, then decide."
    )

    user_prompt = (
        "You are given the Methods section of a biology paper and one generated action in an IR.\n"
        "Your tasks:\n"
        "1. Generate 2-4 verification questions about this action "
        "(e.g., materials coverage, condition correctness, order dependency).\n"
        "2. For each question, reason step-by-step and quote short evidence spans from the Methods text.\n"
        "3. Based on all evidence, label the action as one of:\n"
        '   - \"supported\" (clearly stated),\n'
        '   - \"partially_supported\" (core idea present but details missing),\n'
        '   - \"hallucinated\" (not found or clearly fabricated),\n'
        '   - \"contradicted\" (Methods says something else).\n'
        "4. If hallucinated or contradicted, propose how to revise or delete this action.\n"
        "5. Optionally suggest if this action should move earlier/later in the sequence.\n\n"
        f"Methods text:\n{methods_text}\n\n"
        "Current action IR (JSON):\n"
        f"{json.dumps({'protocol_id': protocol_id, 'action': action}, ensure_ascii=False, indent=2)}\n\n"
        "Respond ONLY in the following JSON format:\n"
        "{\n"
        '  \"verification_questions\": [\"...\", \"...\"],\n'
        '  \"reasoning_traces\": [  // same length as verification_questions\n'
        '    {\n'
        '      \"question\": \"...\",\n'
        '      \"thought\": \"...\",   // what you decided to check\n'
        '      \"evidence_spans\": [\"...\", \"...\"],  // short quotes from Methods\n'
        '      \"local_verdict\": \"supported | partially_supported | hallucinated | contradicted\"\n'
        "    }\n"
        "  ],\n"
        '  \"global_verdict\": \"supported | partially_supported | hallucinated | contradicted\",\n'
        '  \"revision_suggestion\": \"...\",   // how to fix or delete the action\n'
        '  \"order_suggestion\": \"no_change | move_before:<some step id> | move_after:<some step id>\"\n'
        "}\n"
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # 너가 사용하는 모델명으로 교체
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    content = resp.choices[0].message.content.strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # JSON 실패 시 최소 정보만 반환
        parsed = {
            "verification_questions": [],
            "reasoning_traces": [],
            "global_verdict": "unknown",
            "revision_suggestion": "",
            "order_suggestion": "no_change",
        }

    # 기존 action에 검증 결과를 붙여서 반환
    out_action = dict(action)
    out_action["verification"] = {
        "global_verdict": parsed.get("global_verdict", "unknown"),
        "verification_questions": parsed.get("verification_questions", []),
        "reasoning_traces": parsed.get("reasoning_traces", []),
        "revision_suggestion": parsed.get("revision_suggestion", ""),
        "order_suggestion": parsed.get("order_suggestion", "no_change"),
    }
    return out_action


# src/nodes/verifier.py (추가)

from typing import TypedDict, List


class GraphState(TypedDict, total=False):
    protocol_id: str
    methods_text: str
    actions: List[Dict[str, Any]]
    actions_verified: List[Dict[str, Any]]


def verifier_node(state: GraphState) -> GraphState:
    """
    LangGraph에서 사용할 수 있는 Verifier 노드.
    state['actions'] 를 검증해서 state['actions_verified'] 로 반환.
    """
    pid = state["protocol_id"]
    methods = state["methods_text"]
    actions = state.get("actions", [])

    verified_actions = []
    for a in actions:
        v = call_llm_verify_action(pid, methods, a)
        verified_actions.append(v)

    new_state = dict(state)
    new_state["actions_verified"] = verified_actions
    return new_state
