# src/nodes/verifier_react.py
import json
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from src.tools.verifier_tools import run_tool

load_dotenv()

client = OpenAI()  # 이미 쓰던 방식대로 초기화

SYSTEM_PROMPT = """
You are a verification agent for wet-lab protocol IR.

For each action, you MUST:
1. Think step-by-step (Thought).
2. Decide which external tool(s) to call:
   - doc_search: search the original JATS article for missing or conflicting details.
   - rag_tool: search external protocols for typical conditions / volumes / order.
   - calculator: perform domain calculations (e.g., convert g <-> rpm, check ratios).
   - reorder_tool: check whether the ordering of actions is consistent with material dependencies.

You are allowed to call MULTIPLE tools in sequence for a single verification:
- For example:
  Thought: I will first look up more context in the JATS article.
  Action: [doc_search: "medium change and passaging condition"]

  (User will return Observation with hits.)

  Thought: I want to compare with external protocols.
  Action: [rag_tool: "cell passaging protocol"]

  Thought: Now I should convert 1100×g to rpm with rotor radius 8.5 cm.
  Action: [calculator: "convert 1100×g to rpm"]

Only when you are satisfied that you have enough evidence, you must output:

  Final Answer: {
    "verified_action": {...},
    "global_verdict": "supported" | "contradicted" | "needs_revision" | "unknown",
    "evidence_spans": [...],
    "revision_notes": "...",
    "order_suggestion": "no_change" | "manual_check_required" | "move_before: <id>" | ...
  }

""".strip()


def _parse_tool_action(action_line: str):
    m = re.match(r'\[?(\w+):\s*["“](.*?)["”]\]?', action_line.strip())
    if m:
        return m.group(1), m.group(2)
    return None, None


def _call_llm(messages):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # 환경에 맞게 수정
        messages=messages,
        temperature=0.3,
    )
    return resp.choices[0].message.content


def verify_actions(protocol_view: Dict[str, Any]) -> Dict[str, Any]:
    methods_text = protocol_view["methods_text"]
    actions = protocol_view["actions"]
    pid = protocol_view["protocol_id"]
    pmcid = protocol_view.get("pmcid")

    verified_actions: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []

    for i, action in enumerate(actions):
        print(f"\n🧪 Verifying action {i + 1}/{len(actions)} - {action.get('action_id')}")

        user_content = (
            "Verify the following action IR against the Methods text.\n\n"
            f"Methods:\n{methods_text}\n\n"
            f"Action IR:\n{json.dumps(action, ensure_ascii=False, indent=2)}\n\n"
            "Remember to use tools via `Action: [tool_name: \"query\"]` and "
            "only output Final Answer when you have enough evidence."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        step_trace: List[Dict[str, Any]] = []
        final_answer = None

        for j in range(5):  # 최대 5번 루프
            resp_text = _call_llm(messages)
            print(f"  ↪️ Iteration {j + 1} response:\n{resp_text}\n")

            thought, action_call = "", ""
            for line in resp_text.splitlines():
                if line.startswith("Thought:"):
                    thought = line[len("Thought:"):].strip()
                elif line.startswith("Action:"):
                    action_call = line[len("Action:"):].strip()

            trace_entry = {"thought": thought, "action": action_call}

            # Final Answer가 포함되어 있으면 파싱 시도
            if "Final Answer:" in resp_text:
                try:
                    ans_text = resp_text.split("Final Answer:")[1].strip()
                    final_answer = json.loads(ans_text)
                    trace_entry["final_answer"] = final_answer
                except Exception:
                    trace_entry["final_answer_raw"] = resp_text
                step_trace.append(trace_entry)
                print("✅ Final Answer received.")
                break

            # 툴 호출
            tool_name, arg = _parse_tool_action(action_call)
            if tool_name:
                tool_context = {
                    "protocol_id": pid,
                    "pmcid": pmcid,
                    "methods_text": methods_text,
                    "current_action": action,
                    "actions": actions,
                }
                obs = run_tool(tool_name, arg, tool_context)

            else:
                obs = {"warning": "No tool called in this step."}

            trace_entry["observation"] = obs
            step_trace.append(trace_entry)

            # 대화 컨텍스트 업데이트
            messages.append({"role": "assistant", "content": resp_text})
            messages.append({"role": "user", "content": f"Observation: {json.dumps(obs, ensure_ascii=False)}"})

        # Final Answer 없으면 "unknown"
        if not final_answer:
            verified = dict(action)
            verified.setdefault("verification", {})
            verified["verification"].update({
                "global_verdict": "unknown",
                "revision_notes": "Verifier could not produce a grounded Final Answer.",
            })
        else:
            verified_action = final_answer.get("verified_action", action)
            verified = dict(verified_action)
            verified["verification"] = {
                "global_verdict": final_answer.get("global_verdict", "unknown"),
                "evidence_spans": final_answer.get("evidence_spans", []),
                "revision_notes": final_answer.get("revision_notes", ""),
                "order_suggestion": final_answer.get("order_suggestion", "no_change"),
            }

        verified_actions.append(verified)
        traces.append({
            "action_id": action.get("action_id"),
            "action_text": action.get("action_text"),
            "trace": step_trace,
        })

    return {
        "protocol_id": pid,
        "methods_text": methods_text,
        "actions_verified": verified_actions,
        "verification_traces": traces,
    }


# src/nodes/verifier_react.py

from src.types import GraphState  # 파일 위쪽 import 구역에 이미 없다면 추가


# ... (위에는 기존 verify_actions 정의가 그대로 둠)

def verifier_node(state: GraphState) -> GraphState:
    """
    LangGraph에서 쓰기 위한 래퍼 노드.
    GraphState --> verify_actions() --> GraphState 로 되돌려준다.
    """
    # 1) GraphState에서 verifier가 필요한 정보만 뽑아서
    protocol_view = {
        "protocol_id": state["protocol_id"],
        "methods_text": state["methods_text"],
        # action_extractor_node가 채운 액션 IR 리스트
        "actions": state.get("actions", []),
        # pmcid는 있으면 쓰고, 없으면 None
        "pmcid": state.get("pmcid"),
    }

    # 2) 기존에 쓰던 verify_actions() 호출
    result = verify_actions(protocol_view)

    # 3) 결과를 GraphState에 합쳐서 반환
    new_state = dict(state)
    new_state["actions_verified"] = result["actions_verified"]
    new_state["verification_traces"] = result["verification_traces"]

    return new_state
