# src/nodes/step_planner.py
import json
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from src.types import GraphState, StepIR

load_dotenv()
client = OpenAI()

import os
import re

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def safe_parse_json(content: str, protocol_id: str, task_id: str) -> dict | None:
    """
    LLM 응답(content)을 JSON으로 파싱.
    - 안 되면 ```json ... ``` 제거 시도
    - 그래도 안 되면 로그 파일에 저장하고 None 반환
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        # 1차 실패 → 코드블록/앞뒤 공백 제거 시도
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned)
            cleaned = cleaned.rstrip("`").strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e2:
            # 그래도 안 되면 로그 남기고 skip
            log_path = os.path.join(LOG_DIR, "step_structurer_json_error.log")
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "protocol_id": protocol_id,
                    "task_id": task_id,
                    "error": str(e2),
                    "raw_content_head": cleaned[:2000],  # 너무 길어지지 않게 앞부분만
                }) + "\n")
            print(f"[WARN] JSON parse failed for {protocol_id} / {task_id}: {e2}")
            return None


def build_step_structurer_prompts(task: dict, methods_text: str):
    """
    step_structurer_new.py에서 쓰던 system/user prompt를
    여기로 옮긴다고 생각하면 됨.
    """
    task_name = task["task_name"]
    task_desc = task.get("description", "")

    system_prompt = """
You are a lab assistant that extracts step-by-step experimental procedures
for a given TASK from a biological methods section.

Follow these strict rules:
- Use ONLY the provided METHODS TEXT and TASK description. Do NOT hallucinate.
- Imagine you actually have to perform THIS TASK in the lab.
- Break down the task into a sequence of concrete steps in correct order.
- Do NOT create too many micro-steps; merge trivial actions that are normally
  done together into one step (typical tasks have ~3–15 steps).
- Each step must include:
  - step_text: one actionable instruction
  - step_rationale: why this step is necessary (null if not stated)
  - span_chunk: the exact supporting text span from METHODS
  - parameters: list of key numeric/conditional parameters if present
  - materials: key reagents/samples/equipment used in this step
- Output JSON with top-level key 'steps':
  { "steps": [ {...}, {...}, ... ] }
"""

    user_prompt = f"""TARGET TASK:
- task_name: {task_name}
- description: {task_desc}

METHODS TEXT:
{methods_text}
"""
    return system_prompt, user_prompt


def call_llm_step_structurer(task: dict,
                             methods_text: str,
                             protocol_id: str,
                             model: str = "gpt-4-1106-preview") -> List[StepIR]:
    sys_prompt, user_prompt = build_step_structurer_prompts(task, methods_text)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    finish_reason = resp.choices[0].finish_reason
    if finish_reason == "length":
        print(f"[WARN] output truncated for {protocol_id} / {task['task_id']}")

    content = resp.choices[0].message.content or ""
    parsed = safe_parse_json(content, protocol_id, task["task_id"])

    if parsed is None:
        return []

    steps = parsed.get("steps", parsed)
    if isinstance(steps, dict):
        steps = [steps]
    elif not isinstance(steps, list):
        # 예상치 못한 타입이면 그냥 비워버리고 return
        print("[WARN] step_structurer: steps is not list/dict, got:", type(steps))
        return []

    normalized_steps = []
    for st in steps:
        if isinstance(st, str):
            # 문자열만 온 경우: 최소 정보만 채운 step dict로 변환
            normalized_steps.append({
                "step_text": st,
                "step_rationale": "",
                "span_chunk": "",
                "parameters_llm_guess": [],
                "materials_llm_guess": [],
            })
        elif isinstance(st, dict):
            normalized_steps.append(st)
        else:
            # 기타 타입은 무시
            print("[WARN] step_structurer: unexpected step type:", type(st))
    steps = normalized_steps

    result: List[StepIR] = []
    task_id = task["task_id"]
    task_name = task.get("task_name", "")

    for idx, st in enumerate(steps):
        step_text = st.get("step_text") or st.get("description") or ""
        step_rationale = st.get("step_rationale")
        span_chunk = st.get("span_chunk", "")

        # LLM이 준 guess를 따로 보존
        params_guess = st.get("parameters", [])
        mats_guess = st.get("materials", [])

        step: StepIR = {
            "step_id": f"{task_id}::S{idx + 1}",
            "task_id": task_id,
            "title": None,
            "action": None,
            "step_text": step_text,
            "step_rationale": step_rationale,
            "span_chunk": span_chunk,
            # 공식 필드는 Condition Extractor가 채움
            "materials": [],
            "parameters": [],
            "evidence_spans": [],
            "verified": False,
            "verification_notes": "",
        }
        # LLM guess 기록
        step["parameters_llm_guess"] = params_guess  # type: ignore
        step["materials_llm_guess"] = mats_guess  # type: ignore
        step["task_name"] = task_name  # type: ignore
        step["protocol_id"] = protocol_id  # type: ignore

        result.append(step)

    return result


def step_planner_node(state: GraphState) -> GraphState:
    tasks = state.get("tasks", [])
    methods = state["methods_text"]
    protocol_id = state["protocol_id"]

    all_steps: List[StepIR] = []
    for t in tasks:
        t_steps = call_llm_step_structurer(
            task=t,
            methods_text=methods,
            protocol_id=protocol_id,
        )
        all_steps.extend(t_steps)

    new_state = dict(state)
    new_state["steps_raw"] = all_steps
    return new_state
