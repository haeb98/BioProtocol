# agents/step_structurer_node.py

import json
from typing import List, Dict, Any

from openai import OpenAI

from .step_state import StepState, Step

_client = OpenAI()


def call_openai_chat(
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
) -> str:
    """
    openai>=1.0.0용 chat.completions 래퍼 (Task Planner 쪽과 동일 패턴).
    """
    resp = _client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content


# -----------------------------
# 프롬프트 빌더
# -----------------------------

def build_step_structurer_prompt(
        methods_text: str,
        tasks: List[Dict[str, Any]],
        max_steps_per_task: int,
) -> str:
    """
    Task Planner 결과를 받아 각 Task별로 Step을 뽑는 baseline 프롬프트.
    - 지금은 Methods 전체를 그대로 주고,
    - 각 Task에 맞는 Step을 1~max_steps_per_task 개씩 생성하게 함.
    """
    # Task 요약 블록
    tasks_block = ""
    for t in tasks:
        tasks_block += f"\n- {t['id']}: {t['title']}\n  {t['description']}\n  type: {t.get('type', 'other')}"

    return f"""
You are an expert protocol writer.

Your goal is to break down each high-level experimental task into
a sequence of concrete but still reasonably high-level steps that
a bench scientist can follow.

INPUTS:
1) TARGET Methods section (free text)
2) A list of high-level tasks (T1, T2, ...)

Requirements:
- For each Task, produce 1 to {max_steps_per_task} steps.
- Each step must be:
  - Actionable (something a human can actually do in the lab),
  - Coherent and ordered (later steps depend on earlier ones).
- Do NOT go down to every pipetting detail, but do describe:
  - main action (e.g. "seed cells", "perform PCR", "run SDS-PAGE"),
  - what is being acted on, and for what purpose.
- Try to keep the number of steps small but sufficient to reproduce the task.
- Always link each step to its parent Task ID.

Return a JSON object with field "steps", where each element has:
- "id": step ID (e.g., "S1", "S2", ... in order of appearance)
- "task_id": parent Task ID (e.g., "T1")
- "title": short name of the step
- "description": 1-2 sentences
- "step_type": one of ["procedure", "setup", "qc", "analysis", "other"]

=== TARGET METHODS ===
{methods_text}

=== TASKS ===
{tasks_block}
"""


# -----------------------------
# JSON 파서
# -----------------------------

def _extract_json_block(text: str) -> str:
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first:last + 1]
    return text


def parse_steps_from_llm(llm_text: str) -> List[Dict[str, Any]]:
    try:
        raw = _extract_json_block(llm_text)
        obj = json.loads(raw)
        steps = obj.get("steps", [])
        if not isinstance(steps, list):
            return []
        return steps
    except Exception:
        return []


# -----------------------------
# 핵심 노드 함수
# -----------------------------

def run_step_structurer(
        state: StepState,
        model: str = "gpt-4o-mini",
) -> StepState:
    """
    LangGraph Node로 쓸 수 있는 Step Structurer.
    - 입력: StepState (protocol_id, methods_text, tasks_planned)
    - 출력: steps_structured, step_raw
    """
    # Task Planner가 뽑은 Task들을 dict로 변환
    task_dicts = [t.model_dump() for t in state.tasks_planned]

    if not task_dicts:
        # Task가 없으면 그냥 빈 상태로 반환
        state.steps_structured = []
        state.step_raw = {
            "error": "no_tasks_planned",
            "message": "No tasks_planned provided to StepState.",
        }
        return state

    prompt = build_step_structurer_prompt(
        methods_text=state.methods_text,
        tasks=task_dicts,
        max_steps_per_task=state.max_steps_per_task,
    )

    llm_text = call_openai_chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    steps_raw = parse_steps_from_llm(llm_text)

    try:
        steps_structured = [Step(**s) for s in steps_raw]
    except Exception:
        steps_structured = []

    state.steps_structured = steps_structured
    state.step_raw = {
        "prompt": prompt,
        "llm_raw_text": llm_text,
    }
    return state
