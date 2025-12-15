# src/nodes/action_extractor.py

import json
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from src.types import StepIR, ActionIR

load_dotenv()
client = OpenAI()

SYSTEM_PROMPT = """
You are an expert in biological wet-lab protocols.

Your task:
Given ONE protocol step and its context, decompose the step into a small
number of ACTIONS, and for each action extract structured MATERIALS,
CONDITIONS, and PRODUCED entities.

Output format:
A JSON array like:
[
  {
    "action": "Add",
    "description": "...",
    "materials": [
      {
        "name": "...",
        "role": "substrate|reagent|buffer|instrument|other",
        "volume": "...",
        "concentration": "...",
        "state": "...",
        "source": "methods_text|llm_inferred"
      }
    ],
    "conditions": [
      {
        "type": "temperature|duration|speed|pH|other",
        "value": "...",
        "unit": "...",
        "source": "methods_text|llm_inferred"
      }
    ],
    "produces": ["..."],
    "evidence_span": "exact phrase from METHODS that supports this action"
  },
  ...
]

Guidelines:
- Split the step into the MINIMAL number of meaningful actions, such as
  "Add", "Mix", "Incubate", "Centrifuge", "Wash", "Resuspend", "Transfer".
- Use the METHODS_TEXT and SPAN_CHUNK as ground truth. Do NOT invent
  impossible details.
- You MAY infer trivial complements (e.g., if '0.1 volume of 8 M LiCl'
  is added to RNA, you can infer that the remaining '1 volume' is the
  diluent) but mark these as "source": "llm_inferred".
- Use parameters_llm_guess and materials_llm_guess as hints to fill
  volume/temperature/duration/etc, but override them if they contradict
  METHODS_TEXT.
- If some attribute is not stated and not safely inferable, use an empty
  string "" for that field.
- Return ONLY a JSON array, no additional text or comments.
"""


def build_user_prompt_for_step(step: StepIR, methods_text: str) -> str:
    payload = {
        "METHODS_TEXT": methods_text,
        "STEP_TEXT": step["step_text"],
        "STEP_RATIONALE": step.get("step_rationale", ""),
        "SPAN_CHUNK": step.get("span_chunk", ""),
        "materials_llm_guess": step.get("materials_llm_guess", []),
        "parameters_llm_guess": step.get("parameters_llm_guess", []),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def extract_json_array(text: str) -> str:
    # 이전에 만든 JSON 추출 헬퍼 재사용
    import re

    if not text:
        return "[]"
    block = re.search(r"```json(.*?)```", text, re.DOTALL)
    if block:
        inner = block.group(1).strip()
        return inner or "[]"
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text.strip()


def call_llm_action_extractor_for_step(step: StepIR,
                                       methods_text: str,
                                       model: str = "gpt-4o-mini") -> List[ActionIR]:
    user_content = build_user_prompt_for_step(step, methods_text)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )
    raw = resp.choices[0].message.content or ""
    json_like = extract_json_array(raw)

    try:
        parsed = json.loads(json_like)
    except json.JSONDecodeError:
        parsed = []

    actions: List[ActionIR] = []
    for idx, a in enumerate(parsed):
        if not isinstance(a, dict):
            continue
        aid = f'{step["step_id"]}::A{idx + 1}'
        actions.append(ActionIR(
            action_id=aid,
            step_id=step["step_id"],
            protocol_id=step["protocol_id"],
            task_id=step["task_id"],
            action=a.get("action", ""),
            description=a.get("description", ""),
            materials=a.get("materials", []),
            conditions=a.get("conditions", []),
            produces=a.get("produces", []),
            evidence_span=a.get("evidence_span", ""),
            source="llm_extracted",
        ))
    return actions


# src/nodes/action_extractor.py (계속)

from src.types import GraphState


def action_extractor_node(state: GraphState) -> GraphState:
    methods_text = state["methods_text"]
    steps = state.get("steps_raw", [])  # or "steps" depending on 설계

    all_actions: List[ActionIR] = []
    for step in steps:
        actions = call_llm_action_extractor_for_step(step, methods_text)
        all_actions.extend(actions)

    new_state = dict(state)
    new_state["actions"] = all_actions
    return new_state
