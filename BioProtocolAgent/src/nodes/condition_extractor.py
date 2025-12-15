# src/nodes/condition_extractor.py

import json
import re
from json import JSONDecodeError
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from src.types import GraphState, StepIR

load_dotenv()
client = OpenAI()


def extract_json_array(text: str) -> str:
    """
    LLM이 앞뒤에 설명을 붙였거나 ```json ... ``` 같은 포맷을 쓴 경우,
    그 안에서 제일 그럴듯한 JSON 배열 부분만 추출하려는 best-effort 헬퍼.
    실패하면 원문 그대로 돌려준다.
    """
    if not text:
        return "[]"

    # ```json ... ``` 블록 안만 추출
    code_block = re.search(r"```json(.*?)```", text, re.DOTALL)
    if code_block:
        inner = code_block.group(1).strip()
        return inner if inner else "[]"

    # 배열 형태 [ ... ] 만 추출
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]

    # 그래도 안 되면 원본 반환 (나중에 JSONDecodeError로 처리)
    return text.strip()


SYSTEM_PROMPT = """
You are a scientific protocol condition extractor.

Given:
- a biological METHODS TEXT
- a list of protocol steps (step_id, step_text, span_chunk, and LLM guesses)

You must fill MATERIALS and PARAMETERS for each step.

Definitions:
- MATERIALS: concrete reagents, biological samples, buffers, media, solutions,
             instruments or equipment explicitly mentioned in METHODS.
- PARAMETERS: numeric or symbolic conditions explicitly stated in METHODS
              (time, temperature, volume, concentration, pH, rpm, etc).

Rules:
- Use ONLY the METHODS TEXT; do NOT hallucinate new materials or parameters.
- Prefer copying exact phrases (e.g., "37 °C", "200 μL", "3% glycerol").
- If nothing is stated for a step, use an empty list.
- Consider the provided *_llm_guess fields as hints, but override them if
  they contradict METHODS TEXT.
- Output MUST be a JSON list:
  [
    {
      "step_id": "...",
      "materials": [...],
      "parameters": [...]
    },
    ...
  ]
"""


def build_user_content(methods_text: str, steps_raw: List[StepIR]) -> str:
    """methods + skeleton + guess를 묶어서 LLM에 넘김"""
    # skeleton에서 필요한 필드만 추린다
    compact = []
    for s in steps_raw:
        compact.append({
            "step_id": s["step_id"],
            "step_text": s["step_text"],
            "span_chunk": s.get("span_chunk", ""),
            "materials_llm_guess": s.get("materials_llm_guess", []),
            "parameters_llm_guess": s.get("parameters_llm_guess", []),
        })
    payload = {
        "methods_text": methods_text,
        "steps": compact,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def call_llm_condition_extractor(methods_text: str,
                                 steps_raw: List[StepIR],
                                 model: str = "gpt-4-1106-preview") -> List[StepIR]:
    user_content = build_user_content(methods_text, steps_raw)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )
    content = resp.choices[0].message.content or " "
    print("\n[ConditionExtractor RAW OUTPUT (truncated)]")
    print(content[:800], "\n")

    json_like = extract_json_array(content)

    try:
        parsed = json.loads(json_like)
    except JSONDecodeError as e:
        print("[WARN] Failed to parse JSON from ConditionExtractor response:", e)
        print("[WARN] Falling back to empty materials/parameters.")
        parsed = []  # fallback: 아무 것도 못 읽었으면 빈 리스트로 진행

    # parsed: [{"step_id": "...", "materials": [...], "parameters": [...]}]
    by_id: Dict[str, Dict[str, Any]] = {}
    if isinstance(parsed, list):
        for r in parsed:
            if isinstance(r, dict) and "step_id" in r:
                by_id[r["step_id"]] = r

    enriched: List[StepIR] = []
    for s in steps_raw:
        ext = by_id.get(s["step_id"], {})
        enriched.append(StepIR(
            **{
                **s,
                "materials": ext.get("materials", []),
                "parameters": ext.get("parameters", []),
            }
        ))
    return enriched


def condition_extractor_node(state: GraphState) -> GraphState:
    methods = state["methods_text"]
    steps_raw = state.get("steps_raw", [])

    new_steps = call_llm_condition_extractor(methods, steps_raw)
    new_state = dict(state)
    new_state["steps"] = new_steps
    return new_state  # LangGraph에서 다음 노드(or END)로 전달
