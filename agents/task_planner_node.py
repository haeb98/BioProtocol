# agents/task_planner_node.py

import json
from typing import List, Dict, Any

from openai import OpenAI

from .planner_state import PlannerState, Task

# OpenAI 클라이언트 (OPENAI_API_KEY 환경변수 사용)
_client = OpenAI()


def call_openai_chat(
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
) -> str:
    """
    openai>=1.0.0용 chat.completions 래퍼.
    반환값은 message.content (str)만 돌려줌.
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

def build_baseline_planner_prompt(methods_text: str, max_tasks: int) -> str:
    """
    RAG 없이 Methods 텍스트만 보고 태스크를 뽑는 프롬프트.
    기존 baseline task_miner에서 쓰던 지침을 이쪽으로 옮겨오면 됨.
    (지금은 예시지만, 네가 실제 사용하던 문구로 덮어써도 됨.)
    """
    return f"""
You are an expert experimental protocol planner.

Read the following Methods section and extract up to {max_tasks} high-level experimental tasks
that a human experimenter must perform to reproduce the study.

Guidelines:
- Focus on *high-level* tasks (e.g., "Cell culture and treatment", "Chromatin preparation and sonication",
  "Western blot analysis", "qRT-PCR quantification").
- Do not go down to pipetting-level micro-steps.
- Each task should be something that could be scheduled or delegated as a block of work.
- Use neutral, clear language.

Return a JSON object with a field "tasks", where each element has:
- "id": "T1", "T2", ...
- "title": short name of the task
- "description": 1-3 sentences describing what is done in this task
- "type": one of ["experiment", "preparation", "qc", "analysis", "other"]

Methods:
{methods_text}
"""


def build_rag_planner_prompt(
        methods_text: str,
        retrieved_protocols: List[Dict[str, Any]],
        max_tasks: int,
) -> str:
    """
    RAG를 통해 가져온 유사 프로토콜들을 참고하는 프롬프트.
    retrieved_protocols: [{ "protocol_id": ..., "title": ..., "methods": ... }, ...] 형식 가정.
    """
    retrieved_block = ""
    for i, p in enumerate(retrieved_protocols, start=1):
        retrieved_block += (
            f"\n### Retrieved Protocol {i} "
            f"({p.get('protocol_id', 'unknown')} - {p.get('title', '')})\n"
        )
        # 너무 길어질 수 있으니 methods는 앞부분만 사용
        retrieved_block += p.get("methods", "")[:4000]

    return f"""
You are an expert experimental protocol planner using retrieval-augmented generation.

Your goal is to plan a set of high-level experimental tasks for the TARGET protocol,
using both the target Methods and similar protocols retrieved from a corpus.

Instructions:
1. First, infer the typical high-level task structure from the retrieved protocols
   (e.g., "Sample preparation → Reaction setup → Post-reaction processing → Analysis").
2. Then, adapt and customize this structure to the TARGET Methods.
3. Do NOT blindly copy unrelated tasks; only keep what is relevant to the target.
4. Merge redundant tasks if they clearly belong together.
5. Return at most {max_tasks} tasks.

Return a JSON object with a field "tasks", where each element has:
- "id": "T1", "T2", ...
- "title": short name of the task
- "description": 1-3 sentences describing what is done in this task
- "type": one of ["experiment", "preparation", "qc", "analysis", "other"]

=== TARGET METHODS ===
{methods_text}

=== RETRIEVED PROTOCOLS (HINTS) ===
{retrieved_block}
"""


# -----------------------------
# JSON 파싱 유틸
# -----------------------------

def _extract_json_block(text: str) -> str:
    """
    LLM이 앞뒤에 설명을 덧붙여도, 첫 '{' ~ 마지막 '}' 부분만 잘라 JSON으로 시도.
    """
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first:last + 1]
    return text


def parse_tasks_from_llm(llm_text: str) -> List[Dict[str, Any]]:
    """
    LLM 출력 텍스트에서 "tasks" 리스트를 파싱하는 유틸.
    - JSON 파싱 실패 시 빈 리스트 리턴.
    """
    try:
        raw = _extract_json_block(llm_text)
        obj = json.loads(raw)
        tasks = obj.get("tasks", [])
        if not isinstance(tasks, list):
            return []
        return tasks
    except Exception:
        return []


# -----------------------------
# (미래용) RAG 검색 함수 스켈레톤
# -----------------------------

def retrieve_similar_protocols(methods_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    RAG용 검색 함수.

    - 지금은 아직 구현하지 않고, NotImplementedError를 던지도록 둠.
    - 나중에 너가 이미 만들어둔 FAISS+BM25 인덱스를 감싸는 래퍼를 여기 연결하면 됨.
      예) protocols_wo_test50.annot.jsonl 기반 검색.

    반환 형식 예시:
    [
      {
        "protocol_id": "Bio-protocol-2096",
        "title": "Some title",
        "methods": "Methods text..."
      },
      ...
    ]
    """
    raise NotImplementedError(
        "retrieve_similar_protocols는 아직 구현되지 않았습니다. "
        "FAISS/BM25 검색 모듈을 연결해서 구현해 주세요."
    )


# -----------------------------
# 핵심 노드 함수: run_task_planner
# -----------------------------

def run_task_planner(state: PlannerState, model: str = "gpt-4o-mini") -> PlannerState:
    """
    LangGraph Node로도 바로 쓸 수 있는 Task Planner 핵심 함수.

    - 입력: PlannerState (protocol_id, methods_text, rag_enabled, max_tasks)
    - 출력: PlannerState (tasks_planned, planner_raw 가 채워진 상태)

    내부 동작:
    1) rag_enabled가 True이면 유사 프로토콜 검색 후 RAG 프롬프트 사용 시도
       (현재는 retrieve_similar_protocols 미구현 → NotImplementedError 발생)
    2) RAG 실패 시 baseline 프롬프트로 fallback
    3) OpenAI Chat 호출
    4) JSON 파싱 후 Task 리스트 채우기
    """
    # 1. 프롬프트 작성 + (옵션) RAG 검색
    retrieved = []
    used_rag = False

    if state.rag_enabled:
        try:
            retrieved = retrieve_similar_protocols(state.methods_text)
            prompt = build_rag_planner_prompt(
                methods_text=state.methods_text,
                retrieved_protocols=retrieved,
                max_tasks=state.max_tasks,
            )
            used_rag = True
        except NotImplementedError:
            # 아직 RAG 미구현이면 baseline으로 자동 fallback
            prompt = build_baseline_planner_prompt(state.methods_text, state.max_tasks)
            used_rag = False
        except Exception:
            # 검색 중 다른 오류 발생해도 baseline으로 fallback
            prompt = build_baseline_planner_prompt(state.methods_text, state.max_tasks)
            used_rag = False
    else:
        prompt = build_baseline_planner_prompt(state.methods_text, state.max_tasks)
        used_rag = False

    # 2. LLM 호출
    llm_text = call_openai_chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    # 3. JSON 파싱 → Task 모델로 변환
    tasks_planned: List[Task]
    tasks_raw = parse_tasks_from_llm(llm_text)
    try:
        tasks_planned = [Task(**t) for t in tasks_raw]
    except Exception:
        tasks_planned = []

    # 4. state 업데이트
    state.tasks_planned = tasks_planned
    state.planner_raw = {
        "used_rag": used_rag,
        "prompt": prompt,
        "retrieved": retrieved,
        "llm_raw_text": llm_text,
    }
    return state
