# src/nodes/task_planner.py
import json
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from src.types import GraphState

load_dotenv()

client = OpenAI()  # OPENAI_API_KEY는 .env 또는 환경변수에서 읽힘


def build_task_miner_prompts(sec_text: str, bio_title: str, article_title: str):
    """
    기존에 task_miner_new.py에서 쓰던 prompt를
    함수 형태로 옮긴 것이라고 생각하면 됨.
    (지금 title-aware로 바꾼 버전을 여기에 붙여넣으면 됨)
    """
    system_prompt = """
You are an expert biomedical scientist helping to extract experimental TASKS
from the methods sections of biology papers.

You are given:
- ARTICLE_TITLE: the overall paper title.
- PROTOCOL_TITLE: the specific Bio-protocol title (a particular assay or sub-experiment).

Your job is to extract a concise list of experimental TASKS that are
DIRECTLY required to execute the PROTOCOL_TITLE experiment, not every
procedure described in the article.

STRICT RULES:
- Use ONLY the provided METHODS TEXT; do NOT hallucinate or invent tasks.
- Many methods sections contain multiple experiments. FIRST, mentally list
  all candidate experimental activities, THEN FILTER them to keep only those
  whose purpose and readout clearly match PROTOCOL_TITLE.
- Ignore tasks that belong only to other assays, controls, or unrelated
  experiments.
- Each remaining task must correspond to a MAJOR phase of the
  PROTOCOL_TITLE experiment (e.g., "prepare chromatin fraction", "perform
  PHB quantification assay").
- Produce ONLY as many tasks as needed to reproduce PROTOCOL_TITLE
  (typically 3–10). Do NOT over-segment trivial actions.
- Do not break a single logical step into multiple tasks, and do not combine
  unrelated procedures into one task.
- Return ONLY JSON with a top-level key 'tasks' containing an array
  of task objects.

For each task, provide:
- task_name (3–8 words)
- description (1–3 lines)
- goal (to + verb …; null if absent)
- span_chunk (source sentence/paragraph that justifies the task)
"""

    user_prompt = f"""ARTICLE_TITLE: {article_title}
PROTOCOL_TITLE: {bio_title}

METHODS TEXT:
{sec_text}
"""
    return system_prompt, user_prompt


def call_llm_task_miner(methods_text: str,
                        bio_title: str,
                        article_title: str,
                        protocol_id: str,
                        model: str = "gpt-4-1106-preview") -> List[Dict[str, Any]]:
    sys_prompt, user_prompt = build_task_miner_prompts(
        sec_text=methods_text,
        bio_title=bio_title,
        article_title=article_title,
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    parsed = json.loads(resp.choices[0].message.content)

    # 응답이 {"tasks": [...]} 형태라고 가정
    tasks = parsed.get("tasks", parsed)
    if isinstance(tasks, dict):
        tasks = [tasks]

    valid_tasks: List[Dict[str, Any]] = []
    for idx, task in enumerate(tasks):
        if not isinstance(task, dict):
            continue
        # 필수 필드 체크 (너가 task_miner_new.py에서 하던 validation 축약본)
        if not all(k in task for k in ("task_name", "description", "span_chunk")):
            continue

        task["task_id"] = f"{protocol_id}::T{idx + 1}"
        task["protocol_id"] = protocol_id
        valid_tasks.append(task)

    return valid_tasks


def task_planner_node(state: GraphState) -> GraphState:
    methods = state["methods_text"]
    bio_title = state["bio"]["title"]
    article_title = state["article"]["title"]
    protocol_id = state["protocol_id"]

    tasks = call_llm_task_miner(
        methods_text=methods,
        bio_title=bio_title,
        article_title=article_title,
        protocol_id=protocol_id,
    )

    new_state = dict(state)
    new_state["tasks"] = tasks
    return new_state
