# scripts/step_structurer_from_tasks.py

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

from openai import OpenAI

client = OpenAI()


@dataclass
class TaskForPrompt:
    id: str  # T1, T2, ...
    orig_task_id: str  # Bio-protocol-2096::1
    title: str
    description: str


@dataclass
class Step:
    id: str  # S1, S2, ...
    task_id: str  # T1, T2, ...
    orig_task_id: str  # 매핑 보존
    title: str
    instruction: str
    expected_result: str


SYSTEM_PROMPT = """
You are an expert biological protocol editor.

You will receive:
1) A Methods / protocol text from a biology paper.
2) A list of high-level experimental TASKS (T1, T2, ...), each with a title and description.

Your job:
- Break each TASK into a small number of executable STEPS (S1, S2, ...).
- Steps must be:
  - Ordered and coherent.
  - At the level of "what the experimenter actually does" (but still high-level).
  - Grouped by task: each step must reference a task_id such as "T1", "T2", etc.

VERY IMPORTANT:
- Do NOT hallucinate detailed parameter values (temperature, volume, time, speed).
- If parameters are missing, keep them abstract (e.g. "incubate cells under appropriate conditions").
- Focus on logical structure and dependency (what comes before/after what).

Output JSON ONLY with this structure:

{
  "steps": [
    {
      "id": "S1",
      "task_id": "T1",
      "title": "Short step title",
      "instruction": "A concise description of what to do in this step.",
      "expected_result": "What state or output is expected after this step."
    },
    ...
  ]
}
"""


def build_user_prompt(
        protocol_title: str,
        protocol_text: str,
        tasks: List[TaskForPrompt],
        max_steps_per_task: int
) -> str:
    lines = []
    lines.append(f"PROTOCOL TITLE:\n{protocol_title}\n")
    lines.append("METHODS / PROTOCOL TEXT:\n")
    lines.append(protocol_text)
    lines.append("\n\nHIGH-LEVEL TASKS:\n")
    for t in tasks:
        lines.append(f"- {t.id}: {t.title}")
        if t.description:
            lines.append(f"  description: {t.description}")
    lines.append(
        f"\nFor each task (T1, T2, ...), generate about 2 to {max_steps_per_task} steps if possible.\n"
        "Return JSON with a flat list 'steps', each referencing a task_id.\n"
    )
    return "\n".join(lines)


def extract_json_from_text(text: str) -> Any:
    """
    모델이 ```json ... ```을 감싸거나 앞뒤에 텍스트를 붙여도
    가장 바깥쪽 { ... } 블록을 찾아 파싱.
    """
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError("No JSON object found in LLM output.")
    json_str = text[first:last + 1]
    return json.loads(json_str)


def call_step_structurer(
        model: str,
        protocol_title: str,
        protocol_text: str,
        tasks: List[TaskForPrompt],
        max_steps_per_task: int = 8,
) -> Tuple[List[Step], str]:
    """
    LLM 호출 → steps 리스트와 raw 텍스트 반환
    """
    user_prompt = build_user_prompt(protocol_title, protocol_text, tasks, max_steps_per_task)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    content = resp.choices[0].message.content or ""
    steps: List[Step] = []

    try:
        obj = extract_json_from_text(content)
        raw_steps = obj.get("steps", [])
        # alias → orig_task_id 매핑
        alias_to_orig = {t.id: t.orig_task_id for t in tasks}

        for idx, s in enumerate(raw_steps):
            task_id = s.get("task_id", "").strip()
            if not task_id:
                # task_id 누락된 경우는 스킵
                continue
            orig_task_id = alias_to_orig.get(task_id, "")

            step = Step(
                id=s.get("id", f"S{idx + 1}"),
                task_id=task_id,
                orig_task_id=orig_task_id,
                title=s.get("title", "").strip(),
                instruction=s.get("instruction", "").strip(),
                expected_result=s.get("expected_result", "").strip(),
            )
            steps.append(step)
    except Exception as e:
        # 파싱 실패 시 steps는 빈 리스트로 두고, llm_raw에 content만 남김
        print(f"[WARN] Failed to parse steps JSON: {e}")

    return steps, content


def load_pairs(path: str) -> Dict[str, Any]:
    data = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pid = obj["protocol_id"]
            data[pid] = obj
    return data


def load_tasks_grouped(path: str) -> Dict[str, List[TaskForPrompt]]:
    grouped: Dict[str, List[TaskForPrompt]] = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pid = obj["protocol_id"]
            title = obj.get("title", "")
            desc = obj.get("description", "")
            orig_task_id = obj.get("task_id", "")

            # 같은 protocol 안에서 순서대로 T1, T2,... 로 부여할 것이므로
            # 일단 임시로 넣고, 나중에 다시 번호를 매겨도 됨.
            grouped[pid].append(
                {
                    "orig_task_id": orig_task_id,
                    "title": title,
                    "description": desc,
                }
            )

    # T1, T2, ... ID 붙여서 TaskForPrompt 리스트로 변환
    grouped_final: Dict[str, List[TaskForPrompt]] = {}
    for pid, items in grouped.items():
        tasks_for_prompt: List[TaskForPrompt] = []
        for idx, it in enumerate(items):
            alias_id = f"T{idx + 1}"
            tasks_for_prompt.append(
                TaskForPrompt(
                    id=alias_id,
                    orig_task_id=it["orig_task_id"],
                    title=it["title"],
                    description=it["description"],
                )
            )
        grouped_final[pid] = tasks_for_prompt

    return grouped_final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=str, required=True,
                        help="gold_pairs_testset.jsonl path")
    parser.add_argument("--tasks", type=str, required=True,
                        help="tasks_baseline.jsonl path (one task per line)")
    parser.add_argument("--out", type=str, required=True,
                        help="output jsonl (steps per protocol)")
    parser.add_argument("--model", type=str, required=True,
                        help="OpenAI model name, e.g. gpt-4o-mini")
    parser.add_argument("--max-steps-per-task", type=int, default=8)

    args = parser.parse_args()

    pairs = load_pairs(args.pairs)
    tasks_grouped = load_tasks_grouped(args.tasks)

    print(f"Loaded {len(pairs)} gold pairs")
    print(f"Loaded tasks for {len(tasks_grouped)} protocols")

    with open(args.out, "w") as fout:
        for pid, tasks in tasks_grouped.items():
            if pid not in pairs:
                print(f"[WARN] protocol_id {pid} not found in pairs, skip")
                continue

            pair = pairs[pid]
            protocol_title = pair.get("title", "")
            # Methods / protocol 텍스트: gold_pairs 구조에 맞게 조정
            # 여기서는 bio.protocol 사용 (이미 flatten된 methods 텍스트)
            bio = pair.get("bio", {})
            protocol_text = bio.get("protocol", "")

            print(f"\n=== {pid} ===")
            print(f"  #tasks: {len(tasks)}")

            steps, llm_raw = call_step_structurer(
                model=args.model,
                protocol_title=protocol_title,
                protocol_text=protocol_text,
                tasks=tasks,
                max_steps_per_task=args.max_steps_per_task,
            )

            rec = {
                "protocol_id": pid,
                "steps": [asdict(s) for s in steps],
                "llm_raw": llm_raw,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()


if __name__ == "__main__":
    main()
