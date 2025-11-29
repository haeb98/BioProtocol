import json
import os
import time
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_sys_prompt():
    return (
        "You are a lab assistant that extracts step-by-step experimental procedures "
        "for a given task from a biological methods section.\n"
        "Follow these strict rules:\n"
        "- Use only the content provided in the input. Do not hallucinate.\n"
        "- Break down the task into a sequence of concrete steps in correct order.\n"
        "- Each step must include a description (step_text), rationale (step_rationale), "
        "the exact supporting text span (span_chunk), and list key parameters and materials.\n"
        "- Output the result strictly as a JSON object with top-level key 'steps', like this:\n\n"
        "{ \"steps\": [ {\"step_text\": ..., \"step_rationale\": ..., \"span_chunk\": ..., "
        "\"parameters\": [...], \"materials\": [...] }, ... ] }\n"
    )


def get_user_prompt(task_name: str, description: str, goal: str, span_chunk: str, full_text: str) -> str:
    return (
        f"TASK NAME: {task_name}\n"
        f"DESCRIPTION: {description}\n"
        f"GOAL: {goal}\n"
        f"METHODS TEXT:\n{full_text}\n"
        f"SPAN CONTEXT:\n{span_chunk}"
    )


def call_llm_json(client, model: str, sys_prompt: str, user_payload: str,
                  temperature: float = 0.3, max_retries: int = 3) -> Dict:
    last_err = None
    for i in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_payload}
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"LLM call failed: {last_err}")


def main(args):
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("ERROR: OPENAI_API_KEY not set")

    client = OpenAI()
    task_data = load_jsonl(args.task_file)
    article_data = load_jsonl(args.article_file)

    article_map = {item["protocol_id"]: item for item in article_data}
    all_results = []

    tmp_out_file = args.out_file + ".tmp"
    Path(tmp_out_file).parent.mkdir(parents=True, exist_ok=True)

    for idx, task in enumerate(tqdm(task_data, desc="Structuring Steps")):
        protocol_id = task["protocol_id"]
        task_id = task["task_id"]
        task_name = task["task_name"]
        description = task.get("description", "")
        goal = task.get("goal", "")
        span_chunk = task.get("span_chunk", "")

        try:
            sec_text = article_map[protocol_id]["sec_text"]
            sys_prompt = get_sys_prompt()
            user_prompt = get_user_prompt(task_name, description, goal, span_chunk, sec_text)
            result = call_llm_json(client, args.model, sys_prompt, user_prompt)

            steps = result.get("steps", [])
            if isinstance(steps, dict):
                steps = [steps]
            for step_idx, step in enumerate(steps):
                step.update({
                    "step_id": f"{task_id}::S{step_idx + 1}",
                    "protocol_id": protocol_id,
                    "task_id": task_id,
                    "task_name": task_name
                })
                all_results.append(step)

        except Exception as e:
            print(f"⚠️ Error in {protocol_id}/{task_id}: {e}")

        # ✅ 임시 저장 주기
        if (idx + 1) % args.save_every == 0:
            write_jsonl(all_results, tmp_out_file)
            print(f"💾 Intermediate saved ({len(all_results)} steps) → {tmp_out_file}")

    # 🔚 최종 저장
    write_jsonl(all_results, args.out_file)
    print(f"✅ Final saved ({len(all_results)} steps) → {args.out_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_file", required=True)
    parser.add_argument("--article_file", required=True)
    parser.add_argument("--out_file", required=True, help="Output file to save JSONL")
    parser.add_argument("--model", default="gpt-4-1106-preview")
    parser.add_argument("--save_every", type=int, default=10, help="Save every N records")
    args = parser.parse_args()
    main(args)
