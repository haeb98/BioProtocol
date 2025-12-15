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


def get_prompts(sec_text: str, bio_title: str, article_title: str):
    system_prompt = """
You are an expert biomedical scientist helping to extract experimental TASKS
from the methods sections of biology papers.

You are given:
- ARTICLE_TITLE: the overall paper title.
- PROTOCOL_TITLE: the specific Bio-protocol title (a particular assay or sub-experiment).

Your job is to extract a concise list of experimental TASKS that are DIRECTLY required to execute the PROTOCOL_TITLE experiment, not every
procedure described in the article.

STRICT RULES:
- Use ONLY the provided METHODS TEXT; do NOT hallucinate or invent tasks.
- Many methods sections contain multiple experiments. FIRST, mentally list all candidate experimental activities, THEN FILTER them to keep only those
  whose purpose and readout clearly match PROTOCOL_TITLE.
- Ignore tasks that belong only to other assays, controls, or unrelated experiments.
- Each remaining task must correspond to a MAJOR phase of the PROTOCOL_TITLE experiment (e.g., "prepare chromatin fraction", "perform
  PHB quantification assay").
- Produce ONLY as many tasks as needed to reproduce PROTOCOL_TITLE(typically 3–10). Do NOT over-segment trivial actions.
- Do not break a single logical step into multiple tasks, and do not combine unrelated procedures into one task.
- Return ONLY JSON with a top-level key 'tasks' containing an array of task objects.

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


def call_llm_json(client, model: str, sys_prompt: str, user_prompt: str,
                  temperature: float = 0.0, max_retries: int = 3) -> Dict:
    last_err = None
    for i in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"LLM call failed: {last_err}")


def main(args):
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("ERROR: OPENAI_API_KEY not set")

    client = OpenAI()
    input_data = load_jsonl(args.in_file)
    output_tasks = []
    discarded_tasks = []

    Path(args.out_dir).mkdir(exist_ok=True, parents=True)
    task_path = Path(args.out_dir) / "b1_tasks_new_title.jsonl"
    discard_path = Path(args.out_dir) / "b1_discarded_tasks_.jsonl"

    for record in tqdm(input_data, desc="Mining Tasks"):
        protocol_id = record["protocol_id"]
        sec_text = record["sec_text"]
        bio_title = record["bio"]["title"]
        article_title = record["article"]["title"]

        try:
            sys_prompt, user_prompt = get_prompts(sec_text=sec_text, bio_title=bio_title, article_title=article_title)
            parsed = call_llm_json(client, args.model, sys_prompt=sys_prompt, user_prompt=user_prompt)

            if isinstance(parsed, dict) and "tasks" in parsed:
                tasks = parsed["tasks"]
            elif isinstance(parsed, list):
                print(f"⚠️ Response was a list, wrapping into 'tasks' for {protocol_id}")
                tasks = parsed
            else:
                print(f"⚠️ Unexpected response format from LLM for {protocol_id}: {parsed}")
                continue

            if isinstance(tasks, dict):
                print(f"⚠️ Only one task returned for {protocol_id}, wrapping into list.")
                tasks = [tasks]

            valid_tasks = []
            for idx, task in enumerate(tasks):
                if not isinstance(task, dict):
                    raise ValueError(f"Task is not a dictionary: {task}")
                if not all(k in task for k in ("task_name", "description", "goal", "span_chunk")):
                    raise ValueError(f"Incomplete task object: {task}")
                task["task_id"] = f"{protocol_id}::T{idx + 1}"
                task.update({
                    "protocol_id": protocol_id,
                    "pmcid": record["pmcid"],
                    "section_list": record["article"].get("section_list", [])
                })
                valid_tasks.append(task)

            if args.max_tasks and len(valid_tasks) > args.max_tasks:
                output_tasks.extend(valid_tasks[:args.max_tasks])
                discarded_tasks.extend(valid_tasks[args.max_tasks:])
            else:
                output_tasks.extend(valid_tasks)

        except Exception as e:
            print(f"⚠️ Error in {protocol_id}: {e}")

    write_jsonl(output_tasks, task_path)
    write_jsonl(discarded_tasks, discard_path)
    print(f"✅ Saved {len(output_tasks)} tasks to {task_path}")
    print(f"🗑️ Discarded {len(discarded_tasks)} excess tasks to {discard_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model", default="gpt-4")
    parser.add_argument("--max_tasks", type=int, default=None)
    args = parser.parse_args()
    main(args)
