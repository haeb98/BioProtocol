#!/usr/bin/env python3
"""
Task Planner using a local Llama model for generating high‑level experimental tasks.

This script reads Methods sections from a test set (gold_pairs_testset.jsonl) and
generates a list of high‑level tasks for each protocol using a locally
loaded Llama chat model (e.g. Llama‑2‑7b‑chat). It uses HuggingFace
transformers to load the model and tokenizer, formats the prompt according to
Llama‑2 chat conventions, and decodes the output into JSON.

Usage example:

    python task_planner_llama.py \
      --pairs data/gold/gold_pairs_testset.jsonl \
      --model-path /path/to/Llama2 \
      --out runs/tasks_llama.jsonl \
      --max-tasks 8

Notes:
  * This script assumes you have previously downloaded a Llama chat model
    (e.g. meta‑llama/Llama‑2‑7b‑chat‑hf) into a local directory and that
    you have accepted any required licenses for those weights.
  * Llama chat models require prompts to be wrapped in the special
    <s>[INST] … [/INST] format with an optional <<SYS>> system prompt. If
    you modify the build_prompt function you must preserve this format.
  * Generation parameters (temperature, top_p, max_new_tokens) may need
    tuning for best results. Here we use conservative defaults.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def build_prompt(methods_text: str, max_tasks: int) -> str:
    """
    Construct the Llama chat prompt with a system instruction and a user
    message. The prompt instructs the model to extract a handful of high‑level
    tasks (4–8 by default) from a Methods section, including the title,
    description, materials list, and goal for each task. The model is told to
    format its response as JSON only.

    Parameters
    ----------
    methods_text : str
        The raw Methods section text for a single protocol.
    max_tasks : int
        Maximum number of tasks to extract. Models may generate fewer tasks if
        the Methods text is short or contains fewer logical phases.

    Returns
    -------
    str
        A formatted prompt string ready to be fed into a Llama chat model.
    """
    # System prompt: instruct the model about its role and the expected
    # structure of the response. Keep the system prompt short to leave more
    # room for the user content.
    system_prompt = (
        "You are an expert experimental protocol assistant. "
        "Given a Methods section from a scientific paper, extract a small "
        "number of high‑level experimental tasks (ideally between 4 and 12) that "
        "a scientist must perform to reproduce the experiment. For each task, "
        "provide: (1) a concise title no more than 12 words, (2) a brief "
        "description summarizing what happens in the task, (3) a list of key "
        "materials needed (use names exactly as they appear), and (4) a goal "
        "phrase starting with 'to' that explains the purpose of the task. "
        "Do not invent tasks or materials that are not mentioned in the text. "
        "Only return valid JSON; do not include any narrative outside the JSON."
    )

    # User prompt: include the Methods section and explicit instructions on
    # format. Note that Llama chat models expect the user prompt after the
    # system prompt. We instruct the model to output a JSON list with
    # appropriate fields.
    user_prompt = (
        f"[Methods Section]\n{methods_text}\n\n"
        "Extract tasks from the Methods section above and return a JSON array of "
        "objects with keys 'title', 'description', 'materials' (an array of "
        "strings), and 'goal'. The JSON must be valid and contain no extra "
        "fields. Limit the number of tasks to {max_tasks} or fewer."
    )

    # Wrap in Llama chat format. The system prompt goes inside <<SYS>> tags
    # and the user prompt follows after a blank line. The closing [/INST]
    # indicates where the model should start its completion.
    prompt = (
            "<s>[INST] <<SYS>>\n" + system_prompt + "\n<</SYS>>\n\n" + user_prompt + " [/INST]"
    )
    return prompt


def generate_tasks_for_protocol(
        model,
        tokenizer,
        methods_text: str,
        max_tasks: int,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_new_tokens: int = 1024,
) -> Optional[List[Dict[str, Any]]]:
    """
    Generate a list of tasks from a Methods section using a local Llama model.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        A loaded causal LM such as Llama‑2‑7b‑chat.
    tokenizer : transformers.PreTrainedTokenizer
        Corresponding tokenizer for the model.
    methods_text : str
        The Methods section for which tasks will be generated.
    max_tasks : int
        Maximum number of tasks to extract. Passed to build_prompt.
    temperature : float
        Sampling temperature for generation. Lower values produce more
        deterministic output.
    top_p : float
        Nucleus sampling top‑p parameter.
    max_new_tokens : int
        Maximum number of new tokens the model may generate for the response.

    Returns
    -------
    Optional[List[Dict[str, Any]]]
        A list of task dictionaries or None if JSON parsing fails.
    """
    prompt = build_prompt(methods_text, max_tasks)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode and strip off the prompt to get only the completion
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Find the first "{" in the output to locate the JSON
    json_start = decoded.find("[")  # expecting a JSON array
    if json_start == -1:
        return None
    json_str = decoded[json_start:].strip()
    # Attempt to parse JSON; handle trailing text gracefully
    try:
        tasks = json.loads(json_str)
        # Ensure tasks is a list of dicts
        if not isinstance(tasks, list):
            return None
        return tasks
    except json.JSONDecodeError:
        # If parsing fails, return None so caller can log a warning
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Task Planner using local Llama model")
    parser.add_argument(
        "--pairs",
        type=str,
        required=True,
        help="Path to gold_pairs_testset.jsonl containing Methods sections.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the directory containing the local Llama chat model and tokenizer.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output JSONL file to save the tasks.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=12,
        help="Maximum number of tasks to extract per protocol.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top‑p parameter.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate for the response.",
    )
    args = parser.parse_args()

    # Load model and tokenizer
    print(f"Loading model from {args.model_path} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16)
    # Use MPS if available, otherwise CPU/GPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    print(f"Model loaded to {device}", file=sys.stderr)

    # Read Methods sections from pairs file
    in_path = Path(args.pairs)
    if not in_path.is_file():
        print(f"Pairs file not found: {in_path}", file=sys.stderr)
        sys.exit(1)
    out_path = Path(args.out)
    count = 0
    with open(in_path, "r", encoding="utf-8") as infile, open(out_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            record = json.loads(line)
            protocol_id = record.get("protocol_id")
            methods_text = record.get("sec_text") or record.get("text")
            if not methods_text:
                continue
            tasks = generate_tasks_for_protocol(
                model,
                tokenizer,
                methods_text,
                max_tasks=args.max_tasks,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
            )
            result = {
                "protocol_id": protocol_id,
                "tasks": tasks,
            }
            outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
            count += 1
            if count % 10 == 0:
                print(f"Processed {count} protocols...", file=sys.stderr)
    print(f"Done. Wrote {count} results to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
