#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

from openai import OpenAI

client = OpenAI()

SYS_PROMPT = """
You are a meticulous protocol structurer. Convert the given Methods and pre-extracted STEPS into a graph IR.

STRICT OUTPUT JSON KEYS: nodes (array), edges (array), warnings (array).

NODE TYPES (use exact labels): Step, QCGate, DataAnalysis.

Step schema:
{id, type:'Step', title, action, materials: string[], params: Param[], produces: string[], task_ref}

Param schema (MANDATORY):
{name: string, value: number|null, unit: string|null, raw: string|null, source: 'parser'}

QCGate schema:
{id, type:'QCGate',
 measurement: {what, method, units},
 acceptance_criteria: {operator, lower, upper, unit} | null,
 decision: {on_accept, on_reject, max_retries},
 fallback: string|null,
 evidence_hint: {sent_idx: number[]}
}

DataAnalysis schema:
{id, type:'DataAnalysis', method, inputs: string[], params: Param[], outputs: string[], task_ref}

Edges: {from, to, label} with label in {'then','on_accept','on_reject'}.

YOU ARE GIVEN:
- methods_text: full Methods section of a biology protocol.
- steps: array of {id, task_id, orig_task_id, title, instruction, expected_result} that already segment the procedure.

INSTRUCTIONS:
1. For each input step, create exactly one Step node in nodes. Reuse the same id and title.
2. Derive 'action' as a short verb phrase from the instruction (e.g., 'centrifuge cells to pellet', 'wash cells').
3. Extract key reagents, buffers, media, kits, and equipment into materials.
4. Extract important numeric or categorical settings into params (e.g., speeds, times, temperatures, volumes, ratios, confluency, dilutions). Use the Param schema.
5. Optionally add QCGate nodes for explicit checks (e.g., 'until 80% confluent', 'if no pellet is visible, repeat'), and DataAnalysis nodes for analysis-only operations.
6. For Step.task_ref, copy the corresponding task_id from the input step (e.g., 'T6').
7. Build edges such that:
   - Within the same task_id, Step nodes form a 'then' chain in execution order.
   - When it is obvious that the output of one step feeds into a step in a different task, connect with a 'then' edge as well.
8. Put any ambiguity or limitation notes into warnings (strings).

Return ONE JSON object with exactly these top-level keys:
- nodes
- edges
- warnings
"""


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def save_jsonl(path: Path, records: List[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_methods_map(pairs_path: Path) -> Dict[str, str]:
    """
    gold_pairs_testset.jsonl 에서
    protocol_id -> methods_text(sec_text or text) 매핑 생성
    """
    methods_map: Dict[str, str] = {}
    for rec in load_jsonl(pairs_path):
        pid = rec.get("protocol_id")
        if not pid:
            continue
        methods = rec.get("sec_text") or rec.get("text") or ""
        methods_map[pid] = methods
    print(f"[INFO] Loaded Methods for {len(methods_map)} protocols from {pairs_path}")
    return methods_map


def call_llm_for_ir(methods_text: str, steps: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    user_payload = {
        "methods_text": methods_text,
        "steps": steps,
    }
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        temperature=0,
    )
    try:
        js = json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"[ERROR] Failed to parse LLM JSON, error={e}")
        js = {"nodes": [], "edges": [], "warnings": [f"llm_parse_error: {str(e)}"]}
    # 안전장치
    js.setdefault("nodes", [])
    js.setdefault("edges", [])
    js.setdefault("warnings", [])
    return js


def main():
    parser = argparse.ArgumentParser(
        description=(
            "LLM-based IR generator from pre-extracted steps.\n"
            "Input:  runs/steps_from_tasks_baseline.jsonl (protocol_id, steps[])\n"
            "        data/gold/gold_pairs_testset.jsonl   (protocol_id, sec_text/text)\n"
            "Output: runs/ir_graphs_from_steps_llm.jsonl (nodes, edges, warnings per protocol)"
        )
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="runs/steps_from_tasks_baseline.jsonl",
        help="Input JSONL with protocol_id and steps (baseline step_structurer output).",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default="data/gold/gold_pairs_testset.jsonl",
        help="Gold pairs JSONL for methods_text (sec_text/text).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/ir_graphs_from_steps_llm.jsonl",
        help="Output IR JSONL file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI model name for IR generation.",
    )
    parser.add_argument(
        "--max_protocols",
        type=int,
        default=0,
        help="Optional cap on number of protocols to process (0 = all).",
    )

    args = parser.parse_args()
    steps_path = Path(args.steps)
    pairs_path = Path(args.pairs)
    out_path = Path(args.output)

    assert steps_path.exists(), f"Steps file not found: {steps_path}"
    assert pairs_path.exists(), f"Pairs file not found: {pairs_path}"

    methods_map = load_methods_map(pairs_path)

    out_records: List[Dict[str, Any]] = []
    print(f"[INFO] Loading steps from {steps_path}")

    for i, rec in enumerate(load_jsonl(steps_path), start=1):
        protocol_id = rec.get("protocol_id")
        if not protocol_id:
            print("[WARN] Missing protocol_id in steps record, skipping.")
            continue

        methods_text = methods_map.get(protocol_id, "")
        if not methods_text:
            print(f"[WARN] No methods_text found for {protocol_id}, continuing with empty string.")

        steps = rec.get("steps") or rec.get("steps_structured") or []
        if not steps:
            print(f"[WARN] No steps found for {protocol_id}, skipping.")
            continue

        print(f"[INFO] [{i}] protocol_id={protocol_id}, #steps={len(steps)}")

        js = call_llm_for_ir(methods_text, steps, args.model)
        nodes = js.get("nodes", [])
        edges = js.get("edges", [])
        warnings = js.get("warnings", [])

        out_rec = {
            "protocol_id": protocol_id,
            "methods_text": methods_text,
            "nodes": nodes,
            "edges": edges,
            "warnings": warnings,
            # 참고용으로 task 통계, steps도 보존해두고 싶으면 아래처럼 추가 가능
            "n_steps_input": len(steps),
        }
        out_records.append(out_rec)

        if args.max_protocols and i >= args.max_protocols:
            print(f"[INFO] Reached max_protocols={args.max_protocols}, stopping.")
            break

    print(f"[INFO] Saving {len(out_records)} IR graphs to {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(out_path, out_records)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
