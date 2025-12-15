# src/eval/gold_to_ir.py
import json
from typing import List

from src.types import StepIR


def hierarchical_to_steps_ir(bio: dict) -> List[StepIR]:
    prot_id = bio["id"]
    h = bio["hierarchical_protocol"]

    steps: List[StepIR] = []

    for key, value in h.items():
        # leaf node: 실제 step 문장
        if isinstance(value, str):
            step_id = key  # ex) "1.1.1"
            step_text = value

            step: StepIR = {
                "step_id": step_id,
                "task_id": step_id.split(".")[0],  # 대충 1,2 로 그룹
                "title": None,
                "action": None,  # 나중에 LLM or rule로 채워도 됨
                "step_text": step_text,
                "step_rationale": None,
                "span_chunk": None,
                "materials": [],
                "parameters": [],
                "evidence_spans": [],
                "verified": True,
                "verification_notes": "gold",
            }
            steps.append(step)

    # step_id 순서대로 정렬
    steps.sort(key=lambda s: s["step_id"])
    return steps


def main():
    with open("data/bio_protocol.json", "r") as f:
        bio_data = json.load(f)

    out_path = "data/gold_steps_ir.jsonl"
    with open(out_path, "w") as out:
        for rec in bio_data:
            steps = hierarchical_to_steps_ir(rec)
            out.write(json.dumps({
                "protocol_id": rec["id"],
                "steps": steps,
            }) + "\n")

    print(f"saved gold IR to {out_path}")


if __name__ == "__main__":
    main()
