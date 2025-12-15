# scripts/gen_gold_actions_ir_10.py

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# --- 프로젝트 루트 경로를 sys.path에 추가 (scripts/ 기준 한 단계 위) ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

# 이제부터는 src.* 임포트 가능
from src.nodes.action_extractor import call_llm_action_extractor_for_step
from src.types import StepIR, ActionIR

# --- 1) 여기 네가 고른 10개 protocol_id 넣기 ---
TARGET_PROTOCOL_IDS = [
    "Bio-protocol-2302",
    "Bio-protocol-972",
    "Bio-protocol-1111",
    "Bio-protocol-2617",
    "Bio-protocol-3607",
    "Bio-protocol-3584",
    "Bio-protocol-3617",
    "Bio-protocol-851",
    "Bio-protocol-1373",
    "Bio-protocol-2096",
]

PAIRS_PATH = Path("data/gold_pairs_testset_v2.jsonl")
OUT_PATH = Path("data/gold_actions_ir_10.jsonl")


def flatten_hierarchical_protocol(proto_id: str,
                                  hier: Dict[str, Any]) -> List[StepIR]:
    """
    BioProBench의 hierarchical_protocol 딕셔너리를 leaf step으로 평탄화해서
    StepIR 형태 리스트로 만든다.

    key 예시:
      "1"        -> 섹션 title (문장 아님)
      "1.1"      -> subsection title
      "1.1.1"    -> 실제 step 텍스트

    여기서는 value가 'str'인 것만 step으로 본다.
    """
    steps: List[StepIR] = []

    # 정렬해서 순서 유지
    for key in sorted(hier.keys(), key=lambda x: [int(p) if p.isdigit() else p for p in x.split(".")]):
        val = hier[key]
        if isinstance(val, str):
            # leaf step
            step_id = f"{proto_id}::{key}"
            # task_id는 가장 상위 숫자만 사용 (예: "1.2.3" -> "1")
            top = key.split(".")[0]
            task_id = f"{proto_id}::T{top}"

            steps.append(StepIR(
                step_id=step_id,
                task_id=task_id,
                title=None,
                action=None,
                step_text=val,
                step_rationale="",
                span_chunk=val,
                materials=[],
                parameters=[],
                evidence_spans=[],
                verified=False,
                verification_notes="",
                parameters_llm_guess=[],
                materials_llm_guess=[],
                task_name="",
                protocol_id=proto_id,
            ))

    return steps


def main():
    print("[gold_actions] loading pairs...")
    records = []
    with PAIRS_PATH.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("protocol_id") in TARGET_PROTOCOL_IDS:
                records.append(rec)

    print(f"[gold_actions] target records: {len(records)} (expected <= {len(TARGET_PROTOCOL_IDS)})")

    with OUT_PATH.open("w") as fout:
        for rec in records:
            pid = rec["protocol_id"]
            bio = rec["bio"]
            hier = bio.get("hierarchical_protocol", {})
            if not hier:
                print(f"[gold_actions] WARNING: no hierarchical_protocol for {pid}, skip")
                continue

            steps = flatten_hierarchical_protocol(pid, hier)

            # gold에서는 methods_text 대신 bio 프로토콜 텍스트 사용
            proto_text = bio.get("protocol", "")

            all_actions: List[ActionIR] = []
            for step in steps:
                actions = call_llm_action_extractor_for_step(step, proto_text)
                all_actions.extend(actions)

            out_rec = {
                "protocol_id": pid,
                "actions": all_actions,
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            print(f"[gold_actions] wrote actions for {pid}: {len(all_actions)} actions")

    print(f"[gold_actions] saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
