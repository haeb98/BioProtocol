"""
SciBERT 기반 Step-M 평가 스크립트 (Graph A vs Graph B 비교)

사용법:
    (.venv) python -m src.eval.eval_ab_scibert
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from src.eval.metrics_sci import (
    step_match_scores_scibert,
    order_score_from_mapping,
)

# ===== 경로 설정 (필요시 수정) =====
DATA_DIR = Path("data")

GOLD_STEPS_PATH = DATA_DIR / "gold_steps_ir.jsonl"
GEN_A_PATH = DATA_DIR / "gen_steps_A.jsonl"  # Graph A: Methods → Step
GEN_B_PATH = DATA_DIR / "gen_steps_B.jsonl"  # Graph B: Task → Step

# SciBERT cosine similarity threshold
SIM_THRESHOLD = 0.7


def load_steps_by_protocol(path: Path) -> Dict[str, List[dict]]:
    """
    jsonl 파일을 읽어서 protocol_id → [steps...] 형태로 변환
    각 라인은 {"protocol_id": str, "steps": [...]} 형태라고 가정
    """
    by_pid: Dict[str, List[dict]] = defaultdict(list)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec.get("protocol_id")
            steps = rec.get("steps", [])
            if pid is None:
                continue
            by_pid[pid] = steps
    return by_pid


def evaluate_run_scibert(
        gold_by_pid: Dict[str, List[dict]],
        pred_by_pid: Dict[str, List[dict]],
        label: str,
        sim_threshold: float = SIM_THRESHOLD,
) -> None:
    """
    gold / pred 를 받아 SciBERT 기반 Step-M & Order score를 프로토콜별 평균으로 출력
    """
    common_pids = sorted(set(gold_by_pid.keys()) & set(pred_by_pid.keys()))
    if not common_pids:
        print(f"[{label}] No common protocol_ids between gold and pred.")
        return

    all_prec: List[float] = []
    all_rec: List[float] = []
    all_f1: List[float] = []
    all_order: List[float] = []

    for pid in common_pids:
        gold_steps = gold_by_pid[pid]
        pred_steps = pred_by_pid.get(pid, [])

        p, r, f1, mapping = step_match_scores_scibert(
            gold_steps, pred_steps, sim_threshold=sim_threshold
        )
        order = order_score_from_mapping(mapping)

        all_prec.append(p)
        all_rec.append(r)
        all_f1.append(f1)
        all_order.append(order)

    n = len(common_pids)
    avg_p = sum(all_prec) / n
    avg_r = sum(all_rec) / n
    avg_f1 = sum(all_f1) / n
    avg_order = sum(all_order) / n

    print(f"\n=== {label} (SciBERT-based Step-M, thresh={sim_threshold}) ===")
    print(f"Protocols evaluated : {n}")
    print(f"Step Precision      : {avg_p:.4f}")
    print(f"Step Recall         : {avg_r:.4f}")
    print(f"Step F1             : {avg_f1:.4f}")
    print(f"Order Score         : {avg_order:.4f}")


def main() -> None:
    print("[INFO] Loading gold and prediction files...")

    gold_by_pid = load_steps_by_protocol(GOLD_STEPS_PATH)
    pred_A_by_pid = load_steps_by_protocol(GEN_A_PATH)
    pred_B_by_pid = load_steps_by_protocol(GEN_B_PATH)

    print(f"[INFO] Gold protocols : {len(gold_by_pid)}")
    print(f"[INFO] A   protocols  : {len(pred_A_by_pid)}")
    print(f"[INFO] B   protocols  : {len(pred_B_by_pid)}")

    # Graph A 평가
    evaluate_run_scibert(
        gold_by_pid=gold_by_pid,
        pred_by_pid=pred_A_by_pid,
        label="Graph A: Methods → Step Structurer",
    )

    # Graph B 평가
    evaluate_run_scibert(
        gold_by_pid=gold_by_pid,
        pred_by_pid=pred_B_by_pid,
        label="Graph B: Task → Step Structurer",
    )


if __name__ == "__main__":
    main()
