# src/eval/eval_ab.py
import json
import os
from typing import Dict, List

from src.eval.metrics import step_match_scores, order_score_from_mapping

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")

GOLD_IR_PATH = os.path.join(DATA_DIR, "gold_steps_ir.jsonl")
A_PATH = os.path.join(DATA_DIR, "gen_steps_A.jsonl")
B_PATH = os.path.join(DATA_DIR, "gen_steps_B.jsonl")


def load_protocol_steps(path: str) -> Dict[str, List[dict]]:
    result = {}
    with open(path, "r") as f:
        for line in f:
            rec = json.loads(line)
            pid = rec["protocol_id"]
            result[pid] = rec["steps"]
    return result


def main():
    # gold / A / B 모두 로드
    gold = load_protocol_steps(GOLD_IR_PATH)
    pred_A = load_protocol_steps(A_PATH)
    pred_B = load_protocol_steps(B_PATH)

    protocol_ids = sorted(set(pred_A.keys()) & set(pred_B.keys()) & set(gold.keys()))

    print(f"Evaluating {len(protocol_ids)} protocols")

    def agg_results(pred_dict: Dict[str, List[dict]], name: str):
        precs, recs, f1s, orders = [], [], [], []

        for pid in protocol_ids:
            g_steps = gold[pid]
            p_steps = pred_dict[pid]

            prec, rec, f1, mapping = step_match_scores(g_steps, p_steps)
            ord_score = order_score_from_mapping(mapping)

            precs.append(prec)
            recs.append(rec)
            f1s.append(f1)
            orders.append(ord_score)

        def avg(lst):
            return sum(lst) / len(lst) if lst else 0.0

        print(f"\n=== {name} ===")
        print(f"Step Precision: {avg(precs):.4f}")
        print(f"Step Recall   : {avg(recs):.4f}")
        print(f"Step F1       : {avg(f1s):.4f}")
        print(f"Order Score   : {avg(orders):.4f}")

    agg_results(pred_A, "A: Methods → Step Structurer")
    agg_results(pred_B, "B: Task → Step Structurer")


if __name__ == "__main__":
    main()
