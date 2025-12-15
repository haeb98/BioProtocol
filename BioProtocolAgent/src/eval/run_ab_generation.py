# src/eval/run_ab_generation.py
import json
import os
from typing import List

from src.data_loader import PAIRS_INDEX, make_initial_state
from src.graph_builder import build_graph
from src.graph_builder_A import build_graph_A
from tqdm import tqdm

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")

OUT_A = os.path.join(DATA_DIR, "gen_steps_A.jsonl")
OUT_B = os.path.join(DATA_DIR, "gen_steps_B.jsonl")


def main():
    # gold_pairs_testset_v2.jsonl에 포함된 protocol_id 46개
    protocol_ids: List[str] = list(PAIRS_INDEX.keys())
    protocol_ids.sort()

    graph_A = build_graph_A()
    graph_B = build_graph()

    # A 조건 실행
    with open(OUT_A, "w") as fa:
        for pid in tqdm(protocol_ids, desc="Running Graph A"):
            state = make_initial_state(pid)
            out = graph_A.invoke(state)
            steps = out.get("steps_raw", [])
            fa.write(json.dumps({
                "protocol_id": pid,
                "steps": steps,
            }) + "\n")

    # B 조건 실행
    with open(OUT_B, "w") as fb:
        for pid in tqdm(protocol_ids, desc="Running Graph B"):
            state = make_initial_state(pid)
            out = graph_B.invoke(state)
            steps = out.get("steps_raw", [])
            fb.write(json.dumps({
                "protocol_id": pid,
                "steps": steps,
            }) + "\n")
            
    print(f"Saved A results to {OUT_A}")
    print(f"Saved B results to {OUT_B}")


if __name__ == "__main__":
    main()
