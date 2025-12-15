# scripts/gen_gen_actions_ir_10.py

import json
import sys
from pathlib import Path
from typing import List

from tqdm import tqdm

# --- 프로젝트 루트 sys.path 추가 ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.graph_builder import build_graph
from src.data_loader import make_initial_state
from src.types import GraphState, ActionIR

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

OUT_PATH = Path("data/gen_actions_ir_10.jsonl")


def main():
    graph = build_graph()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w") as fout:
        for pid in tqdm(TARGET_PROTOCOL_IDS, desc="Generating gen_actions"):
            try:
                init_state: GraphState = make_initial_state(pid)
            except Exception as e:
                print(f"[gen_actions] ERROR: cannot make_initial_state({pid}): {e}")
                continue

            try:
                final_state: GraphState = graph.invoke(init_state)
            except Exception as e:
                print(f"[gen_actions] ERROR: graph.invoke failed for {pid}: {e}")
                continue

            actions: List[ActionIR] = final_state.get("actions", [])
            print(f"[gen_actions] {pid}: {len(actions)} actions")

            out_rec = {
                "protocol_id": pid,
                "actions": actions,
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    print(f"[gen_actions] saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
