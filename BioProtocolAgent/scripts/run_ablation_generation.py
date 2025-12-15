# scripts/run_ablation_generation_10.py
import json
import sys
from pathlib import Path

from tqdm import tqdm

# --- 프로젝트 루트 sys.path 추가 ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.protocol_ids_10 import PROTOCOL_IDS_10
from src.data_loader import make_initial_state
from src.graph_builder_ablation import build_graph

MODES = ["P6"]
# MODES = ["P1", "P2", "P3", "P4", "P5", "P6"]
OUT_DIR = Path("data/ablation")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    for mode in MODES:
        print(f"\n=== Running mode {mode} ===")
        graph = build_graph(mode)
        out_path = OUT_DIR / f"gen_actions_{mode}_10.jsonl"

        with out_path.open("w", encoding="utf-8") as f_out:
            for pid in tqdm(PROTOCOL_IDS_10):
                state = make_initial_state(pid)
                final_state = graph.invoke(state)
                actions = final_state["actions"]
                rec = {
                    "protocol_id": pid,
                    "actions": actions,
                    "protocol_text": final_state.get("protocol_text"),
                }
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
