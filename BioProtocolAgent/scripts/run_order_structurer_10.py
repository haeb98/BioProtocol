# scripts/run_order_structurer_10.py
import json
import sys
from pathlib import Path
from typing import Dict, Any

# --- 프로젝트 루트 sys.path 추가 ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.nodes.order_structurer import call_llm_order_structurer_for_actions

IN_PATH = Path("data/gen_actions_ir_10.jsonl")
OUT_PATH = Path("data/gen_actions_ordered_ir_10.jsonl")


def main():
    print(f"[order] reading {IN_PATH} ...")
    lines = IN_PATH.read_text(encoding="utf-8").splitlines()

    out_lines = []
    for line in lines:
        if not line.strip():
            continue
        rec = json.loads(line)
        pid = rec["protocol_id"]
        methods = rec.get("methods_text", "")
        actions = rec.get("actions", [])

        print(f"[order] protocol {pid}, {len(actions)} actions -> ordering...")

        ordered_actions = call_llm_order_structurer_for_actions(
            protocol_id=pid,
            methods_text=methods,
            actions=actions,
        )

        rec_out: Dict[str, Any] = dict(rec)
        rec_out["actions_ordered"] = ordered_actions  # 새 필드로 넣거나
        # 혹은 아래처럼 actions 자체를 덮어써도 됨:
        # rec_out["actions"] = ordered_actions

        out_lines.append(json.dumps(rec_out, ensure_ascii=False))

    OUT_PATH.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"[order] wrote ordered actions to {OUT_PATH}")


if __name__ == "__main__":
    main()
