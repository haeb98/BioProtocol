# scripts/run_verifier_demo_react.py
import json
import sys
from pathlib import Path

# --- 프로젝트 루트 sys.path 추가 ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.build_protocol_view import build_protocol_view
from src.nodes.verifier_react import verify_actions

TARGET_PROTOCOL = "Bio-protocol-3607"  # 원하는 id로 변경

OUT_IR = Path("data/verifier_demo_bio3607_ir_chain.json")
OUT_TRACE = Path("data/verifier_demo_bio3607_trace_chain.json")


def main():
    proto = build_protocol_view(TARGET_PROTOCOL)
    print(proto["pmcid"])
    result = verify_actions(proto)

    # IR 결과
    with OUT_IR.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "protocol_id": result["protocol_id"],
                "actions_verified": result["actions_verified"],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # trace 결과
    with OUT_TRACE.open("w", encoding="utf-8") as f:
        json.dump(
            result["verification_traces"],
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"✅ Verified IR saved to {OUT_IR}")
    print(f"✅ Verification traces saved to {OUT_TRACE}")


if __name__ == "__main__":
    main()
