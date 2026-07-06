# scripts/build_protocol_view.py
import json
from pathlib import Path
from typing import Dict, Any

GOLD_PATH = Path("data/gold_pairs_testset_v2.jsonl")
GEN_ACTIONS_PATH = Path("data/gen_actions_ir_10.jsonl")


def _load_gold_pairs() -> Dict[str, str]:
    """
    protocol_id -> methods_text(sec_text)
    gold_pairs_testset_v2.jsonl 구조에 맞게 key 조정 필요.
    """
    mapping: Dict[str, str] = {}
    with GOLD_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)

            # 🔴 여기 구조는 실제 파일 보고 맞춰줘야 함
            # 예시1) rec["protocol_id"], rec["article"]["sec_text"]
            # 예시2) rec["bio"]["id"], rec["article"]["sec_text"]
            pid = rec["protocol_id"]
            methods_text = rec["sec_text"]
            pmcid = rec["pmcid"]

            mapping[pid] = {
                "methods_text": methods_text,
                "pmcid": pmcid,
            }
    return mapping


def _load_gen_actions() -> Dict[str, Any]:
    """
    protocol_id -> actions 리스트
    """
    mapping: Dict[str, Any] = {}
    with GEN_ACTIONS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pid = rec["protocol_id"]
            mapping[pid] = rec["actions"]
    return mapping


def build_protocol_view(protocol_id: str) -> Dict[str, Any]:
    gold_map = _load_gold_pairs()
    gen_map = _load_gen_actions()

    if protocol_id not in gold_map:
        raise KeyError(f"{protocol_id} not found in gold_pairs_testset_v2.jsonl")
    if protocol_id not in gen_map:
        raise KeyError(f"{protocol_id} not found in gen_actions_ir_10.jsonl")

    return {
        "protocol_id": protocol_id,
        "pmcid": gold_map[protocol_id].get("pmcid"),
        "methods_text": gold_map[protocol_id].get("methods_text"),
        "actions": gen_map[protocol_id],
    }


if __name__ == "__main__":
    # 간단 sanity check
    test_id = "Bio-protocol-2096"  # 존재하는 id로 바꿔서 확인
    pv = build_protocol_view(test_id)
    print("protocol_id:", pv["protocol_id"])
    print("#actions:", len(pv["actions"]))
    print("methods_text preview:\n", pv["methods_text"][:500])
