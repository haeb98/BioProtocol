# scripts/check_order_changes.py
import json

RAW_PATH = "data/gen_actions_ir_10.jsonl"
ORD_PATH = "data/gen_actions_ordered_ir_10.jsonl"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    raw = load_jsonl(RAW_PATH)
    ordered = load_jsonl(ORD_PATH)
    ordered_map = {rec["protocol_id"]: rec for rec in ordered}

    print("Protocol ID\t#Actions\t#Changed")
    for rec in raw:
        pid = rec["protocol_id"]
        raw_actions = rec["actions"]
        ord_actions = ordered_map[pid]["actions"]

        raw_ids = [a["action_id"] for a in raw_actions]
        ord_ids = [a["action_id"] for a in ord_actions]

        # sanity check: 구성은 같은지
        assert set(raw_ids) == set(ord_ids), f"Action set mismatch for {pid}"

        changed = sum(1 for i in range(len(raw_ids)) if raw_ids[i] != ord_ids[i])

        print(f"{pid}\t{len(raw_ids)}\t{changed}")


if __name__ == "__main__":
    main()
