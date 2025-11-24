# scripts/task_miner_baseline.py

import argparse
import json
from pathlib import Path

from agents.planner_state import PlannerState
from agents.task_planner_node import run_task_planner


def iter_pairs(path: Path):
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(
        description="Baseline Task Planner (no RAG) on gold_pairs_testset."
    )
    parser.add_argument(
        "--pairs",
        type=str,
        required=True,
        help="Input JSONL of gold pairs (e.g., data/gold/gold_pairs_testset.jsonl)",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output JSONL path for predicted tasks (e.g., runs/tasks_baseline.jsonl)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI chat model name (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=12,
        help="Maximum number of tasks to generate per protocol.",
    )
    args = parser.parse_args()

    in_path = Path(args.pairs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as out_f:
        for obj in iter_pairs(in_path):
            protocol_id = obj["protocol_id"]
            # gold_pairs_testset.jsonl 에서 methods 텍스트 필드 이름에 맞게 사용
            # (네가 gold_pairs_testset에 sec_text로 저장해둔 상태)
            methods_text = obj.get("sec_text") or obj.get("text") or ""

            state = PlannerState(
                protocol_id=protocol_id,
                methods_text=methods_text,
                rag_enabled=False,  # 🔵 baseline: RAG 사용 안 함
                max_tasks=args.max_tasks,
            )

            state = run_task_planner(state, model=args.model)

            out_obj = {
                "protocol_id": protocol_id,
                "tasks": [t.model_dump() for t in state.tasks_planned],
                "llm_raw": state.planner_raw,
            }
            out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            out_f.flush()


if __name__ == "__main__":
    main()
