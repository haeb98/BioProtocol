import argparse
import json
from collections import defaultdict
from typing import List, Dict

from evaluate_tasks_semantic import (
    eval_string_exact,
    eval_keyword_overlap,
    eval_embedding_steps,
)


def load_gold_titles(path: str) -> Dict[str, List[str]]:
    gold = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pid = obj["protocol_id"]
            tasks = obj.get("tasks", [])
            titles = []
            for t in tasks:
                title = t.get("title") or t.get("name") or ""
                if title.strip():
                    titles.append(title)
            gold[pid] = titles
    return gold


def load_pred_titles(path: str) -> Dict[str, List[str]]:
    preds = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pid = obj["protocol_id"]
            title = obj.get("task_name", "")
            if title.strip():
                preds[pid].append(title)
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, help="gold_tasks_testset.jsonl")
    parser.add_argument("--pred", required=True, help="b1_tasks_new___.jsonl")
    parser.add_argument("--output", required=False, help="평가 결과 저장 경로")
    parser.add_argument("--keyword_top_k", type=int, default=5)
    parser.add_argument("--keyword_threshold", type=float, default=0.3)
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--embed_threshold", type=float, default=0.7)
    parser.add_argument("--embed_device", type=str, default=None)
    args = parser.parse_args()

    gold = load_gold_titles(args.gold)
    pred = load_pred_titles(args.pred)

    results = {}
    print("🔍 String-based evaluation...")
    results.update(eval_string_exact(gold, pred))

    print("🔍 Keyword-overlap evaluation...")
    results.update(eval_keyword_overlap(gold, pred,
                                        top_k=args.keyword_top_k,
                                        threshold=args.keyword_threshold))

    print("🔍 Embedding-based step evaluation...")
    results.update(eval_embedding_steps(
        gold,
        pred,
        model_name=args.embed_model,
        threshold=args.embed_threshold,
        device=args.embed_device
    ))

    print("\n=== Task Mining Evaluation Results ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Saved to {args.output}")


if __name__ == "__main__":
    main()
