import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# SciBERT 로딩
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
model.eval()


def get_embedding(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    sim = np.dot(a, b) / denom
    return max(min(sim, 1.0), -1.0)  # 유효 범위 보정


def compute_task_scores(gold_tasks: List[str], pred_tasks: List[str], threshold: float = 0.65) -> Dict:
    gold_embs = [get_embedding(t) for t in gold_tasks]
    pred_embs = [get_embedding(t) for t in pred_tasks]

    matched_pred_idx = set()
    matched = 0

    for g_emb in gold_embs:
        best_sim = 0.0
        best_idx = -1
        for idx, p_emb in enumerate(pred_embs):
            if idx in matched_pred_idx:
                continue
            sim = cosine_similarity(g_emb, p_emb)
            if sim > threshold and sim > best_sim:
                best_sim = sim
                best_idx = idx
        if best_idx >= 0:
            matched_pred_idx.add(best_idx)
            matched += 1

    precision = matched / len(pred_tasks) if pred_tasks else 0.0
    recall = matched / len(gold_tasks) if gold_tasks else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "num_gold": len(gold_tasks),
        "num_pred": len(pred_tasks),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }


def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(gold_path: str, pred_path: str, out_csv: str):
    gold_data = load_jsonl(gold_path)
    pred_data = load_jsonl(pred_path)

    # gold: protocol_id → list of gold task titles
    gold_map = {
        item["protocol_id"]: [t["title"] for t in item["tasks"] if t.get("title")]
        for item in gold_data
    }

    # pred: protocol_id → list of predicted task_names
    pred_map = defaultdict(list)
    for item in pred_data:
        pid = item.get("protocol_id")
        if pid and item.get("task_name"):
            pred_map[pid].append(item["task_name"])

    result = []
    for pid in sorted(gold_map.keys()):
        gold_tasks = gold_map[pid]
        pred_tasks = pred_map.get(pid, [])
        scores = compute_task_scores(gold_tasks, pred_tasks)
        result.append({"protocol_id": pid, **scores})

    # Save
    out_path = Path(out_csv)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["protocol_id", "num_gold", "num_pred", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(result)

    print(f"✅ Saved to {out_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_path", required=True, help="Path to gold_tasks_testset.jsonl")
    parser.add_argument("--pred_path", required=True, help="Path to b1_tasks_new___.jsonl")
    parser.add_argument("--out_csv", required=True, help="Output CSV file path")
    args = parser.parse_args()

    main(args.gold_path, args.pred_path, args.out_csv)
