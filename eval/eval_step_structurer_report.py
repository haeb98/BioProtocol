import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# Load SciBERT
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")


def get_embedding(text: str) -> np.ndarray:
    if not text.strip():
        return np.zeros(768)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def match_steps(pred_steps: List[str], gold_steps: List[str], threshold=0.7):
    gold_embs = [get_embedding(t) for t in gold_steps if t.strip()]
    pred_embs = [get_embedding(t) for t in pred_steps if t.strip()]
    matched = 0
    for g in gold_embs:
        if any(cosine_similarity(g, p) >= threshold for p in pred_embs):
            matched += 1
    return matched


def get_param_mentions(step: Dict) -> int:
    return len(step.get("parameters", []))


def get_material_mentions(step: Dict) -> int:
    return len(step.get("materials", []))


def hallucination_rate(pred_steps: List[Dict], sec_text: str) -> float:
    hallucinated = 0
    for s in pred_steps:
        if s.get("span_chunk", "") not in sec_text:
            hallucinated += 1
    return hallucinated / len(pred_steps) if pred_steps else 0.0


def order_score(pred_steps: List[str], gold_steps: List[str]) -> float:
    matched_order = 0
    matched_pairs = min(len(pred_steps), len(gold_steps))
    for i in range(matched_pairs):
        if pred_steps[i].strip().lower() == gold_steps[i].strip().lower():
            matched_order += 1
    return matched_order / matched_pairs if matched_pairs else 0.0


def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_per_protocol(gold_path: str, pred_path: str, article_path: str, out_csv: str):
    gold_data = load_jsonl(gold_path)
    pred_data = load_jsonl(pred_path)
    sec_texts = {x["protocol_id"]: x["sec_text"] for x in load_jsonl(article_path)}

    gold_map = {g["protocol_id"]: [s["text"] for s in g["steps"]] for g in gold_data}
    pred_steps_grouped = defaultdict(list)
    for step in pred_data:
        pred_steps_grouped[step["protocol_id"]].append(step)

    rows = []
    for pid in gold_map:
        gold_steps = gold_map[pid]
        pred_steps = pred_steps_grouped.get(pid, [])
        pred_step_texts = [s["step_text"] for s in pred_steps if s.get("step_text")]

        matched = match_steps(pred_step_texts, gold_steps)
        precision = matched / len(pred_step_texts) if pred_step_texts else 0.0
        recall = matched / len(gold_steps) if gold_steps else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        total_params = sum(get_param_mentions(s) for s in pred_steps)
        total_materials = sum(get_material_mentions(s) for s in pred_steps)
        halluc_rate = hallucination_rate(pred_steps, sec_texts.get(pid, ""))
        order = order_score(pred_step_texts, gold_steps)

        rows.append({
            "protocol_id": pid,
            "num_gold": len(gold_steps),
            "num_pred": len(pred_step_texts),
            "matched_steps": matched,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "param_count": total_params,
            "material_count": total_materials,
            "hallucination_rate": round(halluc_rate, 4),
            "order_score": round(order, 4)
        })

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Evaluation saved to {out_csv}")


if __name__ == "__main__":
    evaluate_per_protocol(
        gold_path="data/gold/gold_steps_testset.jsonl",
        pred_path="runs/b2_steps_new.jsonl",
        article_path="data/gold/gold_pairs_testset_v2.jsonl",
        out_csv="report/eval_step_structurer.csv"
    )
