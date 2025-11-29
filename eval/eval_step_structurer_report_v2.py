import json
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ---------- Embedding Utilities ----------
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")


def get_embedding(text: str) -> np.ndarray:
    if not text.strip():
        return np.zeros(768)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


# ---------- Matching ----------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return 1 - cosine(a, b)


def match_steps(gold_steps, pred_steps, threshold=0.8) -> List[Tuple[int, int]]:
    gold_embs = [get_embedding(s) for s in gold_steps]
    pred_embs = [get_embedding(s) for s in pred_steps]

    matched = []
    used_preds = set()
    for gi, g_emb in enumerate(gold_embs):
        best_score = threshold
        best_idx = -1
        for pi, p_emb in enumerate(pred_embs):
            if pi in used_preds:
                continue
            sim = cosine_sim(g_emb, p_emb)
            if sim > best_score:
                best_score = sim
                best_idx = pi
        if best_idx != -1:
            matched.append((gi, best_idx))
            used_preds.add(best_idx)
    return matched


# ---------- Coverage ----------
def get_param_coverage(params: List, gold_text: str) -> int:
    return sum(1 for p in params if isinstance(p, str) and p.lower() in gold_text.lower()
               or isinstance(p, dict) and any(
        v.lower() in gold_text.lower() for v in p.values() if isinstance(v, str))
               )


def get_material_coverage(mats: List, gold_text: str) -> int:
    return sum(1 for m in mats if isinstance(m, str) and m.lower() in gold_text.lower()
               or isinstance(m, dict) and any(
        v.lower() in gold_text.lower() for v in m.values() if isinstance(v, str))
               )


# ---------- Main Evaluation ----------
def evaluate_per_protocol(gold_map, pred_map, out_csv="report/eval_step_structurer.csv"):
    rows = []

    for pid in tqdm(gold_map.keys()):
        gold_steps_raw = gold_map[pid]
        pred_steps = pred_map.get(pid, [])

        gold_texts = [step["text"] for step in gold_steps_raw]
        pred_texts = [step["step_text"] for step in pred_steps]

        matched_pairs = match_steps(gold_texts, pred_texts)
        matched_gold = {gi for gi, _ in matched_pairs}
        matched_pred = {pi for _, pi in matched_pairs}

        precision = len(matched_pairs) / len(pred_texts) if pred_texts else 0
        recall = len(matched_pairs) / len(gold_texts) if gold_texts else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        # Order score (Spearman correlation)
        if len(matched_pairs) >= 2:
            gold_order = [gi for gi, _ in matched_pairs]
            pred_order = [pi for _, pi in matched_pairs]
            corr, _ = spearmanr(gold_order, pred_order)
            order_score = corr if not np.isnan(corr) else 0
        else:
            order_score = 0

        # Parameter / Material coverage
        param_total = material_total = param_cov = material_cov = 0
        for gi, pi in matched_pairs:
            pred_step = pred_steps[pi]
            gold_text = gold_texts[gi]
            param_list = pred_step.get("parameters", [])
            material_list = pred_step.get("materials", [])
            param_total += len(param_list)
            material_total += len(material_list)
            param_cov += get_param_coverage(param_list, gold_text)
            material_cov += get_material_coverage(material_list, gold_text)

        rows.append({
            "protocol_id": pid,
            "num_gold": len(gold_texts),
            "num_pred": len(pred_texts),
            "matched": len(matched_pairs),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "order_score": round(order_score, 3),
            "param_count": param_total,
            "param_covered": param_cov,
            "material_count": material_total,
            "material_covered": material_cov,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"✅ Evaluation report saved to {out_csv}")


# ---------- JSONL Loaders ----------
def load_jsonl(path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_gold(path: str) -> Dict[str, List[Dict]]:
    data = load_jsonl(path)
    return {x["protocol_id"]: x["steps"] for x in data}


def load_pred(path: str) -> Dict[str, List[Dict]]:
    data = load_jsonl(path)
    pred_map = {}
    for step in data:
        pid = step["protocol_id"]
        pred_map.setdefault(pid, []).append(step)
    return pred_map


if __name__ == "__main__":
    gold_path = "data/gold/gold_steps_testset.jsonl"
    pred_path = "runs/b2_steps_new.jsonl"
    out_path = "report/eval_step_structurer_v2_3.csv"

    gold = load_gold(gold_path)
    pred = load_pred(pred_path)
    evaluate_per_protocol(gold, pred, out_path)
