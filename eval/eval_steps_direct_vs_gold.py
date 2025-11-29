import argparse
import json

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_direct_steps(raw_steps):
    steps = []

    if isinstance(raw_steps, list):
        for s in raw_steps:
            if isinstance(s, str):
                steps.append(s.strip())

            elif isinstance(s, dict):
                # Case 1: has description key
                if "description" in s:
                    steps.append(s["description"].strip())

                # Case 2: has action key
                elif "action" in s:
                    steps.append(s["action"].strip())

                # ✅ Case 3: dict with single numeric key → extract value
                elif len(s) == 1:
                    value = list(s.values())[0]
                    if isinstance(value, str):
                        steps.append(value.strip())

    elif isinstance(raw_steps, dict) and "steps" in raw_steps:
        return normalize_direct_steps(raw_steps["steps"])

    return steps


class SciBERTEmbedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        self.model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

    def encode(self, texts):
        if not texts:
            return torch.zeros((0, 768))
        inputs = self.tokenizer(
            texts, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]  # [CLS] token
        return F.normalize(embeddings, p=2, dim=1)


def greedy_match(pred_texts, gold_texts, embedder, threshold=0.8):
    if not pred_texts or not gold_texts:
        return 0, 0, 0

    pred_embs = embedder.encode(pred_texts)
    gold_embs = embedder.encode(gold_texts)

    sim_matrix = torch.matmul(gold_embs, pred_embs.T)  # cosine similarity
    matched = 0
    used_preds = set()

    for i in range(len(gold_texts)):
        sim_scores = sim_matrix[i]
        sorted_idx = torch.argsort(sim_scores, descending=True)
        for j in sorted_idx:
            if j.item() in used_preds:
                continue
            if sim_scores[j] >= threshold:
                matched += 1
                used_preds.add(j.item())
                break

    return matched, len(pred_texts), len(gold_texts)


def evaluate(pred_path, gold_path, output_path):
    pred_data = load_jsonl(pred_path)
    gold_data = load_jsonl(gold_path)

    pred_map = {
        ex["protocol_id"]: normalize_direct_steps(ex.get("steps", [])) for ex in pred_data
    }
    gold_map = {
        ex["protocol_id"]: [s["text"] for s in ex.get("steps", [])] for ex in gold_data
    }

    embedder = SciBERTEmbedder()
    results = []

    for pid, gold_steps in tqdm(gold_map.items(), desc="Evaluating protocols"):
        pred_steps = pred_map.get(pid, [])
        matched, num_pred, num_gold = greedy_match(pred_steps, gold_steps, embedder)
        precision = matched / num_pred if num_pred else 0
        recall = matched / num_gold if num_gold else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        results.append({
            "protocol_id": pid,
            "num_pred": num_pred,
            "num_gold": num_gold,
            "matched": matched,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        })

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved to {output_path}")
    print(df.describe().round(3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Path to steps_direct_new.jsonl")
    parser.add_argument("--gold", required=True, help="Path to gold_steps_testset.jsonl")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    args = parser.parse_args()

    evaluate(args.pred, args.gold, args.output)
