import json
import os

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModel


def load_gold_protocol(path="prototype/data/bio_protocol.json", protocol_id="Bio-protocol-2096"):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for entry in data:
        if entry.get("id") == protocol_id:
            return entry.get("input", ""), entry.get("protocol", "")
    raise ValueError("Gold protocol not found")


import re


def extract_numbered_steps(text):
    # Look for all digit-dot-prefixed segments (e.g. 1. Step text)
    # Positive lookahead to find next digit-dot or end of string
    pattern = r'(\d{1,2}\.[^#]*?)(?=\d{1,2}\.|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    return [s.strip().replace("\n", " ") for s in matches if len(s.strip()) > 0]


def extract_section_and_split(text):
    """
    Split the given text into material and protocol sections based on known section headers.
    Return: material_lines (List[str]), protocol_lines (List[str])
    """
    section_patterns = [
        "## Procedure:", "# Procedure:", "# Methods:",
        "Methods and Procedures:", "Methods:", "Protocol:"
    ]

    for pattern in section_patterns:
        if pattern in text:
            before, after = text.split(pattern, 1)
            material_lines = [line.strip() for line in before.strip().split('\n') if line.strip()]
            protocol_lines = [line.strip() for line in after.strip().split('\n') if line.strip()]
            return material_lines, protocol_lines

    # If no pattern matched, return everything as protocol by default
    all_lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    return [], all_lines


def extract_params(text):
    # 숫자 + 단위 패턴 (예: 1 mL, 5 min, 130xg, 37°C, 0.25%, 10μL)
    return re.findall(r'\b\d+(?:\.\d+)?\s?(?:[a-zA-Z°μ%/]+)', text)


def embed_sentences(sentences, tokenizer, model):
    encoded = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded)
    return output.last_hidden_state.mean(dim=1).cpu().numpy()


def cosine_similarity_matrix(a, b):
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.matmul(a_norm, b_norm.T)


def match_elements(gen_items, gold_items, embedder, threshold=0.8, return_match_count=False):
    if not gen_items or not gold_items:
        return (0, 0, 0, 0) if return_match_count else (0, 0, 0)

    gen_emb = embed_sentences(gen_items, *embedder)
    gold_emb = embed_sentences(gold_items, *embedder)

    matched = set()
    for i, g_vec in enumerate(gen_emb):
        sims = np.dot(gold_emb, g_vec) / (np.linalg.norm(gold_emb, axis=1) * np.linalg.norm(g_vec))
        best = np.argmax(sims)
        if sims[best] >= threshold and best not in matched:
            matched.add(best)

    tp = len(matched)
    fp = len(gen_items) - tp
    fn = len(gold_items) - tp
    precision, recall, f1 = precision_recall_fscore_support(
        [1] * tp + [0] * fp, [1] * tp + [1] * fp, average='binary')[:3]

    if return_match_count:
        return precision, recall, f1, tp
    return precision, recall, f1


def match_params(gold_steps, gen_steps):
    gold_params = sum([extract_params(s) for s in gold_steps], [])
    gen_params = sum([extract_params(s) for s in gen_steps], [])

    gold_set = set(gold_params)
    gen_set = set(gen_params)
    tp = len(gold_set & gen_set)
    fp = len(gen_set - gold_set)
    fn = len(gold_set - gen_set)

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1


def evaluate_protocols(gold_input, gold_protocol, generated_dir="prototype/output/"):
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
    embedder = (tokenizer, model)

    gold_steps = extract_numbered_steps(gold_protocol)
    gold_materials = [line.strip() for line in gold_input.strip().split('\n') if line.strip()]

    for filename in os.listdir(generated_dir):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(generated_dir, filename), encoding="utf-8") as f:
            content = f.read()

        gen_materials, gen_steps = extract_section_and_split(content)

        step_p, step_r, step_f1, matched_count = match_elements(gen_steps, gold_steps, embedder,
                                                                return_match_count=True)
        mat_p, mat_r, mat_f1 = match_elements(gen_materials, gold_materials, embedder)
        param_p, param_r, param_f1 = match_params(gold_steps, gen_steps)

        # order_score = 0.0
        # if step_matches:
        #     gen_order = [i for i, _ in step_matches]
        #     gold_order = [j for _, j in step_matches]
        #     order_score = spearmanr(gen_order, gold_order).correlation or 0.0

        print(f"=== {filename} ===")
        print(f"# gold_steps: {len(gold_steps)}")
        print(f"# gen_steps: {len(gen_steps)}")
        print(f"# matched_steps: {matched_count}")
        print(f"step_f1: {step_f1:.2f} | step_precision: {step_p:.2f} | step_recall: {step_r:.2f}")
        print(f"material_f1: {mat_f1:.2f} | material_precision: {mat_p:.2f} | material_recall: {mat_r:.2f}")
        print(f"param_f1: {param_f1:.2f} | param_precision: {param_p:.2f} | param_recall: {param_r:.2f}")
        # print(f"order_score (spearman corr.): {order_score:.2f}\n")


if __name__ == "__main__":
    gold_input, gold_protocol = load_gold_protocol()
    evaluate_protocols(gold_input, gold_protocol)
