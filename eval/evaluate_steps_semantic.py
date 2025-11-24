#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate structured steps against gold hierarchical protocol steps.

- Input (gold): data/gold/gold_steps_testset.jsonl
    {"protocol_id": "...",
     "steps": [
        {"step_id": "1.1", "task_id": "T1", "text": "Seed cells at ..."},
        ...
     ]}

- Input (pred): runs/steps_from_tasks_baseline.jsonl
    {"protocol_id": "...",
     "steps": [
        {
          "id": "S1",
          "task_id": "T1",
          "title": "...",
          "instruction": "Seed cells at ...",   # main text
          ...
        },
        ...
     ]}

We evaluate:
  1) string-exact
  2) keyword-based semantic match (Jaccard over extracted keywords)
  3) sentence-embedding-based step semantic similarity
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple


############################################################
# Utility: normalization & tokenization
############################################################

def normalize_text(s: str) -> str:
    """Lowercase + collapse whitespace for strict string match."""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


STOPWORDS = {
    # 아주 간단한 stopword 세트 (필요하면 확장 가능)
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "to",
    "with", "at", "by", "from", "as", "is", "are", "was", "were",
    "be", "being", "been", "this", "that", "these", "those",
    "using", "use", "used", "into", "within", "per", "via"
}


def tokenize_keywords(text: str, top_k: int = 5) -> List[str]:
    """
    매우 단순한 키워드 추출:
    - 알파벳/숫자 단어 토큰화
    - stopword 제거
    - 숫자만 있는 토큰 제거
    - 앞에서부터 고유 토큰 top_k 개만 사용
    """
    text = text.lower()
    tokens = re.findall(r"[a-z0-9\-\+]+", text)
    keywords = []
    seen = set()
    for t in tokens:
        if t in STOPWORDS:
            continue
        # 숫자만으로 된 토큰은 버림 (예: 1, 24, 3x 등은 필요에 따라 조정 가능)
        if re.fullmatch(r"\d+(\.\d+)?", t):
            continue
        if t not in seen:
            seen.add(t)
            keywords.append(t)
        if len(keywords) >= top_k:
            break
    return keywords


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


############################################################
# Loading gold & pred
############################################################

def load_gold_steps(path: str) -> Dict[str, List[str]]:
    """
    gold_steps_testset.jsonl 형식 로딩:
    {protocol_id: [step_text_1, step_text_2, ...]}
    """
    pid_to_steps = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pid = obj["protocol_id"]
            for s in obj.get("steps", []):
                txt = s.get("text", "").strip()
                if txt:
                    pid_to_steps[pid].append(txt)
    return pid_to_steps


def load_pred_steps(path: str) -> Dict[str, List[str]]:
    """
    steps_from_tasks_baseline.jsonl / steps_from_tasks_rag.jsonl 형식 로딩:
    - step["instruction"] 우선 사용
    - 없으면 step["text"]
    """
    pid_to_steps = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pid = obj.get("protocol_id")
            if not pid:
                continue
            steps = obj.get("steps", [])
            for s in steps:
                txt = s.get("instruction") or s.get("text") or ""
                txt = txt.strip()
                if txt:
                    pid_to_steps[pid].append(txt)
    return pid_to_steps


############################################################
# String-exact matching (1:1)
############################################################

def evaluate_string_exact(
        gold: Dict[str, List[str]],
        pred: Dict[str, List[str]],
) -> Tuple[int, int, int]:
    """
    1:1 string exact match (normalized) per protocol_id.
    Returns: (tp, total_gold, total_pred)
    """
    tp = 0
    total_gold = 0
    total_pred = 0

    for pid, gold_steps in gold.items():
        gold_norms = [normalize_text(s) for s in gold_steps]
        total_gold += len(gold_norms)
        pred_steps = pred.get(pid, [])
        pred_norms = [normalize_text(s) for s in pred_steps]
        total_pred += len(pred_norms)

        used_gold = [False] * len(gold_norms)
        for pn in pred_norms:
            for i, gn in enumerate(gold_norms):
                if not used_gold[i] and pn == gn:
                    tp += 1
                    used_gold[i] = True
                    break

    return tp, total_gold, total_pred


############################################################
# Keyword-based semantic matching (1:1, Jaccard)
############################################################

def evaluate_keyword_semantic(
        gold: Dict[str, List[str]],
        pred: Dict[str, List[str]],
        top_k: int = 5,
        threshold: float = 0.3,
) -> Tuple[int, int, int]:
    """
    사전 정의된 키워드 추출 방식 + Jaccard 유사도 기반 매칭.
    Returns: (tp, total_gold, total_pred)
    """
    tp = 0
    total_gold = 0
    total_pred = 0

    for pid, gold_steps in gold.items():
        gold_kw = [tokenize_keywords(s, top_k=top_k) for s in gold_steps]
        pred_steps = pred.get(pid, [])
        pred_kw = [tokenize_keywords(s, top_k=top_k) for s in pred_steps]

        total_gold += len(gold_kw)
        total_pred += len(pred_kw)

        used_gold = [False] * len(gold_kw)

        for pk in pred_kw:
            best_i = -1
            best_score = 0.0
            for i, gk in enumerate(gold_kw):
                if used_gold[i]:
                    continue
                score = jaccard(pk, gk)
                if score > best_score:
                    best_score = score
                    best_i = i
            if best_i >= 0 and best_score >= threshold:
                used_gold[best_i] = True
                tp += 1

    return tp, total_gold, total_pred


############################################################
# Embedding-based semantic matching
############################################################

def evaluate_embedding_steps(
        gold: Dict[str, List[str]],
        pred: Dict[str, List[str]],
        model_name: str,
        device: str,
        threshold: float = 0.7,
) -> Tuple[int, int, int]:
    """
    Sentence-Transformer 임베딩 기반 step semantic match.
    - gold/pred 임베딩 후 cosine similarity 행렬에서 greedy 1:1 매칭.
    """
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        raise ImportError(
            "sentence-transformers 가 설치되어 있어야 합니다. "
            "예: pip install sentence-transformers"
        )

    model = SentenceTransformer(model_name, device=device)

    tp = 0
    total_gold = 0
    total_pred = 0

    for pid, gold_steps in gold.items():
        pred_steps = pred.get(pid, [])
        if not gold_steps or not pred_steps:
            total_gold += len(gold_steps)
            total_pred += len(pred_steps)
            continue

        total_gold += len(gold_steps)
        total_pred += len(pred_steps)

        gold_emb = model.encode(gold_steps, convert_to_tensor=True, show_progress_bar=False)
        pred_emb = model.encode(pred_steps, convert_to_tensor=True, show_progress_bar=False)

        sim = util.cos_sim(pred_emb, gold_emb)  # [num_pred, num_gold]

        # greedy 매칭
        used_gold = [False] * len(gold_steps)
        for p_idx in range(len(pred_steps)):
            # 각 pred p_idx 에 대해 가장 유사한 gold 후보를 찾는다.
            best_i = -1
            best_score = -1.0
            for g_idx in range(len(gold_steps)):
                if used_gold[g_idx]:
                    continue
                score = float(sim[p_idx, g_idx])
                if score > best_score:
                    best_score = score
                    best_i = g_idx
            if best_i >= 0 and best_score >= threshold:
                used_gold[best_i] = True
                tp += 1

    return tp, total_gold, total_pred


############################################################
# Metric helper
############################################################

def compute_prf(tp: int, total_gold: int, total_pred: int) -> Tuple[float, float, float]:
    precision = tp / total_pred if total_pred > 0 else 0.0
    recall = tp / total_gold if total_gold > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


############################################################
# Main
############################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=str, required=True,
                        help="gold_steps_testset.jsonl 경로")
    parser.add_argument("--pred", type=str, required=True,
                        help="예측 step jsonl 경로 (steps_from_tasks_*.jsonl)")
    parser.add_argument("--output", type=str, required=True,
                        help="결과를 저장할 json 파일 경로")

    parser.add_argument("--keyword_top_k", type=int, default=5)
    parser.add_argument("--keyword_threshold", type=float, default=0.3)

    parser.add_argument("--embed_model", type=str,
                        default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--embed_threshold", type=float, default=0.7)
    parser.add_argument("--embed_device", type=str, default="cpu",
                        help="cpu / cuda / mps 등")

    args = parser.parse_args()

    print(f"Loading gold from {args.gold}")
    gold = load_gold_steps(args.gold)
    print(f"Loading pred from {args.pred}")
    pred = load_pred_steps(args.pred)

    print("Gold protocols:", len(gold))
    print("Pred protocols:", len(pred))
    print("Total gold steps:", sum(len(v) for v in gold.values()))
    print("Total pred steps:", sum(len(v) for v in pred.values()))

    # 1) String-exact
    print("Evaluating string-exact metrics...")
    se_tp, se_gold, se_pred = evaluate_string_exact(gold, pred)
    se_p, se_r, se_f1 = compute_prf(se_tp, se_gold, se_pred)

    # 2) Keyword-based
    print("Evaluating keyword-based metrics...")
    kw_tp, kw_gold, kw_pred = evaluate_keyword_semantic(
        gold, pred,
        top_k=args.keyword_top_k,
        threshold=args.keyword_threshold,
    )
    kw_p, kw_r, kw_f1 = compute_prf(kw_tp, kw_gold, kw_pred)

    # 3) Embedding-based
    print("Evaluating embedding-based step metrics...")
    em_tp, em_gold, em_pred = evaluate_embedding_steps(
        gold, pred,
        model_name=args.embed_model,
        device=args.embed_device,
        threshold=args.embed_threshold,
    )
    em_p, em_r, em_f1 = compute_prf(em_tp, em_gold, em_pred)

    results = {
        "string_exact_precision": se_p,
        "string_exact_recall": se_r,
        "string_exact_f1": se_f1,
        "string_exact_tp": se_tp,
        "string_exact_gold": se_gold,
        "string_exact_pred": se_pred,

        "keyword_precision": kw_p,
        "keyword_recall": kw_r,
        "keyword_f1": kw_f1,
        "keyword_tp": kw_tp,
        "keyword_gold": kw_gold,
        "keyword_pred": kw_pred,
        "keyword_top_k": args.keyword_top_k,
        "keyword_threshold": args.keyword_threshold,

        "step_precision": em_p,
        "step_recall": em_r,
        "step_f1": em_f1,
        "step_threshold": args.embed_threshold,
        "step_total_pred": em_pred,
        "step_total_gold": em_gold,
        "step_matched_pred": em_tp,  # == tp
        "step_matched_gold": em_tp,
        "embed_model": args.embed_model,
    }

    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
