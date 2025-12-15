# src/eval/metrics.py
import re
from typing import List, Tuple, Dict, Any


def normalize_text(text: str) -> List[str]:
    if text is None:
        return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [t for t in text.split() if t]
    return tokens


def token_f1(gold: str, pred: str) -> float:
    g_tok = normalize_text(gold)
    p_tok = normalize_text(pred)
    if not g_tok or not p_tok:
        return 0.0
    g_set = set(g_tok)
    p_set = set(p_tok)
    inter = len(g_set & p_set)
    if inter == 0:
        return 0.0
    prec = inter / len(p_set)
    rec = inter / len(g_set)
    return 2 * prec * rec / (prec + rec)


def step_match_scores(
        gold_steps: List[Dict[str, Any]],
        pred_steps: List[Dict[str, Any]],
        sim_threshold: float = 0.2,
) -> Tuple[float, float, float, List[int]]:
    """
    각 gold step에 대해 가장 유사한 pred step을 찾아 Step-level Precision/Recall/F1 계산.
    또한 gold→pred 인덱스 매핑 리스트를 반환 (순서 평가에 사용).
    """
    gold_texts = [s["step_text"] for s in gold_steps]
    pred_texts = [s["step_text"] for s in pred_steps]

    if not gold_texts or not pred_texts:
        return 0.0, 0.0, 0.0, []

    gold_to_pred_idx: List[int] = []
    matched_pred_indices = set()

    for g_idx, g in enumerate(gold_texts):
        best_score = 0.0
        best_j = -1
        for j, p in enumerate(pred_texts):
            score = token_f1(g, p)
            if score > best_score:
                best_score = score
                best_j = j
        if best_score >= sim_threshold:
            gold_to_pred_idx.append(best_j)
            matched_pred_indices.add(best_j)
        else:
            gold_to_pred_idx.append(-1)

    num_matched_gold = sum(1 for x in gold_to_pred_idx if x != -1)
    num_matched_pred = len(matched_pred_indices)

    prec = num_matched_pred / len(pred_steps)
    rec = num_matched_gold / len(gold_steps)
    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)

    return prec, rec, f1, gold_to_pred_idx


def order_score_from_mapping(gold_to_pred_idx: List[int]) -> float:
    """
    gold_to_pred_idx: gold step i가 매칭된 pred step 인덱스 (또는 -1).
    매칭된 것들 사이에서 순서 일관성 비율 계산.
    """
    # 매칭된 gold만 모아서 pred index 순서 시퀀스로 만듦
    indices = [idx for idx in gold_to_pred_idx if idx != -1]
    n = len(indices)
    if n <= 1:
        return 1.0  # trivially perfect

    total_pairs = 0
    correct_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_pairs += 1
            if indices[i] < indices[j]:
                correct_pairs += 1

    if total_pairs == 0:
        return 1.0
    return correct_pairs / total_pairs
