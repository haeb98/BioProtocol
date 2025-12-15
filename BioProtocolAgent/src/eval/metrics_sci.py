# src/eval/metrics_sci.py

from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModel

# 1) SciBERT 로드 (전역에서 한 번만)
_SCI_MODEL_NAME = "allenai/scibert_scivocab_uncased"
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_tokenizer = AutoTokenizer.from_pretrained(_SCI_MODEL_NAME)
_model = AutoModel.from_pretrained(_SCI_MODEL_NAME).to(_device)
_model.eval()


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """[batch, seq, hidden] → [batch, hidden] 평균 풀링"""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def scibert_embed(texts: List[str], max_length: int = 128) -> torch.Tensor:
    """문장 리스트를 SciBERT 임베딩으로 변환 (L2 정규화된 벡터)"""
    if len(texts) == 0:
        return torch.empty(0, _model.config.hidden_size)

    enc = _tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(_device)

    with torch.no_grad():
        out = _model(**enc)

    sent_emb = _mean_pool(out.last_hidden_state, enc["attention_mask"])
    sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)  # cosine 위해 정규화
    return sent_emb.cpu()  # 이후 CPU에서 계산


def step_match_scores_scibert(
        gold_steps: List[Dict],
        pred_steps: List[Dict],
        sim_threshold: float = 0.7,
) -> Tuple[float, float, float, List[int]]:
    """
    SciBERT 임베딩 기반 Step 매칭
    - gold_steps / pred_steps: 각 원소에 'step_text' 필드 포함
    - sim_threshold 이상이면 매칭 성공으로 간주
    return: (precision, recall, f1, gold_to_pred_idx)
    """
    if len(gold_steps) == 0 and len(pred_steps) == 0:
        return 1.0, 1.0, 1.0, []

    if len(gold_steps) == 0:
        return 0.0, 1.0, 0.0, []

    if len(pred_steps) == 0:
        return 1.0, 0.0, 0.0, [-1] * len(gold_steps)

    gold_texts = [g["step_text"] for g in gold_steps]
    pred_texts = [p["step_text"] for p in pred_steps]

    # 1) SciBERT 임베딩
    gold_emb = scibert_embed(gold_texts)  # [G, H]
    pred_emb = scibert_embed(pred_texts)  # [P, H]

    # 2) cosine similarity 행렬 [G, P]
    sim_matrix = torch.matmul(gold_emb, pred_emb.T)  # cosine (이미 L2 정규화됨)

    gold_to_pred_idx: List[int] = []
    matched_pred_indices = set()

    for g_idx in range(len(gold_steps)):
        sims = sim_matrix[g_idx]  # [P]
        best_j = int(torch.argmax(sims).item())
        best_score = float(sims[best_j].item())

        if best_score >= sim_threshold:
            gold_to_pred_idx.append(best_j)
            matched_pred_indices.add(best_j)
        else:
            gold_to_pred_idx.append(-1)

    num_matched_gold = sum(1 for x in gold_to_pred_idx if x != -1)
    num_matched_pred = len(matched_pred_indices)

    precision = num_matched_pred / len(pred_steps) if pred_steps else 0.0
    recall = num_matched_gold / len(gold_steps) if gold_steps else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, gold_to_pred_idx


def order_score_from_mapping(gold_to_pred_idx: List[int]) -> float:
    """기존 token 버전과 동일: 매칭된 step들 사이 순서 일관성 비율"""
    indices = [idx for idx in gold_to_pred_idx if idx != -1]
    n = len(indices)
    if n <= 1:
        return 1.0

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
