#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------- 공통 유틸 ----------

def normalize_title(text: str) -> str:
    """간단한 정규화: 소문자, 공백 정리, 끝 마침표 제거 등"""
    if text is None:
        return ""
    t = text.strip().lower()
    # 끝의 마침표/콜론 등 제거
    while len(t) > 0 and t[-1] in [".", ":", ";"]:
        t = t[:-1]
        t = t.strip()
    # 다중 공백 -> 한 칸
    t = " ".join(t.split())
    return t


def load_tasks(path: str) -> Dict[str, List[str]]:
    """
    jsonl 파일을 읽어서 protocol_id -> [task_title, ...] 딕셔너리로 변환
    """
    data = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pid = obj["protocol_id"]
            tasks = obj.get("tasks", [])
            titles = []
            for t in tasks:
                # t가 문자열이거나 dict인 경우 모두 처리
                if isinstance(t, str):
                    titles.append(t)
                elif isinstance(t, dict):
                    title = t.get("title") or t.get("name") or ""
                    titles.append(title)
            data[pid].extend(titles)
    return data


def precision_recall_f1(tp: int, pred_cnt: int, gold_cnt: int) -> Tuple[float, float, float]:
    p = tp / pred_cnt if pred_cnt > 0 else 0.0
    r = tp / gold_cnt if gold_cnt > 0 else 0.0
    if p + r == 0:
        f1 = 0.0
    else:
        f1 = 2 * p * r / (p + r)
    return p, r, f1


# ---------- 1) 문자열 기반 매칭 ----------

def eval_string_exact(gold: Dict[str, List[str]], pred: Dict[str, List[str]]) -> Dict[str, float]:
    """정규화된 문자열 완전 일치 기준으로 micro P/R/F1 계산"""
    tp = 0
    gold_cnt = 0
    pred_cnt = 0

    for pid, gold_tasks in gold.items():
        gold_norm = set(normalize_title(t) for t in gold_tasks if t.strip())
        gold_cnt += len(gold_norm)

        pred_tasks = pred.get(pid, [])
        pred_norm = set(normalize_title(t) for t in pred_tasks if t.strip())
        pred_cnt += len(pred_norm)

        tp += len(gold_norm.intersection(pred_norm))

    p, r, f1 = precision_recall_f1(tp, pred_cnt, gold_cnt)
    return {
        "string_exact_precision": p,
        "string_exact_recall": r,
        "string_exact_f1": f1,
        "string_exact_tp": tp,
        "string_exact_gold": gold_cnt,
        "string_exact_pred": pred_cnt,
    }


# ---------- 2) 키워드 기반 매칭 ----------

def build_keyword_repr(all_texts: List[str],
                       top_k: int = 5) -> Tuple[TfidfVectorizer, Dict[str, List[str]]]:
    """
    전체 태스크 문장에 대해 TF-IDF를 학습하고,
    문장별 상위 top_k 키워드 리스트를 생성하는 헬퍼.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=5000
    )
    tfidf = vectorizer.fit_transform(all_texts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    keyword_map = {}
    for idx, text in enumerate(all_texts):
        row = tfidf[idx].toarray().flatten()
        if row.sum() == 0:
            keyword_map[text] = []
            continue
        top_idx = row.argsort()[-top_k:][::-1]
        keywords = [feature_names[i] for i in top_idx if row[i] > 0]
        keyword_map[text] = list(dict.fromkeys(keywords))  # 중복 제거, 순서 유지

    return vectorizer, keyword_map


def jaccard(a: List[str], b: List[str]) -> float:
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def eval_keyword_overlap(
        gold: Dict[str, List[str]],
        pred: Dict[str, List[str]],
        top_k: int = 5,
        threshold: float = 0.3
) -> Dict[str, float]:
    """
    각 태스크 문장에서 상위 TF-IDF 키워드 집합을 만들고
    Jaccard 유사도 >= threshold 인 경우를 '의미 있는 매치'로 간주.
    """

    # 모든 문장 모으기 (정규화는 하지 않고 원문 사용 – 키워드 품질 위해)
    all_texts = []
    for tasks in gold.values():
        all_texts.extend(tasks)
    for tasks in pred.values():
        all_texts.extend(tasks)
    all_texts = [t for t in all_texts if t.strip()]

    if not all_texts:
        return {
            "keyword_precision": 0.0,
            "keyword_recall": 0.0,
            "keyword_f1": 0.0,
        }

    _, keyword_map = build_keyword_repr(all_texts, top_k=top_k)

    tp = 0
    gold_cnt = 0
    pred_cnt = 0

    for pid, gold_tasks in gold.items():
        g_titles = [t for t in gold_tasks if t.strip()]
        p_titles = [t for t in pred.get(pid, []) if t.strip()]

        gold_cnt += len(g_titles)
        pred_cnt += len(p_titles)

        # gold 기준으로 각각의 best pred 찾기 (many-to-one 허용)
        for g in g_titles:
            g_kw = keyword_map.get(g, [])
            best_sim = 0.0
            for p in p_titles:
                p_kw = keyword_map.get(p, [])
                sim = jaccard(g_kw, p_kw)
                if sim > best_sim:
                    best_sim = sim
            if best_sim >= threshold:
                tp += 1

    p, r, f1 = precision_recall_f1(tp, pred_cnt, gold_cnt)
    return {
        "keyword_precision": p,
        "keyword_recall": r,
        "keyword_f1": f1,
        "keyword_tp": tp,
        "keyword_gold": gold_cnt,
        "keyword_pred": pred_cnt,
        "keyword_top_k": top_k,
        "keyword_threshold": threshold,
    }


# ---------- 3) 임베딩 기반 Step Precision / Step Recall ----------

def compute_embeddings(model: SentenceTransformer,
                       texts: List[str],
                       batch_size: int = 32) -> np.ndarray:
    return np.array(model.encode(texts, batch_size=batch_size, show_progress_bar=True))


def eval_embedding_steps(
        gold: Dict[str, List[str]],
        pred: Dict[str, List[str]],
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        threshold: float = 0.7,
        device: str = None,
) -> Dict[str, float]:
    """
    각 태스크 문장을 SentenceTransformer 임베딩으로 바꾸고,
    cos 유사도 >= threshold 인 경우를 의미상 일치하는 Step 으로 간주.

    Step Precision (SP):
      - 예측 태스크 중 의미상 일치하는 골드 태스크가 있는 비율

    Step Recall (SR):
      - 골드 태스크 중 의미상 일치하는 예측 태스크가 있는 비율

    (BioProBench 의 Step metrics 아이디어를 따른 단순 버전)
    """

    # 1) 프로토콜별로 평가하면서 micro average
    total_pred = 0
    total_gold = 0
    matched_pred = 0
    matched_gold = 0

    # 모델 로드
    print(f"Loading sentence-transformer model: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    for pid, gold_tasks in gold.items():
        g_titles = [t for t in gold_tasks if t.strip()]
        p_titles = [t for t in pred.get(pid, []) if t.strip()]

        if not g_titles and not p_titles:
            continue

        total_gold += len(g_titles)
        total_pred += len(p_titles)

        if not g_titles or not p_titles:
            continue

        # 임베딩 계산
        texts = g_titles + p_titles
        emb = compute_embeddings(model, texts)
        g_emb = emb[: len(g_titles)]
        p_emb = emb[len(g_titles):]

        # cos similarity matrix: (num_pred, num_gold)
        sim_matrix = cosine_similarity(p_emb, g_emb)

        # 예측 기준: 각 예측태스크가 어떤 골드와도 δ 이상이면 match
        for i in range(len(p_titles)):
            if sim_matrix[i].max() >= threshold:
                matched_pred += 1

        # 골드 기준: 각 골드태스크가 어떤 예측과도 δ 이상이면 match
        for j in range(len(g_titles)):
            if sim_matrix[:, j].max() >= threshold:
                matched_gold += 1

    sp = matched_pred / total_pred if total_pred > 0 else 0.0
    sr = matched_gold / total_gold if total_gold > 0 else 0.0
    if sp + sr == 0:
        sf1 = 0.0
    else:
        sf1 = 2 * sp * sr / (sp + sr)

    return {
        "step_precision": sp,
        "step_recall": sr,
        "step_f1": sf1,
        "step_threshold": threshold,
        "step_total_pred": total_pred,
        "step_total_gold": total_gold,
        "step_matched_pred": matched_pred,
        "step_matched_gold": matched_gold,
        "embed_model": model_name,
    }


# ---------- 메인 ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, help="gold_tasks_testset.jsonl")
    parser.add_argument("--pred", required=True, help="예측 태스크 jsonl")
    parser.add_argument("--output", required=False, help="결과를 저장할 json 파일 경로")
    parser.add_argument("--keyword_top_k", type=int, default=5)
    parser.add_argument("--keyword_threshold", type=float, default=0.3)
    parser.add_argument("--embed_model", type=str,
                        default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--embed_threshold", type=float, default=0.7)
    parser.add_argument("--embed_device", type=str, default=None,
                        help="cpu, cuda, mps 등 (None이면 자동)")
    args = parser.parse_args()

    gold = load_tasks(args.gold)
    pred = load_tasks(args.pred)

    results = {}

    # 1) 문자열 기반
    print("Evaluating string-exact metrics...")
    results.update(eval_string_exact(gold, pred))

    # 2) 키워드 기반
    print("Evaluating keyword-based metrics...")
    results.update(
        eval_keyword_overlap(
            gold,
            pred,
            top_k=args.keyword_top_k,
            threshold=args.keyword_threshold,
        )
    )

    # 3) 임베딩 기반 Step metrics
    print("Evaluating embedding-based step metrics...")
    results.update(
        eval_embedding_steps(
            gold,
            pred,
            model_name=args.embed_model,
            threshold=args.embed_threshold,
            device=args.embed_device,
        )
    )

    # 출력
    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
