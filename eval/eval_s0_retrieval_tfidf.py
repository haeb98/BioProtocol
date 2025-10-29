# eval/eval_s0_retrieval.py
"""
S0 Retrieval 평가:
- gold_pairs.jsonl의 각 샘플에 대해 여러 질의를 만들어 코퍼스에서 검색
- 정답 protocol_id가 top-k 안에 들어오는지 확인하여 Recall@k, MRR@k, nDCG@k 집계
- 질의 생성:
  Q1: gold.protocol.title
  Q2: article.text에서 TF-IDF 상위 n-gram을 질의로 (경량 자동 질의)

입력:
  data/gold/gold_pairs.jsonl
  data/rag/corpus_fts.sqlite, data/rag/faiss_by_id.index, data/rag/doc_ids.npy
출력:
  eval/s0_report.json, eval/s0_samples.tsv (실수 사례)
"""

import math
import pathlib

import numpy as np
import orjson
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from rag.retriever_light import HybridRetrieverLight

ROOT = pathlib.Path(".")
GOLD = ROOT / "data/gold/gold_pairs.jsonl"
OUT_DIR = ROOT / "eval";
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "s0_report.json"
OUT_SAMPLES = OUT_DIR / "s0_samples.tsv"

# 검색 파라미터
K_LEX = 50
K_SEM = 50
TOPN = 50

# TF-IDF 기반 질의 파라미터
NGRAM_RANGE = (1, 2)
MAX_FEATURES = 5000
TOP_KEYWORDS = 10  # TF-IDF 상위 키워드 개수
MIN_DOC_LEN = 1000  # 논문 텍스트 최소 길이(작으면 Q2 생략)


def load_gold(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            pairs.append(orjson.loads(line))
    return pairs


def build_tfidf_queries(articles, top_n=TOP_KEYWORDS):
    """
    articles: list[str]  (각 샘플의 article.text)
    반환: list[list[str]]  (샘플별 키워드 리스트 문자열)
    """
    # 매우 긴 텍스트도 있으니 경량 설정
    vectorizer = TfidfVectorizer(
        ngram_range=NGRAM_RANGE,
        max_features=MAX_FEATURES,
        stop_words="english",
        lowercase=True
    )
    X = vectorizer.fit_transform(articles)
    feats = np.array(vectorizer.get_feature_names_out())
    queries = []
    for i in range(X.shape[0]):
        row = X.getrow(i).toarray().ravel()
        idx = row.argsort()[::-1][:top_n]
        toks = feats[idx]
        q = " ".join(toks)
        queries.append(q)
    return queries


def metrics_from_ranks(ranks, k_values=(1, 3, 5, 8, 10)):
    """
    ranks: list[int or None]  (각 샘플의 '정답 순위'(0-based). 못 찾으면 None)
    """
    res = {}
    n = len(ranks)
    for k in k_values:
        hit = sum(1 for r in ranks if r is not None and r < k)
        res[f"Recall@{k}"] = hit / n if n else 0.0

    # MRR@TOPN
    mrr = 0.0
    for r in ranks:
        if r is not None and r < TOPN:
            mrr += 1.0 / (r + 1)
    res[f"MRR@{TOPN}"] = mrr / n if n else 0.0

    # nDCG@TOPN (binary relevance: 정답 1개)
    ndcg = 0.0
    for r in ranks:
        if r is None or r >= TOPN:
            continue
        # DCG = 1/log2(rank+2) (rank는 0-based)
        dcg = 1.0 / math.log2(r + 2)
        idcg = 1.0  # 최선은 rank=0 → 1/log2(1+1) = 1
        ndcg += dcg / idcg
    res[f"nDCG@{TOPN}"] = ndcg / n if n else 0.0
    return res


def main():
    pairs = load_gold(GOLD)
    if not pairs:
        raise SystemExit(f"no gold pairs found at {GOLD}")
    R = HybridRetrieverLight()  # FTS+FAISS 하이브리드

    # Q1: 프로토콜 제목 기반 질의
    q1_list = [(p["protocol"]["title"] or "").strip() for p in pairs]

    # Q2: 논문 텍스트 기반 TF-IDF 질의 (너무 짧으면 빈 문자열)
    articles = [(p["article"]["text"] or "") for p in pairs]
    articles_for_tfidf = [(t if len(t) >= MIN_DOC_LEN else "") for t in articles]
    # TF-IDF는 비어있는 문서를 넣으면 오류 → 최소 길이 미만은 일괄 대체
    # 완전히 비어있으면 fit_transform이 안되므로, 최소 하나는 채워두기
    any_nonempty = any(bool(t.strip()) for t in articles_for_tfidf)
    if not any_nonempty:
        q2_list = ["" for _ in pairs]
    else:
        # 비어있는 것은 공백 하나로 대체(벡터라이저가 무시)
        _art = [(t if t.strip() else " ") for t in articles_for_tfidf]
        q2_list = build_tfidf_queries(_art, top_n=TOP_KEYWORDS)

    sample_logs = []
    best_ranks = []  # 샘플별 best rank (여러 질의 중 가장 좋은 순위)
    for i, p in enumerate(tqdm(pairs, desc="eval")):
        pid = p["protocol_id"]  # 정답 doc_id
        # 질의 후보
        q_candidates = []
        if q1_list[i]:
            q_candidates.append(("Q1_title", q1_list[i]))
        if q2_list[i] and q2_list[i].strip() not in ("", " "):
            q_candidates.append(("Q2_tfidf", q2_list[i]))

        # 최소한 하나는 만들어지도록 보정
        if not q_candidates:
            # title이나 tfidf가 모두 비면 keywords라도 시도
            kw = p["protocol"].get("keywords") or ""
            if kw:
                q_candidates.append(("Q0_keywords", kw))

        rank_best = None
        logs_this = []
        for tag, q in q_candidates:
            res = R.search(q, k_lex=K_LEX, k_sem=K_SEM, topn=TOPN)
            # 순위 찾기
            rank = None
            for r_idx, r in enumerate(res):
                if r["doc_id"] == pid:
                    rank = r_idx
                    break
            rank_best = rank if (rank_best is None or (rank is not None and rank < rank_best)) else rank_best

            # 간단 로그(상위 3개 preview)
            top3 = "; ".join([f"{x['doc_id']}|{x['title'][:60]}" for x in res[:3]])
            logs_this.append([tag, q[:120].replace("\n", " "), rank if rank is not None else -1, top3])

        best_ranks.append(rank_best)
        # 샘플 로그 한 줄 저장
        sample_logs.append([
            pid,
            p["protocol"]["title"][:80].replace("\n", " "),
            rank_best if rank_best is not None else -1,
            "|".join([f"{t}:{r}" for (t, _, r, _) in logs_this]),
            " || ".join([top for (*_, top) in logs_this])
        ])

    # 지표 집계
    report = metrics_from_ranks(best_ranks, k_values=(1, 3, 5, 8, 10))
    report["num_samples"] = len(pairs)

    OUT_JSON.write_text(orjson.dumps(report, option=orjson.OPT_INDENT_2).decode(), encoding="utf-8")
    with open(OUT_SAMPLES, "w", encoding="utf-8") as f:
        f.write("protocol_id\ttitle\tbest_rank\tqueries(rank)\ttop3_preview\n")
        for row in sample_logs:
            f.write("\t".join([str(x) for x in row]) + "\n")

    print("[OK] report:", OUT_JSON)
    print("[OK] samples:", OUT_SAMPLES)
    print(report)


if __name__ == "__main__":
    main()
