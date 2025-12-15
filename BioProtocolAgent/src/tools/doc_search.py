# src/tools/doc_search.py
from typing import List, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.jats_reader import load_jats_xml, extract_article_spans

# 한 번 벡터라이저/문서벡터를 캐싱하기 위한 전역 변수
_cache = {}


def _build_index_for_pmcid(pmcid: str):
    """
    특정 pmcid에 대해 TF-IDF 인덱스를 만들고 캐싱.
    """
    if pmcid in _cache:
        return _cache[pmcid]

    tree = load_jats_xml(pmcid)
    spans = extract_article_spans(tree, max_len=2000)  # (label, text) 목록
    if not spans:
        raise ValueError(f"No spans extracted for pmcid={pmcid}")

    labels = [label for (label, _) in spans]
    texts = [text for (_, text) in spans]

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    doc_vecs = vectorizer.fit_transform(texts)

    _cache[pmcid] = {
        "labels": labels,
        "texts": texts,
        "vectorizer": vectorizer,
        "doc_vecs": doc_vecs,
    }
    return _cache[pmcid]


def doc_search_fulltext(pmcid: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    주어진 pmcid 논문 전체(JATS)에서 query와 유사한 텍스트 span을 상위 top_k개 반환.

    반환 형식:
    [
      {
        "section": "sec" / "abstract" / ...,
        "score": 0.73,
        "text": "...."
      },
      ...
    ]
    """
    idx = _build_index_for_pmcid(pmcid)
    vectorizer = idx["vectorizer"]
    doc_vecs = idx["doc_vecs"]
    labels = idx["labels"]
    texts = idx["texts"]

    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, doc_vecs)[0]  # shape: (n_docs,)

    # 높은 순으로 정렬해서 top_k 선택
    sorted_idx = sims.argsort()[::-1][:top_k]

    results = []
    for i in sorted_idx:
        results.append({
            "section": labels[i],
            "score": float(sims[i]),
            "text": texts[i],
        })
    return results
