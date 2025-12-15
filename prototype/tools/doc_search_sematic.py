import re

from sentence_transformers import SentenceTransformer, util

# SciBERT나 BioBERT보다 light한 모델로 시작 (추후 대체 가능)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def doc_search(query, methods_text, top_k=3):
    # 문장 분리
    sentences = re.split(r'(?<=[.?!])\s+', methods_text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    # query & sentence 임베딩
    query_embedding = model.encode(query, convert_to_tensor=True)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    # 유사도 계산
    cosine_scores = util.cos_sim(query_embedding, sentence_embeddings)[0]
    top_results = sorted(enumerate(cosine_scores), key=lambda x: x[1], reverse=True)[:top_k]

    results = [sentences[i] for i, score in top_results if score > 0.4]  # threshold 조정 가능
    return "\n".join(results)
