"""
입력:
  - data/processed/protocol_corpus_docs.jsonl  # {"doc_id": str|int, "text": "..."}
출력:
  - data/rag/faiss_by_id.index                 # FAISS 벡터 인덱스(벡터만 저장)
  - data/rag/doc_ids.npy                       # 인덱스 순서에 대응하는 원본 doc_id(문자열/정수)

환경변수:
  - EMB_MODEL   (기본: BAAI/bge-base-en-v1.5)
  - EMB_BATCH   (기본: 32)
  - WIN, STEP   (기본: WIN=1000, STEP=800)
  - EMB_DEVICE  (기본: mps 가능하면 mps, 아니면 cpu)
"""
import os
import pathlib

import faiss
import numpy as np
import orjson
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT = pathlib.Path("../scripts")
IN = ROOT / "data/processed/protocol_corpus_docs.jsonl"
RAG = ROOT / "data/rag";
RAG.mkdir(parents=True, exist_ok=True)
IDX = RAG / "faiss_by_id.index"
IDS = RAG / "doc_ids.npy"

MODEL_NAME = os.environ.get("EMB_MODEL", "BAAI/bge-base-en-v1.5")
BATCH = int(os.environ.get("EMB_BATCH", "32"))
WIN = int(os.environ.get("WIN", "1000"))
STEP = int(os.environ.get("STEP", "800"))
DEVICE = os.environ.get("EMB_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")


def windows(s: str, w: int, step: int):
    s = s or ""
    if len(s) <= w:
        return [s]
    out = []
    i = 0
    while i < len(s):
        out.append(s[i:i + w])
        i += step
    return out


def stream_jsonl(path: pathlib.Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield orjson.loads(line)


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    if mat.ndim == 1:
        den = np.linalg.norm(mat) + 1e-12
        return (mat / den).astype("float32")
    den = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return (mat / den).astype("float32")


def main():
    assert IN.exists(), f"not found: {IN}"

    # 1) doc_ids/texts 로드 (문자열 id도 그대로 보존)
    doc_ids, texts = [], []
    for d in stream_jsonl(IN):
        doc_ids.append(str(d["doc_id"]))  # ← 문자열로 보관
        texts.append(d.get("text", "") or "")
    np.save(IDS, np.array(doc_ids, dtype=object))  # ← 문자열 배열 저장

    # 2) 모델 준비
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)  # cosine 용 (임베딩은 L2 정규화 필요)

    # 3) 문서 임베딩: 세그먼트 임베딩 평균 → L2 정규화 → add
    pbar = tqdm(range(0, len(texts), BATCH), desc=f"embed+add [{MODEL_NAME}|{DEVICE}]")
    for i in pbar:
        batch_txts = texts[i:i + BATCH]
        doc_vecs = []
        for t in batch_txts:
            segs = windows(t, WIN, STEP)
            seg_embs = model.encode(
                segs,
                batch_size=BATCH,  # 세그먼트 인퍼런스 배치
                normalize_embeddings=True,  # 세그먼트 벡터 L2 정규화
                show_progress_bar=False
            )
            vec = seg_embs.mean(axis=0)  # 세그 평균
            vec = l2_normalize(vec)  # 문서 벡터 재정규화
            doc_vecs.append(vec)
        doc_vecs = np.stack(doc_vecs, axis=0).astype("float32")
        index.add(doc_vecs)

    faiss.write_index(index, str(IDX))
    print(f"[OK] FAISS: {IDX} | doc_ids: {IDS} | dim={dim} | device={DEVICE} | n={len(doc_ids)}")


if __name__ == "__main__":
    # MPS에서 일부 연산 미지원 시 CPU 폴백
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
