#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_41_faiss_hnsw_index.py
- 입력 JSONL 코퍼스 → 임베딩 → FAISS HNSW 인덱스
- 필요: faiss-cpu, sentence-transformers
"""

import argparse
import faiss
import json
import numpy as np
import pathlib

from sentence_transformers import SentenceTransformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--outdir", default="data/rag/indexes/faiss_protocols")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")  # 환경에 맞게 변경 가능
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--dim", type=int, default=384)  # all-MiniLM-L6-v2 = 384
    ap.add_argument("--M", type=int, default=32)  # HNSW M
    ap.add_argument("--efC", type=int, default=200)  # HNSW efConstruction
    args = ap.parse_args()

    outd = pathlib.Path(args.outdir);
    outd.mkdir(parents=True, exist_ok=True)

    ids = [];
    texts = []
    with open(args.corpus, "r", encoding="utf-8") as fin:
        for line in fin:
            try:
                rec = json.loads(line)
            except:
                continue
            ids.append(rec.get("id", ""))
            texts.append(rec.get("text", ""))

    if not texts:
        print("[WARN] no texts loaded");
        return

    model = SentenceTransformer(args.model)
    embs = []
    for i in range(0, len(texts), args.batch_size):
        embs.append(model.encode(texts[i:i + args.batch_size], show_progress_bar=False, convert_to_numpy=True,
                                 normalize_embeddings=True))
    X = np.vstack(embs).astype("float32")

    index = faiss.IndexHNSWFlat(args.dim, args.M)
    index.hnsw.efConstruction = args.efC
    index.add(X)

    faiss.write_index(index, str(outd / "faiss.index"))
    (outd / "ids.txt").write_text("\n".join(ids), encoding="utf-8")

    print(f"[OK] FAISS(HNSW) -> {outd} (vecs={len(ids)}, dim={args.dim})")


if __name__ == "__main__":
    main()
