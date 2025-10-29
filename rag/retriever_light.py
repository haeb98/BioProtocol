# rag/retriever_light.py (패치)
import pathlib
import sqlite3

import faiss
import numpy as np
import regex as re
from sentence_transformers import SentenceTransformer

ROOT = pathlib.Path(".")
DB = ROOT / "data/rag/corpus_fts.sqlite"
IDX = ROOT / "data/rag/faiss_by_id.index"
IDS = ROOT / "data/rag/doc_ids.npy"


def _fts_sanitize(q: str) -> str:
    """
    FTS5 MATCH 안전 질의로 정리:
    - 컬럼 한정자/연산자로 오해 받을 수 있는 문자 제거/치환
      :, ^, *, ~, ", ', (), {}, [], | 등
    - AND/OR/NOT 같은 키워드는 공백으로 치환
    - 공백 정리
    - 결과가 비면 구문 검색용으로 전체를 따옴표로 감싼 버전 반환
    """
    if not q:
        return ""
    s = q
    # 연산자성 특수문자 제거
    s = re.sub(r'[:\^\*\~"\'\(\)\{\}\[\]\|]', ' ', s)
    # 불리언 키워드 무력화
    s = re.sub(r'\b(AND|OR|NOT|NEAR)\b', ' ', s, flags=re.I)
    # 공백 정리
    s = re.sub(r'\s+', ' ', s).strip()
    if not s:
        # 빈 문자열이면 구문 검색 폴백
        s = '"' + q.replace('"', '""')[:200] + '"'
    return s


class HybridRetrieverLight:
    def __init__(self, emb_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.con = sqlite3.connect(DB);
        self.con.row_factory = sqlite3.Row
        self.index = faiss.read_index(str(IDX))
        self.doc_ids = np.load(IDS, allow_pickle=True)
        self.model = SentenceTransformer(emb_model)

    def search(self, query: str, k_lex=50, k_sem=50, topn=8):
        # 1) 렉시컬 (FTS5)
        cur = self.con.cursor()
        fts_q = _fts_sanitize(query)
        lex_rows = []
        try:
            cur.execute(
                "SELECT doc_id, title, snippet(docs, 2, '[', ']', '…', 10) AS snip "
                "FROM docs WHERE docs MATCH ? LIMIT ?",
                (fts_q, k_lex)
            )
            lex_rows = cur.fetchall()
        except sqlite3.OperationalError:
            # 폴백: 전체 쿼리를 구문 검색으로
            phrase = '"' + query.replace('"', '""')[:200] + '"'
            cur.execute(
                "SELECT doc_id, title, snippet(docs, 2, '[', ']', '…', 10) AS snip "
                "FROM docs WHERE docs MATCH ? LIMIT ?",
                (phrase, k_lex)
            )
            lex_rows = cur.fetchall()

        lex_ids = [r["doc_id"] for r in lex_rows]

        # 2) 시맨틱 (FAISS)
        qv = self.model.encode([query], normalize_embeddings=True).astype("float32")
        D, I = self.index.search(qv, k_sem)
        sem_ids = [self.doc_ids[i] for i in I[0]]

        # 3) 병합(간단 가중)
        scores = {}
        for rank, did in enumerate(lex_ids):
            scores[did] = scores.get(did, 0.0) + 1.0 / (rank + 1)
        for rank, did in enumerate(sem_ids):
            scores[did] = scores.get(did, 0.0) + 0.5 / (rank + 1)

        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topn]
        out = []
        for did, sc in top:
            row = self.con.execute(
                "SELECT title, text as preview FROM docs WHERE doc_id=?",
                (did,)
            ).fetchone()
            out.append({"doc_id": did, "title": row["title"], "score": sc, "text_preview": row["preview"]})
        return out
