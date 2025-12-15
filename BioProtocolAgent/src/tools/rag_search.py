# src/tools/rag_search.py
import json
import os
from typing import List, Dict, Any, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === 경로 설정 ===
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../src/tools
SRC_DIR = os.path.dirname(THIS_DIR)  # .../src
PROJECT_ROOT = os.path.dirname(os.path.dirname(SRC_DIR))  # .../BioProtocol

CORPUS_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "rag",
    "corpus",
    "protocols_wo_test50.annot.jsonl",
)

# ⚠️ 이 키 이름은 실제 파일 구조에 맞게 조정해야 할 수 있음
#   예: "protocol", "input", "text", "protocol_text" 등
TEXT_KEY_CANDIDATES = ["protocol", "text", "input", "full_text", "protocol_text"]


class RagTfidfEngine:
    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
        self.docs: List[Dict[str, Any]] = []
        self.texts: List[str] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_matrix = None

        self._load_corpus()
        self._build_index()

    def _extract_text(self, rec: Dict[str, Any]) -> Optional[str]:
        """
        JSON 레코드에서 실제 검색할 텍스트 필드를 찾는다.
        파일 구조에 맞게 TEXT_KEY_CANDIDATES 를 조정해도 된다.
        """
        for k in TEXT_KEY_CANDIDATES:
            if k in rec and isinstance(rec[k], str) and rec[k].strip():
                return rec[k]
        return None

    def _load_corpus(self):
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"RAG corpus not found: {self.corpus_path}")

        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text = self._extract_text(rec)
                if not text:
                    continue

                pid = rec.get("protocol_id") or rec.get("id") or rec.get("bio_id")
                self.docs.append({
                    "protocol_id": pid,
                    "text": text,
                    "raw": rec,
                })
                self.texts.append(text)

        print(f"[RAG] Loaded {len(self.docs)} documents from {self.corpus_path}")

    def _build_index(self):
        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            stop_words="english",
        )
        self.doc_matrix = self.vectorizer.fit_transform(self.texts)
        print(f"[RAG] Built TF-IDF index with shape {self.doc_matrix.shape}")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query.strip():
            return []

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_matrix)[0]

        # 상위 top_k 인덱스
        top_idx = sims.argsort()[::-1][:top_k]

        hits: List[Dict[str, Any]] = []
        for idx in top_idx:
            score = float(sims[idx])
            if score <= 0:
                continue
            doc = self.docs[idx]
            hits.append({
                "protocol_id": doc["protocol_id"],
                "text": doc["text"],
                "score": score,
                "source": "external_protocol",  # 🔹 ReAct trace에서 구분용
            })
        return hits


# === 싱글톤 래퍼 ===
_ENGINE: Optional[RagTfidfEngine] = None


def _get_engine() -> RagTfidfEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = RagTfidfEngine(CORPUS_PATH)
    return _ENGINE


def rag_tool(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    ReAct tool용 래퍼.
    """
    engine = _get_engine()
    hits = engine.search(query, top_k=top_k)
    return {"hits": hits}
