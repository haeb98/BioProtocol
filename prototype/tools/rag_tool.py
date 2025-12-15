# prototype/tools/rag_tool.py

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize once (globally)
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
FAISS_INDEX_PATH = "data/rag/indexes/faiss_protocols/faiss.index"
IDS_PATH = "data/rag/indexes/faiss_protocols/ids.txt"
INDEX = faiss.read_index(FAISS_INDEX_PATH)

with open(IDS_PATH, "r", encoding="utf-8") as f:
    ID_TEXTS = [line.strip() for line in f.readlines()]


def rag_tool(query, top_k=3):
    """
    Retrieve top-k semantically similar protocol chunks from FAISS index.
    """
    emb = MODEL.encode([query])
    D, I = INDEX.search(np.array(emb).astype('float32'), top_k)
    retrieved = [ID_TEXTS[i] for i in I[0]]
    return "\n\n".join(retrieved)
