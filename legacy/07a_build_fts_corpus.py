# scripts/12a_build_fts_corpus.py
"""
입력:
  - data/processed/protocol_corpus_docs.jsonl
출력:
  - data/rag/corpus_fts.sqlite  (FTS5 인덱스)
"""
import pathlib
import sqlite3

import orjson
from tqdm import tqdm

ROOT = pathlib.Path("../scripts")
IN = ROOT / "data/processed/protocol_corpus_docs.jsonl"
DB = ROOT / "data/rag/corpus_fts.sqlite"
DB.parent.mkdir(parents=True, exist_ok=True)


def main():
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(doc_id, title, text, domain, tokenize='porter');")
    n = 0
    with open(IN, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="insert FTS"):
            d = orjson.loads(line)
            cur.execute(
                "INSERT INTO docs(doc_id,title,text,domain) VALUES (?,?,?,?)",
                (d["doc_id"], d["title"], d["text"], d["meta"].get("primary_domain", "Unknown"))
            )
            n += 1
    con.commit();
    con.close()
    print(f"[OK] FTS DB: {DB} (rows={n})")


if __name__ == "__main__":
    main()
