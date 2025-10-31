#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_40_bm25_sqlite_index.py
- 입력 JSONL 코퍼스 → SQLite FTS5 테이블 생성
"""

import argparse
import json
import pathlib
import sqlite3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="data/rag/corpus/protocols_wo_test25.jsonl")
    ap.add_argument("--db", default="data/rag/indexes/bm25_protocols.sqlite")
    args = ap.parse_args()

    dbp = pathlib.Path(args.db);
    dbp.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(dbp))
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(id, title, domain, text, tokenize='unicode61');")
    cur.execute("DELETE FROM docs;")

    n = 0
    with open(args.corpus, "r", encoding="utf-8") as fin:
        for line in fin:
            try:
                rec = json.loads(line)
            except:
                continue
            cur.execute("INSERT INTO docs(id, title, domain, text) VALUES(?,?,?,?)",
                        (rec.get("id", ""), rec.get("title", ""), rec.get("domain", "Unknown"), rec.get("text", "")))
            n += 1
    con.commit();
    con.close()
    print(f"[OK] BM25(SQLite) -> {dbp} (rows={n})")


if __name__ == "__main__":
    main()
