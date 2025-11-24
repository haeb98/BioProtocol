#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S2B Task Grounder (Hybrid, robust)
- BM25 (SQLite FTS5, safe MATCH) + FAISS(HNSW) 재랭킹
- DB는 docs(rowid, text)만 있다고 가정 (meta_json 의존 X)
- router_hint/role 부스팅은 corpus JSONL 기반(meta_map)에서만 처리
- 다양한 params 형식(list[str]/list[dict]/[]) 자동 정규화
- 증거 스니펫/파라미터 후보 추출 및 부착

Inputs:
  --ir       runs/s2_parser.ir.jsonl            # S2 Parser 산출 IR
  --bm25     data/rag/indexes/bm25_protocols.sqlite
  --faiss-dir data/rag/indexes/faiss_protocols (없으면 WARN 후 BM25만 사용)
  --corpus   data/rag/corpus/protocols_wo_test50.annot.jsonl
  --out      runs/s2b_grounded.ir.jsonl
  --topk     10
  --embed-model sentence-transformers/all-MiniLM-L6-v2 (default)
"""

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ============== FTS5 safe helpers ==============
_FTS5_TOKEN_RX = re.compile(r"\w+")


def fts5_sanitize(q: str, max_terms: int = 16) -> str:
    """임의 텍스트 → FTS5 MATCH 안전 문자열."""
    if not q:
        return "protocol"
    toks = _FTS5_TOKEN_RX.findall(q.lower())
    if not toks:
        return "protocol"
    return " ".join(toks[:max_terms])


# ============== Regex helpers (숫자+단위) ==============
NUMUNIT_RX = re.compile(
    r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>"
    r"(?:°c|degc|c|k|°f|min|mins|minute|minutes|h|hr|hrs|hour|hours|s|sec|secs|second|seconds|"
    r"ml|ul|l|µl|μl|nm|mm|cm|µm|μm|rpm|g|xg|mg/ml|ng/ml|µg/ml|μg/ml|mM|µM|μM|nM|%|v/v|w/v)"
    r")\b", re.I)


def extract_num_units(text: str, max_hits: int = 12) -> List[Dict[str, Any]]:
    out = []
    for m in NUMUNIT_RX.finditer(text or ""):
        val = m.group("val")
        unit = m.group("unit")
        out.append({"name": None, "value": val, "unit": unit, "raw": m.group(0)})
        if len(out) >= max_hits:
            break
    return out


def guess_name_from_text(s: str) -> Optional[str]:
    if ":" in s:
        left = s.split(":", 1)[0].strip()
        return left or None
    return None


# ============== Params normalizer ==============
def normalize_params(params: Any) -> List[Dict[str, Any]]:
    """
    입력:
      - list[str]
      - list[dict]
      - dict / str / None
    출력:
      - list[dict] with keys: name, value, unit, raw, source
    """
    out: List[Dict[str, Any]] = []
    if not params:
        return out
    if isinstance(params, list):
        for p in params:
            if isinstance(p, dict):
                out.append({
                    "name": p.get("name"),
                    "value": p.get("value"),
                    "unit": p.get("unit"),
                    "raw": p.get("raw"),
                    "source": p.get("source"),
                })
            elif isinstance(p, str):
                raw = p.strip()
                m = NUMUNIT_RX.search(raw)
                name = guess_name_from_text(raw)
                val = m.group("val") if m else None
                unit = m.group("unit") if m else None
                out.append({"name": name, "value": val, "unit": unit, "raw": raw, "source": None})
            else:
                out.append({"name": None, "value": None, "unit": None, "raw": str(p), "source": None})
    else:
        if isinstance(params, dict):
            out.append({
                "name": params.get("name"),
                "value": params.get("value"),
                "unit": params.get("unit"),
                "raw": params.get("raw"),
                "source": params.get("source"),
            })
        elif isinstance(params, str):
            raw = params.strip()
            m = NUMUNIT_RX.search(raw)
            name = guess_name_from_text(raw)
            val = m.group("val") if m else None
            unit = m.group("unit") if m else None
            out.append({"name": name, "value": val, "unit": unit, "raw": raw, "source": None})
    return out


def merge_param_candidates(base: List[Dict[str, Any]], cands: List[Dict[str, Any]],
                           source_info: Dict[str, Any], max_total=24) -> List[Dict[str, Any]]:
    base = base or []
    for c in cands:
        d = {
            "name": c.get("name"),
            "value": c.get("value"),
            "unit": c.get("unit"),
            "raw": c.get("raw"),
            "source": source_info,
        }
        base.append(d)
        if len(base) >= max_total:
            break
    return base


# ============== BM25 wrapper (SQLite FTS5, meta_json 없음 가정) ==============
class BM25Index:
    """
    docs(rowid, text)만 있다고 가정.
    meta_json은 전혀 사용하지 않고, 메타는 corpus JSONL에서만 읽는다.
    """

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def search(self, query: str, topk: int = 10) -> List[Dict[str, Any]]:
        safe_q = fts5_sanitize(query)
        cur = self.conn.cursor()

        rows = []
        # 1차: bm25(docs) 사용
        try:
            cur.execute(
                """SELECT rowid, bm25(docs) AS bm25_score
                   FROM docs
                   WHERE docs MATCH ?
                   ORDER BY bm25_score
                   LIMIT ?""",
                (safe_q, topk * 3),
            )
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            # 2차: bm25() 미지원/문법 에러 → 점수 없이 rowid만
            cur.execute(
                """SELECT rowid
                   FROM docs
                   WHERE docs MATCH ?
                   LIMIT ?""",
                (safe_q, topk * 3),
            )
            tmp = cur.fetchall()
            for r in tmp:
                r = dict(r)
                r["bm25_score"] = 10.0
                rows.append(r)

        hits: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            base = float(d.get("bm25_score", 10.0))
            score = 1.0 / (1.0 + base)  # 작을수록 좋은 bm25 → 클수록 좋은 score
            hits.append({"rowid": int(d["rowid"]), "score": float(score)})
        return hits


# ============== FAISS wrapper ==============
class FAISSIndex:
    def __init__(self, index_dir: str, model_name: str):
        from sentence_transformers import SentenceTransformer
        import faiss, numpy as np  # noqa: F401

        self.model = SentenceTransformer(model_name)

        index_dir = Path(index_dir)
        # 파일명 유연하게 처리: faiss.index 우선, 없으면 index.faiss 시도
        cand1 = index_dir / "faiss.index"
        cand2 = index_dir / "index.faiss"
        if cand1.exists():
            index_path = cand1
        elif cand2.exists():
            index_path = cand2
        else:
            raise RuntimeError(f"FAISS index file not found in {index_dir}")

        self.index = faiss.read_index(str(index_path))

        ids_path = index_dir / "rowids.npy"
        if not ids_path.exists():
            raise RuntimeError(f"Missing rowids.npy in {index_dir}")
        self.ids = np.load(str(ids_path))

    def encode(self, text: str):
        import numpy as np
        v = self.model.encode([text or ""], normalize_embeddings=True)
        return np.asarray(v, dtype="float32")

    def search(self, query: str, topk: int = 20) -> List[Dict[str, Any]]:
        qv = self.encode(query)
        D, I = self.index.search(qv, topk)
        hits = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            rowid = int(self.ids[idx])
            score = float(1.0 - dist) if dist <= 1.0 else float(dist)
            hits.append({"rowid": rowid, "score": score})
        return hits


# ============== Corpus loader (rowid alignment) ==============
def load_corpus_maps(path: str) -> Tuple[Dict[int, str], Dict[int, Dict[str, Any]]]:
    """
    line 번호(1-based) == rowid 로 가정.
    각 line: {"text": "...", "meta": {...}} 또는
             {"chunk": "...", "id": ..., "url": ..., ...}
    """
    text_map: Dict[int, str] = {}
    meta_map: Dict[int, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                r = json.loads(line)
            except:
                continue
            text = r.get("text") or r.get("chunk") or ""
            text_map[i] = text
            meta = r.get("meta") or {}
            for k in ("id", "url", "title", "router_hint", "role"):
                if k not in meta and k in r:
                    meta[k] = r[k]
            meta_map[i] = meta
    return text_map, meta_map


# ============== Hybrid search ==============
def hybrid_search(query: str, topk: int,
                  bm25: Optional[BM25Index],
                  faiss: Optional[FAISSIndex],
                  text_map: Dict[int, str],
                  meta_map: Dict[int, Dict[str, Any]],
                  router_hint: Optional[str]) -> List[Dict[str, Any]]:
    # 1) BM25
    bm_hits = bm25.search(query, topk=topk) if bm25 else []

    # 2) FAISS
    fa_hits: List[Dict[str, Any]] = []
    if faiss:
        fa = faiss.search(query, topk=topk * 2)
        for h in fa:
            fa_hits.append({"rowid": int(h["rowid"]), "score": float(h["score"])})

    # 3) score merge (max)
    agg: Dict[int, float] = {}
    for h in bm_hits:
        rid = int(h["rowid"])
        agg[rid] = max(agg.get(rid, 0.0), float(h["score"]))
    for h in fa_hits:
        rid = int(h["rowid"])
        agg[rid] = max(agg.get(rid, 0.0), float(h["score"]))

    # 4) meta 기반 router_hint 부스팅 + 최종 리스트
    out: List[Dict[str, Any]] = []
    for rid, sc in agg.items():
        meta = meta_map.get(rid, {})
        rh = (meta.get("router_hint") or "").lower()
        final_sc = sc
        if router_hint and rh == router_hint.lower():
            final_sc *= 1.05
        out.append({
            "rowid": rid,
            "score": float(final_sc),
            "text": text_map.get(rid, ""),
            "meta": meta,
        })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:topk]


# ============== IR IO ==============
def iter_ir(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except:
                continue
            yield r


def write_ir(path: str, recs: List[Dict[str, Any]]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ============== Query builder for Step ==============
def build_query_for_step(step: Dict[str, Any], task_title: Optional[str]) -> Tuple[str, Optional[str]]:
    title = (step.get("title") or "").strip()
    action = (step.get("action") or "").strip()
    mats: List[str] = []
    for m in (step.get("materials") or []):
        if isinstance(m, dict):
            nm = m.get("name") or m.get("material") or ""
            if nm:
                mats.append(nm)
        elif isinstance(m, str):
            mats.append(m)
    query = " ".join([task_title or "", title, action, " ".join(mats)]).strip()
    if not query:
        query = task_title or title or "protocol"

    rh = None
    low = (title + " " + action).lower()
    if any(k in low for k in ("measure", "quantify", "assay", "read", "od600", "absorbance", "fluorescence")):
        rh = "measurement"
    elif any(k in low for k in ("mix", "prepare", "dilute", "buffer", "solution", "stock")):
        rh = "recipe"
    return query, rh


# ============== Evidence attach ==============
def make_snippet(text: str, max_len: int = 220) -> str:
    t = (text or "").replace("\n", " ")
    return (t[:max_len] + "…") if len(t) > max_len else t


# def attach_evidence_and_params(step: Dict[str, Any], hits: List[Dict[str, Any]]) -> Dict[str, Any]:
#     step["params"] = normalize_params(step.get("params"))
#     ev_list = step.get("evidence") or []
#     pc_list = step.get("param_candidates") or []
#
#     for rank, h in enumerate(hits[:1], 1):  # top-1 evidence
#         meta = h.get("meta") or {}
#         ev = {
#             "rank": rank,
#             "rowid": h["rowid"],
#             "score": h["score"],
#             "doc_id": meta.get("id"),
#             "url": meta.get("url"),
#             "title": meta.get("title"),
#             "router_hint": meta.get("router_hint"),
#             "snippet": make_snippet(h.get("text", "")),
#         }
#         ev_list.append(ev)
#
#         cands = extract_num_units(h.get("text", ""))
#         src = {"rowid": h["rowid"], "url": meta.get("url"), "title": meta.get("title")}
#         pc_list = merge_param_candidates(pc_list, cands, src, max_total=24)
#
#     step["evidence"] = ev_list
#     step["param_candidates"] = pc_list
#     return step

def maybe_attach_evidence_only(step: Dict[str, Any],
                               hits: List[Dict[str, Any]],
                               topk: int = 1) -> Dict[str, Any]:
    """
    step: IR의 단일 Step (S2 Parser output)
    hits: hybrid_search 결과 리스트 (각 원소는 BM25/FAISS hit dict)

    이 함수는 evidence만 붙이고,
    step["params"]나 param_candidates는 건드리지 않는다.
    """

    # evidence 리스트 초기화/유지
    ev_list = step.get("evidence") or []
    step["evidence"] = ev_list

    for rank, h in enumerate(hits[:topk], 1):
        meta = h.get("meta") or {}
        ev = {
            "rank": rank,
            "rowid": h.get("rowid"),
            "score": h.get("score", 0.0),
            "doc_id": meta.get("id"),
            "url": meta.get("url"),
            "title": meta.get("title"),
            "router_hint": meta.get("router_hint"),
            # 원래 코드와 동일하게 text에서 snippet 생성
            "snippet": make_snippet(h.get("text", "")),
        }
        ev_list.append(ev)

    return step


# ============== MAIN ==============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir", required=True)
    ap.add_argument("--bm25", required=False, default=None)
    ap.add_argument("--faiss-dir", required=False, default=None)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    # corpus → text_map, meta_map
    text_map, meta_map = load_corpus_maps(args.corpus)

    # BM25
    bm25 = BM25Index(args.bm25) if args.bm25 else None

    # FAISS (없으면 BM25만 사용)
    faiss = None
    if args.faiss_dir:
        try:
            faiss = FAISSIndex(args.faiss_dir, args.embed_model)
        except Exception as e:
            print(f"[WARN] FAISS disabled ({e})")

    grounded: List[Dict[str, Any]] = []
    n_steps, n_ev = 0, 0

    for rec in iter_ir(args.ir):
        nodes = rec.get("nodes") or []
        new_nodes: List[Dict[str, Any]] = []

        for n in nodes:
            if n.get("type") != "Step":
                new_nodes.append(n)
                continue
            task_ref = n.get("task_ref")
            task_title = None  # 필요하면 S2A 결과에서 추가로 매핑

            query, r_hint = build_query_for_step(n, task_title)
            hits = hybrid_search(query, args.topk, bm25, faiss, text_map, meta_map, r_hint)
            n = maybe_attach_evidence_only(n, hits, topk=args.topk)
            new_nodes.append(n)
            n_steps += 1
            if hits:
                n_ev += 1

        out_rec = dict(rec)
        out_rec["nodes"] = new_nodes
        grounded.append(out_rec)

    write_ir(args.out, grounded)
    print(f"[OK] grounded IR -> {args.out} (steps={n_steps}, with_evidence={n_ev})")


if __name__ == "__main__":
    main()
