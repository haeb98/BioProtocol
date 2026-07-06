#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
[스크립트 3] Methods(sec_text) 기반 문장 grounding rate 비교

왜 필요한가(이유)
- gold protocol과 generated protocol 모두, "문장들이 Methods에서 근거를 찾을 수 있는가"를 비교해야
  hallucination(혹은 근거 부족) 성향을 정량적으로 비교 가능.
- 단일 LLM vs 멀티에이전트(P1~P6) 비교에서 "Verifier 전후 개선"을 보여줄 때도 유용.

공식 문서 근거(링크/출처 태그)
- sentence-transformers cosine similarity: https://www.sbert.net/
- chunking은 긴 문서 검색/매칭에서 흔히 쓰는 실무적 분할 전략(검색/유사도 계산 안정화 목적)
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sentence_transformers import SentenceTransformer, util


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _as_str(x: Any) -> str:
    return x.strip() if isinstance(x, str) else ""


def extract_methods_text(rec: Dict[str, Any]) -> str:
    if _as_str(rec.get("sec_text")):
        return rec["sec_text"].strip()
    article = rec.get("article") or {}
    if _as_str(article.get("sec_text")):
        return article["sec_text"].strip()
    sections = article.get("sections")
    if isinstance(sections, dict) and _as_str(sections.get("Methods")):
        return sections["Methods"].strip()
    return ""


def _collect_sentences_from_any(obj: Any, out: List[str]) -> None:
    if obj is None:
        return
    if isinstance(obj, str):
        s = _norm_ws(obj)
        if s:
            out.append(s)
        return
    if isinstance(obj, list):
        for x in obj:
            _collect_sentences_from_any(x, out)
        return
    if isinstance(obj, dict):
        for k in ["step_text", "text", "sentence", "content", "description", "instruction"]:
            if isinstance(obj.get(k), str) and obj[k].strip():
                out.append(_norm_ws(obj[k]))
        for v in obj.values():
            _collect_sentences_from_any(v, out)


def extract_gold_hier_protocol_sentences(rec: Dict[str, Any]) -> List[str]:
    candidates = []
    for k in ["hierarchical_protocol", "hierarchicalProtocol", "protocol", "gold_protocol", "goldProtocol"]:
        if k in rec and rec[k] is not None:
            candidates.append(rec[k])
    article = rec.get("article") or {}
    for k in ["hierarchical_protocol", "protocol", "gold_protocol"]:
        if k in article and article[k] is not None:
            candidates.append(article[k])

    sents: List[str] = []
    if candidates:
        for c in candidates:
            _collect_sentences_from_any(c, sents)
    else:
        _collect_sentences_from_any(rec, sents)

    uniq, seen = [], set()
    for s in sents:
        if len(s) < 3:
            continue
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + chunk_chars])
        if i + chunk_chars >= len(text):
            break
        i += max(1, chunk_chars - overlap)
    return chunks


def grounded_rate(
        model: SentenceTransformer,
        methods_text: str,
        sentences: List[str],
        thr: float,
        chunk_chars: int,
        overlap: int,
) -> Dict[str, Any]:
    sents = [s for s in (sentences or []) if isinstance(s, str) and s.strip()]
    if not sents:
        return {"grounded_rate": 0.0, "ungrounded_rate": 0.0, "evaluable": 0, "grounded": 0}

    chunks = chunk_text(methods_text or "", chunk_chars, overlap)
    if not chunks:
        return {"grounded_rate": 0.0, "ungrounded_rate": 1.0, "evaluable": len(sents), "grounded": 0}

    chunk_emb = model.encode(chunks, convert_to_tensor=True, show_progress_bar=False)

    grounded = 0
    for s in sents:
        s_emb = model.encode([s], convert_to_tensor=True, show_progress_bar=False)
        sim = util.cos_sim(s_emb, chunk_emb).cpu().numpy().flatten()
        if sim.max() >= thr:
            grounded += 1

    rate = grounded / len(sents)
    return {"grounded_rate": rate, "ungrounded_rate": 1.0 - rate, "evaluable": len(sents), "grounded": grounded}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default="/Users/haeb/Workspaces/BioProtocol/BioProtocolAgent")
    ap.add_argument("--gold_pairs", type=str, default="data/gold_pairs_testset_v2.jsonl")

    ap.add_argument("--generated_dir", type=str, default="reports/llm_protocols")
    ap.add_argument("--pattern", type=str, default="generated_P*.jsonl")
    ap.add_argument("--out_dir", type=str, default="reports/grounding_eval")

    ap.add_argument("--embed_model", type=str, default="allenai/scibert_scivocab_uncased")
    ap.add_argument("--embed_device", type=str, default=None)
    ap.add_argument("--grounding_threshold", type=float, default=0.60)
    ap.add_argument("--chunk_chars", type=int, default=1200)
    ap.add_argument("--chunk_overlap", type=int, default=200)
    args = ap.parse_args()

    root = Path(args.project_root)
    gold_pairs_path = root / args.gold_pairs
    gen_dir = root / args.generated_dir
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # gold maps
    methods_map: Dict[str, str] = {}
    goldproto_map: Dict[str, List[str]] = {}
    for rec in load_jsonl(gold_pairs_path):
        pid = rec.get("protocol_id")
        if not pid:
            continue
        methods_map[pid] = extract_methods_text(rec)
        goldproto_map[pid] = extract_gold_hier_protocol_sentences(rec)

    model = SentenceTransformer(args.embed_model, device=args.embed_device)

    rows = []

    # GOLD grounding
    for pid, gold_sents in goldproto_map.items():
        gr = grounded_rate(model, methods_map.get(pid, ""), gold_sents, args.grounding_threshold, args.chunk_chars,
                           args.chunk_overlap)
        rows.append({"mode": "GOLD", "protocol_id": pid, "n_sents": len(gold_sents), **gr})

    # P1~P6 grounding
    gen_files = sorted(gen_dir.glob(args.pattern))
    if not gen_files:
        raise FileNotFoundError(f"No generated files matched: {gen_dir / args.pattern}")

    for gen_path in gen_files:
        mode = gen_path.stem.replace("generated_", "")
        recs = load_jsonl(gen_path)
        for rec in recs:
            pid = rec["protocol_id"]
            pred_sents = rec.get("sentences", []) or []
            gr = grounded_rate(model, methods_map.get(pid, ""), pred_sents, args.grounding_threshold, args.chunk_chars,
                               args.chunk_overlap)
            rows.append({"mode": mode, "protocol_id": pid,
                         "n_sents": len([s for s in pred_sents if isinstance(s, str) and s.strip()]), **gr})

    df = pd.DataFrame(rows).sort_values(["mode", "protocol_id"])
    out_pp = out_dir / "per_protocol_grounding.csv"
    df.to_csv(out_pp, index=False)

    df_mode = df.groupby("mode").agg({
        "grounded_rate": "mean",
        "ungrounded_rate": "mean",
        "evaluable": "mean",
        "grounded": "mean",
        "n_sents": "mean",
    }).reset_index()
    out_sum = out_dir / "summary_modes_grounding.csv"
    df_mode.to_csv(out_sum, index=False)

    print(f"[SAVE] {out_pp}")
    print(f"[SAVE] {out_sum}")
    print("✅ Done.")


if __name__ == "__main__":
    main()
