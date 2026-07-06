#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
[스크립트 2] gold hierarchical protocol vs P1~P6 generated protocol 비교

왜 필요한가(이유)
- IR(action dict)끼리 비교하면 "실제 자연어 프로토콜 품질"과 연결이 약함.
- 따라서 Writer(LLM)가 만든 자연어 프로토콜을 gold hierarchical protocol과 직접 비교:
  BLEU/ROUGE 같은 표준 지표 + 임베딩 기반 문장 매칭(step F1) + TF-IDF cosine.

공식 문서 근거(링크/출처 태그)
- sacrebleu: https://github.com/mjpost/sacrebleu
- rouge-score: https://github.com/google-research/google-research/tree/master/rouge
- sentence-transformers: https://www.sbert.net/
- TF-IDF: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


def _join_sents(sents: List[str]) -> str:
    return "\n".join([_norm_ws(s) for s in sents if _norm_ws(s)]).strip()


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


def load_gold_maps(gold_pairs_path: Path) -> Dict[str, List[str]]:
    goldproto_map: Dict[str, List[str]] = {}
    for rec in load_jsonl(gold_pairs_path):
        pid = rec.get("protocol_id")
        if not pid:
            continue
        goldproto_map[pid] = extract_gold_hier_protocol_sentences(rec)
    return goldproto_map


def compute_bleu_rouge(pred_docs: List[str], ref_docs: List[str]) -> Dict[str, float]:
    out = {"bleu": 0.0, "rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}

    try:
        import sacrebleu  # type: ignore
        out["bleu"] = float(sacrebleu.corpus_bleu(pred_docs, [ref_docs]).score)
    except Exception:
        pass

    try:
        from rouge_score import rouge_scorer  # type: ignore
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1 = r2 = rl = 0.0
        n = 0
        for p, r in zip(pred_docs, ref_docs):
            s = scorer.score(r, p)  # ref, pred
            r1 += s["rouge1"].fmeasure
            r2 += s["rouge2"].fmeasure
            rl += s["rougeL"].fmeasure
            n += 1
        if n:
            out["rouge1_f"] = r1 / n
            out["rouge2_f"] = r2 / n
            out["rougeL_f"] = rl / n
    except Exception:
        pass

    return out


def embed_step_metrics(model: SentenceTransformer, gold_sents: List[str], pred_sents: List[str], thr: float) -> Tuple[
    float, float, float]:
    g = [s for s in gold_sents if s.strip()]
    p = [s for s in pred_sents if s.strip()]
    if not g or not p:
        return 0.0, 0.0, 0.0

    emb = model.encode(g + p, convert_to_tensor=True, show_progress_bar=False)
    g_emb = emb[: len(g)]
    p_emb = emb[len(g):]
    sim = util.cos_sim(p_emb, g_emb).cpu().numpy()  # (P,G)

    matched_pred = int(sum(1 for i in range(len(p)) if sim[i].max() >= thr))
    matched_gold = int(sum(1 for j in range(len(g)) if sim[:, j].max() >= thr))

    sp = matched_pred / len(p)
    sr = matched_gold / len(g)
    f1 = 2 * sp * sr / (sp + sr) if (sp + sr) else 0.0
    return sp, sr, f1


def tfidf_cosine(gold_sents: List[str], pred_sents: List[str]) -> float:
    g_txt = _join_sents(gold_sents)
    p_txt = _join_sents(pred_sents)
    if not g_txt or not p_txt:
        return 0.0
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=20000)
    X = vec.fit_transform([g_txt, p_txt])
    return float(cosine_similarity(X[0], X[1])[0, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default="/Users/haeb/Workspaces/BioProtocol/BioProtocolAgent")
    ap.add_argument("--gold_pairs", type=str, default="data/gold_pairs_testset_v2.jsonl")
    ap.add_argument("--generated_dir", type=str, default="reports/llm_protocols")
    ap.add_argument("--pattern", type=str, default="generated_P*.jsonl")
    ap.add_argument("--out_dir", type=str, default="reports/protocol_eval")

    ap.add_argument("--embed_model", type=str, default="allenai/scibert_scivocab_uncased")
    ap.add_argument("--embed_device", type=str, default=None)
    ap.add_argument("--embed_threshold", type=float, default=0.70)
    args = ap.parse_args()

    root = Path(args.project_root)
    gold_pairs_path = root / args.gold_pairs
    generated_dir = root / args.generated_dir
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    goldproto_map = load_gold_maps(gold_pairs_path)

    model = SentenceTransformer(args.embed_model, device=args.embed_device)

    gen_files = sorted(generated_dir.glob(args.pattern))
    if not gen_files:
        raise FileNotFoundError(f"No generated files matched: {generated_dir / args.pattern}")

    per_protocol_rows = []
    summary_rows = []

    for gen_path in gen_files:
        mode = gen_path.stem.replace("generated_", "")
        recs = load_jsonl(gen_path)

        pred_docs, ref_docs = [], []
        for rec in recs:
            pid = rec["protocol_id"]
            pred_sents = rec.get("sentences", []) or []
            gold_sents = goldproto_map.get(pid, []) or []

            pred_doc = _join_sents(pred_sents)
            gold_doc = _join_sents(gold_sents)
            pred_docs.append(pred_doc)
            ref_docs.append(gold_doc)

            sp, sr, f1 = embed_step_metrics(model, gold_sents, pred_sents, args.embed_threshold)
            tfidf_sim = tfidf_cosine(gold_sents, pred_sents)

            per_protocol_rows.append({
                "mode": mode,
                "protocol_id": pid,
                "n_gold_sents": len(gold_sents),
                "n_pred_sents": len([s for s in pred_sents if s.strip()]),
                "embed_step_precision": sp,
                "embed_step_recall": sr,
                "embed_step_f1": f1,
                "tfidf_cosine": tfidf_sim,
            })

        bleu_rouge = compute_bleu_rouge(pred_docs, ref_docs)
        df_mode = pd.DataFrame([r for r in per_protocol_rows if r["mode"] == mode])
        summary_rows.append({
            "mode": mode,
            **bleu_rouge,
            "avg_embed_step_f1": float(df_mode["embed_step_f1"].mean()) if not df_mode.empty else 0.0,
            "avg_tfidf_cosine": float(df_mode["tfidf_cosine"].mean()) if not df_mode.empty else 0.0,
            "embed_model": args.embed_model,
            "embed_threshold": args.embed_threshold,
        })

    df_pp = pd.DataFrame(per_protocol_rows).sort_values(["mode", "protocol_id"])
    df_sum = pd.DataFrame(summary_rows).sort_values(["mode"])

    out_pp = out_dir / "per_protocol.csv"
    out_sum = out_dir / "summary_modes.csv"
    df_pp.to_csv(out_pp, index=False)
    df_sum.to_csv(out_sum, index=False)

    print(f"[SAVE] {out_pp}")
    print(f"[SAVE] {out_sum}")
    print("✅ Done.")


if __name__ == "__main__":
    main()
