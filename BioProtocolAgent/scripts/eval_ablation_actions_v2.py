# scripts/eval_ablation_actions_v2.py
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer, util

# 프로젝트 루트 등록
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

GOLD_ACTIONS_PATH = PROJECT_ROOT / "data" / "gold_actions_ir_10.jsonl"
GOLD_PAIRS_PATH = PROJECT_ROOT / "data" / "gold_pairs_testset_v2.jsonl"
ABLATION_DIR = PROJECT_ROOT / "data" / "ablation"

STEP_MATCH_THRESH = 0.7  # gold vs gen step 매칭 임계값
GROUNDING_THRESH = 0.55  # methods grounding 임계값 (chunk 기반)
METHODS_CHUNK_CHARS = 1200  # methods chunk 크기(대충 800~1500 사이 추천)
METHODS_CHUNK_OVERLAP = 200  # overlap


# -------------------------
# JSONL 로드
# -------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    recs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def build_protocol_map(recs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for r in recs:
        pid = r.get("protocol_id")
        if pid:
            out[pid] = r
    return out


# -------------------------
# gold_pairs 에서 methods/sec_text 로드
# -------------------------

def extract_methods_text(rec: Dict[str, Any]) -> str:
    # 1) top-level sec_text
    if isinstance(rec.get("sec_text"), str) and rec["sec_text"].strip():
        return rec["sec_text"].strip()

    article = rec.get("article") or {}

    # 2) article.sec_text
    if isinstance(article.get("sec_text"), str) and article["sec_text"].strip():
        return article["sec_text"].strip()

    # 3) article.sections.Methods
    sections = article.get("sections")
    if isinstance(sections, dict):
        m = sections.get("Methods")
        if isinstance(m, str) and m.strip():
            return m.strip()

    return ""


def load_methods_map(gold_pairs_path: Path) -> Dict[str, str]:
    out = {}
    recs = load_jsonl(gold_pairs_path)
    for r in recs:
        pid = r.get("protocol_id")
        if not pid:
            continue
        txt = extract_methods_text(r)
        if txt:
            out[pid] = txt
    return out


# -------------------------
# action text / materials / conditions
# -------------------------

def get_action_text(a: Dict[str, Any]) -> str:
    # 기존 eval_ablation_actions.py와 동일한 방식 유지 (action+description) :contentReference[oaicite:7]{index=7}
    return "action:" + (a.get("action") or "None") + ", description:" + (a.get("description") or "None")


def normalize_token(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace()).strip()


def extract_materials(a: Dict[str, Any]) -> List[str]:
    mats = a.get("materials") or []
    norm = []
    for m in mats:
        if isinstance(m, str):
            norm.append(normalize_token(m))
        elif isinstance(m, dict):
            name = m.get("name") or m.get("material") or ""
            norm.append(normalize_token(name))
    return [m for m in norm if m]


def extract_conditions(a: Dict[str, Any]) -> List[Tuple[str, Optional[float], str]]:
    conds = []
    params = a.get("conditions") or a.get("parameters") or []
    for p in params:
        if not isinstance(p, dict):
            continue
        name = normalize_token(str(p.get("name") or p.get("type") or ""))
        value_raw = str(p.get("value") or p.get("amount") or "")
        unit = normalize_token(str(p.get("unit") or p.get("units") or ""))

        val = None
        try:
            m = re.search(r"[\d\.]+", value_raw.replace(",", ""))
            if m:
                val = float(m.group())
        except Exception:
            val = None
        conds.append((name, val, unit))
    return conds


def material_soft_iou(
        gold_mats: List[str],
        pred_mats: List[str],
        model: SentenceTransformer,
        threshold: float = 0.75,
) -> float:
    if not gold_mats and not pred_mats:
        return 1.0
    if not gold_mats or not pred_mats:
        return 0.0

    gold_clean = [m.strip().lower() for m in gold_mats if m.strip()]
    pred_clean = [m.strip().lower() for m in pred_mats if m.strip()]
    if not gold_clean and not pred_clean:
        return 1.0
    if not gold_clean or not pred_clean:
        return 0.0

    emb_gold = model.encode(gold_clean, convert_to_tensor=True, show_progress_bar=False)
    emb_pred = model.encode(pred_clean, convert_to_tensor=True, show_progress_bar=False)
    sim_matrix = util.cos_sim(emb_gold, emb_pred).cpu().numpy()

    matched_pred = set()
    match_count = 0
    for i, sims in enumerate(sim_matrix):
        j_best = int(sims.argmax())
        if sims[j_best] >= threshold and j_best not in matched_pred:
            matched_pred.add(j_best)
            match_count += 1

    union_size = len(set(gold_clean)) + len(set(pred_clean)) - match_count
    return match_count / union_size if union_size > 0 else 1.0


def soft_iou_conditions(
        gold: List[Tuple[str, Optional[float], str]],
        pred: List[Tuple[str, Optional[float], str]],
        value_tol: float = 0.2,
) -> float:
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0

    matched_g = set()
    matched_p = set()

    for i, (gn, gv, gu) in enumerate(gold):
        for j, (pn, pv, pu) in enumerate(pred):
            if j in matched_p:
                continue
            if gn and pn and (gn in pn or pn in gn):
                if gv is not None and pv is not None and pv != 0:
                    diff = abs(gv - pv) / max(abs(pv), 1e-6)
                    if diff <= value_tol:
                        matched_g.add(i);
                        matched_p.add(j);
                        break
                else:
                    matched_g.add(i);
                    matched_p.add(j);
                    break

    inter = len(matched_g)
    union = len(gold) + len(pred) - inter
    return inter / union if union > 0 else 0.0


def compute_order_score(matched_pairs: List[Tuple[int, int]]) -> float:
    if len(matched_pairs) < 2:
        return 1.0
    cons = 0
    total = 0
    matched_pairs = sorted(matched_pairs, key=lambda x: x[0])
    for i in range(len(matched_pairs)):
        gi, pi = matched_pairs[i]
        for j in range(i + 1, len(matched_pairs)):
            gj, pj = matched_pairs[j]
            total += 1
            if (gi - gj) * (pi - pj) > 0:
                cons += 1
    return cons / total if total > 0 else 0.0


# -------------------------
# Evidence coverage
# -------------------------

def has_evidence(a: Dict[str, Any]) -> bool:
    for k in ("span_chunk", "evidence", "evidence_spans"):
        if k not in a:
            continue
        v = a.get(k)
        if v is None:
            continue
        if isinstance(v, str) and v.strip():
            return True
        if isinstance(v, list) and len(v) > 0:
            return True
        if isinstance(v, dict) and len(v) > 0:
            return True
    return False


# -------------------------
# Grounding (hallucination proxy)
# - 기존 코드의 "candidate 하나라도 threshold 미만이면 fail"을 완화
# - Methods 전체 1덩어리 대신 chunk로 쪼개서 비교
# -------------------------

def chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i + chunk_chars]
        chunks.append(chunk)
        if i + chunk_chars >= len(text):
            break
        i += max(1, chunk_chars - overlap)
    return chunks


def build_candidate_texts(action: Dict[str, Any]) -> List[str]:
    cand_texts: List[str] = []

    # 0) action 자체 텍스트 (eval_ablation_actions.py와 동일한 키 우선순위) :contentReference[oaicite:8]{index=8}
    action_text = action.get("description") or ""
    if isinstance(action_text, str) and action_text.strip():
        cand_texts.append(action_text.strip())

    # 1) materials
    mats = action.get("materials", [])
    for m in mats:
        if isinstance(m, str) and m.strip():
            cand_texts.append(m.strip())
        elif isinstance(m, dict):
            name = (m.get("name") or m.get("label") or m.get("material") or "")
            name = str(name).strip()
            if name:
                cand_texts.append(name)

    # 2) conditions / parameters
    conds = action.get("conditions", []) or action.get("parameters", [])
    for cond in conds:
        if not isinstance(cond, dict):
            continue
        parts = []
        for key in ("name", "type", "value", "unit", "units"):
            v = cond.get(key)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
        if parts:
            cand_texts.append(" ".join(parts))

    return cand_texts


def is_action_grounded_any(
        action: Dict[str, Any],
        methods_chunks: List[str],
        model: SentenceTransformer,
        threshold: float = GROUNDING_THRESH,
) -> Optional[bool]:
    """
    grounded 판정:
    - action에서 만든 candidate 텍스트 중 '하나라도'
      methods chunk 중 '어느 하나라도'와 cosine >= threshold면 grounded(True)
    - candidate가 없으면 None (분모에서 제외)
    - methods_chunks가 없으면 False (보수적)
    """
    cand_texts = build_candidate_texts(action)
    if not cand_texts:
        return None

    if not methods_chunks:
        return False

    # 임베딩
    cand_emb = model.encode(cand_texts, convert_to_tensor=True, show_progress_bar=False)
    chunk_emb = model.encode(methods_chunks, convert_to_tensor=True, show_progress_bar=False)

    # (N_cand, N_chunk)
    sim = util.cos_sim(cand_emb, chunk_emb).cpu().numpy()
    # cand 하나라도, chunk 하나라도 threshold 넘으면 OK
    return bool((sim >= threshold).any())


def compute_grounding_hallucination_rate(
        methods_text: str,
        actions: List[Dict[str, Any]],
        model: SentenceTransformer,
) -> float:
    methods_chunks = chunk_text(methods_text, METHODS_CHUNK_CHARS, METHODS_CHUNK_OVERLAP)

    total = 0
    ungrounded = 0
    for act in actions:
        grounded = is_action_grounded_any(act, methods_chunks, model)
        if grounded is None:
            continue
        total += 1
        if grounded is False:
            ungrounded += 1
    if total == 0:
        return 0.5  # 판정 불가가 많으면 불확실로 둠
    return ungrounded / total


# -------------------------
# protocol-level 평가 (gold vs gen)
# -------------------------

def evaluate_protocol(
        model: SentenceTransformer,
        gold_rec: Dict[str, Any],
        gen_rec: Dict[str, Any],
        methods_text: str,
) -> Dict[str, Any]:
    pid = gold_rec["protocol_id"]
    gold_actions = gold_rec["actions"]
    gen_actions = gen_rec.get("actions", [])

    gold_texts = [get_action_text(a) for a in gold_actions]
    gen_texts = [get_action_text(a) for a in gen_actions]

    if not gold_texts or not gen_texts:
        # 그래도 evidence/grounding 같은 건 계산 가능
        ev_rate = float(np.mean([has_evidence(a) for a in gen_actions])) if gen_actions else 0.0
        halluc = compute_grounding_hallucination_rate(methods_text, gen_actions, model) if gen_actions else 0.0
        return {
            "protocol_id": pid,
            "n_gold": len(gold_texts),
            "n_pred": len(gen_texts),
            "step_precision": 0.0,
            "step_recall": 0.0,
            "step_f1": 0.0,
            "order_score": 0.0,
            "mat_iou": 0.0,
            "cond_iou": 0.0,
            "grounding_hallucination_rate": halluc,
            "evidence_coverage": ev_rate,
        }

    gold_emb = model.encode(gold_texts, convert_to_tensor=True, show_progress_bar=False)
    gen_emb = model.encode(gen_texts, convert_to_tensor=True, show_progress_bar=False)

    sim = util.cos_sim(gold_emb, gen_emb).cpu().numpy()
    cost = -sim
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_pairs = []
    tp = 0
    for gi, pj in zip(row_ind, col_ind):
        if sim[gi, pj] >= STEP_MATCH_THRESH:
            matched_pairs.append((gi, pj))
            tp += 1

    n_gold = len(gold_actions)
    n_pred = len(gen_actions)

    prec = tp / n_pred if n_pred > 0 else 0.0
    rec = tp / n_gold if n_gold > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0

    order_score = compute_order_score(matched_pairs)

    mat_ious = []
    cond_ious = []
    for gi, pj in matched_pairs:
        g = gold_actions[gi]
        p = gen_actions[pj]
        mat_ious.append(material_soft_iou(extract_materials(g), extract_materials(p), model, threshold=0.7))
        cond_ious.append(soft_iou_conditions(extract_conditions(g), extract_conditions(p)))

    mat_iou = float(np.mean(mat_ious)) if mat_ious else 0.0
    cond_iou = float(np.mean(cond_ious)) if cond_ious else 0.0

    # grounding hallucination rate (methods 기반)
    halluc = compute_grounding_hallucination_rate(methods_text, gen_actions, model)

    # evidence coverage
    ev_rate = float(np.mean([has_evidence(a) for a in gen_actions])) if gen_actions else 0.0

    return {
        "protocol_id": pid,
        "n_gold": n_gold,
        "n_pred": n_pred,
        "step_precision": prec,
        "step_recall": rec,
        "step_f1": f1,
        "order_score": order_score,
        "mat_iou": mat_iou,
        "cond_iou": cond_iou,
        "grounding_hallucination_rate": halluc,
        "evidence_coverage": ev_rate,
    }


# -------------------------
# main: ablation 폴더 자동 탐색 (P1~P6 모두)
# -------------------------

def main():
    print("[eval_v2] Loading SciBERT...")
    model = SentenceTransformer("allenai/scibert_scivocab_uncased")

    print("[eval_v2] Loading gold actions...")
    gold_map = build_protocol_map(load_jsonl(GOLD_ACTIONS_PATH))

    print("[eval_v2] Loading methods (gold_pairs_testset_v2.jsonl)...")
    methods_map = load_methods_map(GOLD_PAIRS_PATH)

    gen_paths = sorted(ABLATION_DIR.glob("gen_actions_P*_10.jsonl"))
    if not gen_paths:
        raise FileNotFoundError(f"No gen_actions_P*_10.jsonl found in {ABLATION_DIR}")

    rows = []
    for gen_path in gen_paths:
        mode = gen_path.stem.replace("gen_actions_", "").replace("_10", "")  # P1..P6
        print(f"\n=== Evaluating {mode} | {gen_path.name} ===")

        gen_map = build_protocol_map(load_jsonl(gen_path))

        # 공통 protocol만 평가 (from-scratch라 누락 가능성도 고려)
        common_pids = sorted(set(gold_map.keys()) & set(gen_map.keys()))
        missing = sorted(set(gold_map.keys()) - set(gen_map.keys()))
        if missing:
            print(f"[WARN] {mode}: missing {len(missing)} protocols (will skip).")

        mode_rows = []
        for pid in common_pids:
            gold_rec = gold_map[pid]
            gen_rec = gen_map[pid]
            methods_text = methods_map.get(pid, "")
            res = evaluate_protocol(model, gold_rec, gen_rec, methods_text)
            res["mode"] = mode
            res["has_methods_text"] = bool(methods_text.strip())
            mode_rows.append(res)
            rows.append(res)

        df_mode = pd.DataFrame(mode_rows)
        print(df_mode[[
            "n_pred", "step_precision", "step_recall", "step_f1",
            "order_score", "mat_iou", "cond_iou",
            "grounding_hallucination_rate", "evidence_coverage"
        ]].mean(numeric_only=True))

    df = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / "data" / "ablation_eval_actions_v2.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✅ Saved all results to {out_path}")

    # 모드별 평균 요약도 같이 저장
    df_summary = df.groupby("mode").agg({
        "n_pred": ["mean", "std"],
        "step_precision": "mean",
        "step_recall": "mean",
        "step_f1": "mean",
        "order_score": "mean",
        "mat_iou": "mean",
        "cond_iou": "mean",
        "grounding_hallucination_rate": "mean",
        "evidence_coverage": "mean",
        "has_methods_text": "mean",
    })
    out_sum = PROJECT_ROOT / "data" / "ablation_eval_actions_v2_summary.csv"
    df_summary.to_csv(out_sum)
    print(f"✅ Saved summary to {out_sum}")


if __name__ == "__main__":
    main()
