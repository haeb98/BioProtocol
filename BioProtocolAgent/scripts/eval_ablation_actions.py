# scripts/eval_ablation_actions.py
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer, util

# 프로젝트 루트 등록
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

GOLD_PATH = PROJECT_ROOT / "data" / "gold_actions_ir_10.jsonl"
GEN_PATTERNS = {
    "P1": PROJECT_ROOT / "data" / "ablation" / "gen_actions_P1_10.jsonl",
    "P2": PROJECT_ROOT / "data" / "ablation" / "gen_actions_P2_10.jsonl",
    "P3": PROJECT_ROOT / "data" / "ablation" / "gen_actions_P3_10.jsonl",
    "P4": PROJECT_ROOT / "data" / "ablation" / "gen_actions_P4_10.jsonl",
    "P5": PROJECT_ROOT / "data" / "ablation" / "gen_actions_P5_10.jsonl",
    "P6": PROJECT_ROOT / "data" / "ablation" / "gen_actions_P6_10.jsonl",
}

THRESH = 0.7  # SciBERT cosine threshold


# ---------- 유틸 ----------

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
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
        pid = r["protocol_id"]
        out[pid] = r
    return out


def get_action_text(a: Dict[str, Any]) -> str:
    return (
            "action:" + (a.get("action") or "None") + ", description:" + (a.get("description") or "None")
    )


def normalize_token(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace()).strip()


def extract_materials(a: Dict[str, Any]) -> List[str]:
    mats = a.get("materials") or []
    # materials 가 문자열 리스트거나 dict 리스트일 수 있음
    norm = []
    for m in mats:
        if isinstance(m, str):
            norm.append(normalize_token(m))
        elif isinstance(m, dict):
            name = m.get("name") or m.get("material") or ""
            norm.append(normalize_token(name))
    return [m for m in norm if m]


def extract_conditions(a: Dict[str, Any]) -> List[Tuple[str, float, str]]:
    """
    (name, value, unit) triple 리스트로 평탄화.
    value 가 숫자 변환 안되면 None 로 두고 이름 기반 비교만 사용.
    """
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
            # "14,000 g" 같은 건 숫자만 뽑기
            import re
            m = re.search(r"[\d\.]+", value_raw.replace(",", ""))
            if m:
                val = float(m.group())
        except Exception:
            val = None

        conds.append((name, val, unit))
    return conds


def soft_iou_materials(gold: List[str], pred: List[str]) -> float:
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    gold_set = set(gold)
    pred_set = set(pred)
    inter = len(gold_set & pred_set)
    union = len(gold_set | pred_set)
    return inter / union if union > 0 else 0.0


def material_soft_iou(
        gold_mats: list[str],
        pred_mats: list[str],
        model: SentenceTransformer,
        threshold: float = 0.75,
) -> float:
    """
    SciBERT 임베딩 기반 Material Soft-IoU.
    - gold_mats, pred_mats: 문자열 리스트
    - threshold 이상 유사도면 '동일 material'로 간주
    - greedy matching으로 1:1 매칭 후 IoU 계산
    """
    # 둘 다 비어 있으면 perfect
    if not gold_mats and not pred_mats:
        return 1.0
    if not gold_mats or not pred_mats:
        return 0.0

    # 전처리: 소문자/strip 정도만 (너무 aggressive 하게 normalize 하지는 않음)
    gold_clean = [m.strip().lower() for m in gold_mats if m.strip()]
    pred_clean = [m.strip().lower() for m in pred_mats if m.strip()]

    if not gold_clean and not pred_clean:
        return 1.0
    if not gold_clean or not pred_clean:
        return 0.0

    # SciBERT 임베딩 계산
    emb_gold = model.encode(gold_clean, convert_to_tensor=True)
    emb_pred = model.encode(pred_clean, convert_to_tensor=True)

    # cosine similarity matrix: [len(gold), len(pred)]
    sim_matrix = util.cos_sim(emb_gold, emb_pred).cpu().numpy()

    matched_pred = set()
    match_count = 0

    # 간단 greedy: 각 gold에 대해 가장 유사한 pred 하나 선택 (threshold 이상 & 아직 안쓴 것만)
    for i, sims in enumerate(sim_matrix):
        j_best = sims.argmax()
        if sims[j_best] >= threshold and j_best not in matched_pred:
            matched_pred.add(j_best)
            match_count += 1

    # IoU 계산: gold/pred 집합의 합집합 대비 매칭 개수
    union_size = len(set(gold_clean)) + len(set(pred_clean)) - match_count
    if union_size == 0:
        return 1.0
    return match_count / union_size


def soft_iou_conditions(
        gold: List[Tuple[str, float, str]],
        pred: List[Tuple[str, float, str]],
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
                # 이름이 충분히 비슷하고,
                if gv is not None and pv is not None:
                    # 값도 허용 오차 이내이면 match
                    if pv == 0:
                        continue
                    diff = abs(gv - pv) / max(abs(pv), 1e-6)
                    if diff <= value_tol:
                        matched_g.add(i)
                        matched_p.add(j)
                        break
                else:
                    # 숫자 없으면 이름만으로 match 인정
                    matched_g.add(i)
                    matched_p.add(j)
                    break

    inter = len(matched_g)
    union = len(gold) + len(pred) - inter
    return inter / union if union > 0 else 0.0


def compute_order_score(matched_pairs: List[Tuple[int, int]]) -> float:
    """
    matched_pairs: [(gold_idx, pred_idx), ...]
    """
    if len(matched_pairs) < 2:
        return 1.0
    cons = 0
    total = 0
    matched_pairs = sorted(matched_pairs, key=lambda x: x[0])  # gold 기준 정렬
    for i in range(len(matched_pairs)):
        gi, pi = matched_pairs[i]
        for j in range(i + 1, len(matched_pairs)):
            gj, pj = matched_pairs[j]
            total += 1
            if (gi - gj) * (pi - pj) > 0:
                cons += 1
    return cons / total if total > 0 else 0.0


def build_candidate_texts(action: Dict[str, Any]) -> List[str]:
    """
    한 action 에서 Methods 와 비교할 candidate 텍스트들을 모은다.
    - action_text / step_text / raw_text
    - materials 이름
    - conditions / parameters 의 (이름 + 값 + 단위)
    """
    cand_texts: List[str] = []

    # 0) action 자체 텍스트
    action_text = (
            action.get("action_text")
            or action.get("step_text")
            or action.get("raw_text")
            or ""
    )
    if action_text.strip():
        cand_texts.append(action_text.strip())

    # 1) materials
    mats = action.get("materials", [])
    for m in mats:
        if isinstance(m, str):
            if m.strip():
                cand_texts.append(m.strip())
        elif isinstance(m, dict):
            name = (
                    m.get("name")
                    or m.get("label")
                    or m.get("material")
                    or ""
            )
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


def is_action_grounded_by_scibert(
        action: dict,
        methods_text: str,
        model: SentenceTransformer,
        threshold: float = 0.6,
) -> bool:
    """
    한 액션의 텍스트(액션 문장 + materials + conditions)가
    Methods '전체 텍스트'와 의미적으로 align 되는지 확인.

    - Methods 전체를 하나의 embedding으로 보고,
    - 각 candidate 텍스트가 이 벡터와 cosine similarity >= threshold 면 grounded로 간주
    - 하나라도 threshold 미만이면, 해당 action 은 hallucinated 로 본다.
    """
    if not methods_text or not methods_text.strip():
        # 근거 텍스트 자체가 없으면, 보수적으로 hallucinated 처리
        return False

    # Methods 전체를 하나의 chunk로 임베딩
    methods_emb = model.encode([methods_text], convert_to_tensor=True)  # shape: (1, D)

    # 액션에서 확인할 텍스트 후보들 모으기
    cand_texts = build_candidate_texts(action)

    # 확인할 candidate가 전혀 없으면, 여기서는 "판정 불가"
    # -> False 대신 None 을 쓰고, rate 계산쪽에서 제외하는 편이 안전
    if not cand_texts:
        return None  # type: ignore

    # candidate 들 임베딩
    cand_emb = model.encode(cand_texts, convert_to_tensor=True)  # shape: (N, D)

    # cand vs methods 전체 (1개) 사이 cosine sim → (N, 1)
    sim_matrix = util.cos_sim(cand_emb, methods_emb).cpu().numpy()
    max_sims = sim_matrix[:, 0]  # 각 candidate 별 similarity

    # 모든 candidate 가 어느 정도 methods 전체와 유사해야 'grounded'
    # 하나라도 threshold 미만이면 hallucinated 로 판단
    if np.any(max_sims < threshold):
        return False
    return True


def compute_hallucination_rate_for_mode(
        methods_text: str,
        actions: List[Dict[str, Any]],
        model: SentenceTransformer,
        threshold: float = 0.6,
) -> float:
    """
    한 프로토콜에 대해 action-level hallucination rate 계산.

    - 각 action 에 대해 is_action_grounded_by_scibert(...) 호출
    - grounded == False 인 action 의 비율을 hallucination_rate 로 보고,
      0.0 은 "모든 action 이 근거 있음", 1.0 은 "모든 action 이 근거 없음" 을 의미.
    - candidate 텍스트가 전혀 없는 action은 '판정 불가'로 보고 분모에서 제외.
    """
    total_actions = 0
    hallucinated_actions = 0

    for act in actions:
        grounded = is_action_grounded_by_scibert(
            act,
            methods_text,
            model,
            threshold=threshold,
        )

        if grounded is None:
            # cand_texts 가 없어서 판정 불가인 action → rate 계산에서 제외
            continue

        total_actions += 1
        if not grounded:
            hallucinated_actions += 1

    if total_actions == 0:
        # 평가 가능한 action 이 하나도 없으면 0.5 정도로 둘 수도 있고,
        # 0.0 으로 둘 수도 있음. 여기서는 0.5(불확실) 선택.
        return 0.5
    return hallucinated_actions / total_actions


# ---------- Action-level 평가 (한 프로토콜) ----------

def evaluate_protocol(
        model: SentenceTransformer,
        gold_rec: Dict[str, Any],
        gen_rec: Dict[str, Any],
) -> Dict[str, Any]:
    pid = gold_rec["protocol_id"]
    gold_actions = gold_rec["actions"]
    gen_actions = gen_rec.get("actions", [])

    # SciBERT 임베딩
    gold_texts = [get_action_text(a) for a in gold_actions]
    gen_texts = [get_action_text(a) for a in gen_actions]

    if not gold_texts or not gen_texts:
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
            "hallucination_rate": 0.0,
        }

    gold_emb = model.encode(gold_texts, convert_to_tensor=True, show_progress_bar=False)
    gen_emb = model.encode(gen_texts, convert_to_tensor=True, show_progress_bar=False)

    sim = util.cos_sim(gold_emb, gen_emb).cpu().numpy()

    # 헝가리안 매칭 (max → min cost 변환)
    cost = -sim
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_pairs = []
    tp = 0
    for gi, pj in zip(row_ind, col_ind):
        if sim[gi, pj] >= THRESH:
            matched_pairs.append((gi, pj))
            tp += 1

    n_gold = len(gold_actions)
    n_pred = len(gen_actions)

    fp = n_pred - tp
    fn = n_gold - tp

    prec = tp / n_pred if n_pred > 0 else 0.0
    rec = tp / n_gold if n_gold > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0

    # Order score
    order_score = compute_order_score(matched_pairs)

    # Material / Condition soft IoU
    mat_ious = []
    cond_ious = []

    for gi, pj in matched_pairs:
        g = gold_actions[gi]
        p = gen_actions[pj]
        g_m = extract_materials(g)
        p_m = extract_materials(p)
        g_c = extract_conditions(g)
        p_c = extract_conditions(p)

        mat_ious.append(material_soft_iou(g_m, p_m, model, threshold=0.7))
        cond_ious.append(soft_iou_conditions(g_c, p_c))

    mat_iou = float(np.mean(mat_ious)) if mat_ious else 0.0
    cond_iou = float(np.mean(cond_ious)) if cond_ious else 0.0

    # Hallucination rate (간단 버전)
    methods_text = gold_rec.get("methods_text", "")
    halluc = compute_hallucination_rate_for_mode(
        methods_text,
        gen_actions,
        model,
        threshold=0.6,
    )

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
        "hallucination_rate": halluc,
    }


# ---------- 메인 ----------

def main():
    print("[eval] Loading SciBERT...")
    model = SentenceTransformer("allenai/scibert_scivocab_uncased")

    print("[eval] Loading gold actions...")
    gold_recs = _load_jsonl(GOLD_PATH)
    gold_map = build_protocol_map(gold_recs)

    rows = []

    for mode, gen_path in GEN_PATTERNS.items():
        print(f"\n=== Evaluating mode {mode} ===")
        gen_recs = _load_jsonl(gen_path)
        gen_map = build_protocol_map(gen_recs)

        for pid, gold_rec in gold_map.items():
            if pid not in gen_map:
                print(f"[WARN] {pid} missing in {mode}, skipping.")
                continue
            gen_rec = gen_map[pid]
            res = evaluate_protocol(model, gold_rec, gen_rec)
            res["mode"] = mode
            rows.append(res)

        # 모드별 평균 출력
        df_mode = pd.DataFrame([r for r in rows if r["mode"] == mode])
        print(df_mode[[
            "step_precision", "step_recall", "step_f1",
            "order_score", "mat_iou", "cond_iou",
            "hallucination_rate"
        ]].mean())

    df = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / "data" / "ablation_eval_actions_all_add.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✅ Saved all results to {out_path}")


if __name__ == "__main__":
    main()
