# scripts/eval_action_soft_iou_10.py

import csv
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from scipy.optimize import linear_sum_assignment  # Hungarian 알고리즘
from sentence_transformers import SentenceTransformer

# ---- 프로젝트 루트 sys.path 추가 ----
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

# ===== 설정값 =====
ACTION_THRESH = 0.7  # 액션 매칭용 similarity threshold
ITEM_THRESH = 0.7  # material / condition soft IoU용 threshold

# ===== SciBERT 임베딩 모델 로드 =====
print("[eval] Loading SciBERT embedder...")
embedder = SentenceTransformer("allenai/scibert_scivocab_uncased")


def embed_texts(texts: List[str]) -> np.ndarray:
    """문자열 리스트를 SciBERT로 임베딩 (L2-normalize)"""
    if not texts:
        return np.zeros((0, 768))
    return embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def soft_iou(gold_items: List[str], pred_items: List[str], thresh: float = ITEM_THRESH) -> float:
    """
    Throt 스타일 IoU에 가까운 soft IoU:
    - gold, pred는 문자열 리스트
    - semantic 유사도가 thresh 이상인 gold 항목은 intersection에 포함
    """
    if not gold_items and not pred_items:
        return 1.0  # 둘 다 비어있으면 IoU = 1로 취급 (여기선 coverage 문제 아님)
    if not gold_items or not pred_items:
        return 0.0

    gold_emb = embed_texts(gold_items)
    pred_emb = embed_texts(pred_items)
    if len(gold_emb) == 0 or len(pred_emb) == 0:
        return 0.0

    sim = np.matmul(gold_emb, pred_emb.T)  # (gold, pred)
    max_sim = sim.max(axis=1)  # 각 gold가 가장 유사한 pred와의 sim
    covered = (max_sim >= thresh)
    intersection = covered.sum()
    union = len(gold_items) + len(pred_items) - intersection
    if union == 0:
        return 1.0
    return intersection / union


def load_actions_jsonl(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """gold_actions_ir_10.jsonl / gen_actions_ir_10.jsonl 로더"""
    mp: Dict[str, List[Dict[str, Any]]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec["protocol_id"]
            mp[pid] = rec["actions"]
    return mp


def load_step_guess(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    gen_steps_B.jsonl 로더
    각 라인: {"protocol_id": "...", "steps": [ {...}, {...}, ... ]}
    """
    print(f"[eval] Loading step-level guess from {path}...")
    step_guess_map: Dict[str, List[Dict[str, Any]]] = {}

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get("protocol_id")
            steps = rec.get("steps", [])
            if not pid:
                continue
            step_guess_map[pid] = steps

    print(f"[eval] Loaded step-level guess for {len(step_guess_map)} protocols.")
    return step_guess_map


def get_action_repr(act: Dict[str, Any]) -> str:
    """액션간 유사도 계산에 사용할 대표 텍스트 생성"""
    for k in ["description", "action_text", "step_text", "span_chunk"]:
        if k in act and act[k]:
            return act[k]

    mats = []
    for m in act.get("materials", []):
        name = m.get("name", "")
        if name:
            mats.append(name)
    conds = []
    for c in act.get("conditions", []):
        t = c.get("type", "")
        v = c.get("value", "")
        if t or v:
            conds.append(f"{t}: {v}".strip(": "))

    return " | ".join(mats + conds)


def get_step_repr(step: Dict[str, Any]) -> str:
    """step guess (gen_steps_B)에서 액션 매칭용 대표 텍스트"""
    if "step_text" in step and step["step_text"]:
        return step["step_text"]
    if "span_chunk" in step and step["span_chunk"]:
        return step["span_chunk"]
    return json.dumps(step, ensure_ascii=False)[:200]


def extract_material_names(act: Dict[str, Any]) -> List[str]:
    return [m.get("name", "") for m in act.get("materials", []) if m.get("name")]


def extract_condition_strings(act: Dict[str, Any]) -> List[str]:
    conds = []
    for c in act.get("conditions", []):
        t = c.get("type", "")
        v = c.get("value", "")
        if t or v:
            conds.append(f"{t}: {v}".strip(": "))
    return conds


def extract_guess_materials(step: Dict[str, Any]) -> List[str]:
    out = []
    for x in step.get("materials_llm_guess", []):
        if isinstance(x, str):
            out.append(x)
    return out


def extract_guess_conditions(step: Dict[str, Any]) -> List[str]:
    out = []
    for x in step.get("parameters_llm_guess", []):
        if isinstance(x, dict):
            out.extend([f"{k}: {v}" for k, v in x.items()])
        elif isinstance(x, str):
            out.append(x)
    return out


def hungarian_match(gold_repr: List[str],
                    pred_repr: List[str],
                    action_thresh: float = ACTION_THRESH) -> List[int]:
    """
    Hungarian 기반 1:1 매칭:
    - gold_repr: gold action들의 대표 텍스트
    - pred_repr: pred action / step들의 대표 텍스트
    - return: len(gold_repr) 길이 리스트, 각 gold index → 매칭된 pred index (또는 -1)
    """
    n_gold = len(gold_repr)
    n_pred = len(pred_repr)
    if n_gold == 0 or n_pred == 0:
        return [-1] * n_gold

    gold_emb = embed_texts(gold_repr)
    pred_emb = embed_texts(pred_repr)
    sim = np.matmul(gold_emb, pred_emb.T)  # (gold, pred)

    # Hungarian은 최소 cost를 찾으므로, cost = 1 - sim (sim 높을수록 cost 작음)
    cost = 1.0 - sim
    row_ind, col_ind = linear_sum_assignment(cost)  # 1:1 매칭

    matched = [-1] * n_gold
    for gi, pj in zip(row_ind, col_ind):
        if sim[gi, pj] >= action_thresh:
            matched[gi] = pj
        else:
            matched[gi] = -1

    # Hungarian은 gold / pred 중 더 작은 쪽만 100% 소화
    # gold가 더 많은 경우, 나머지는 여전히 -1 상태
    return matched


def main():
    GOLD_ACTIONS_PATH = Path("data/gold_actions_ir_10.jsonl")
    GEN_ACTIONS_PATH = Path("data/gen_actions_ir_10.jsonl")
    STEP_GUESS_PATH = Path("data/gen_steps_B_fixed.jsonl")
    PAIRS_PATH = Path("data/gold_pairs_testset_v2.jsonl")
    CSV_OUT = Path("data/eval_actions_soft_iou_10_1to1.csv")

    # Domain 정보 로드 (있으면)
    domain_map: Dict[str, str] = {}
    if PAIRS_PATH.exists():
        with PAIRS_PATH.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                pid = rec.get("protocol_id")
                bio = rec.get("bio", {})
                cls = bio.get("classification", {})
                domain = cls.get("primary_domain", "")
                if pid:
                    domain_map[pid] = domain

    print("[eval] Loading gold/gen actions...")
    gold_map = load_actions_jsonl(GOLD_ACTIONS_PATH)
    gen_map = load_actions_jsonl(GEN_ACTIONS_PATH)
    step_guess_map = load_step_guess(STEP_GUESS_PATH)

    results = []

    for pid, gold_actions in gold_map.items():
        gen_actions = gen_map.get(pid, [])
        step_guesses = step_guess_map.get(pid, [])

        gold_repr = [get_action_repr(a) for a in gold_actions]
        gen_repr = [get_action_repr(a) for a in gen_actions]
        step_repr = [get_step_repr(s) for s in step_guesses]

        # gold ↔ gen_actions Hungarian 1:1 매칭
        matched_gen_idx = hungarian_match(gold_repr, gen_repr, action_thresh=ACTION_THRESH)
        # gold ↔ step_guess Hungarian 1:1 매칭
        matched_step_idx = hungarian_match(gold_repr, step_repr, action_thresh=ACTION_THRESH)

        mat_ious_gen = []
        mat_ious_guess = []
        cond_ious_gen = []
        cond_ious_guess = []

        for g_idx, gold_act in enumerate(gold_actions):
            # gold side
            gold_mats = extract_material_names(gold_act)
            gold_conds = extract_condition_strings(gold_act)

            # ---- vs gen_actions ----
            if matched_gen_idx[g_idx] >= 0 and gen_actions:
                pred_act = gen_actions[matched_gen_idx[g_idx]]
                pred_mats = extract_material_names(pred_act)
                pred_conds = extract_condition_strings(pred_act)
            else:
                pred_mats = []
                pred_conds = []

            mat_ious_gen.append(soft_iou(gold_mats, pred_mats, thresh=ITEM_THRESH))
            cond_ious_gen.append(soft_iou(gold_conds, pred_conds, thresh=ITEM_THRESH))

            # ---- vs step_guess (baseline) ----
            if matched_step_idx[g_idx] >= 0 and step_guesses:
                step = step_guesses[matched_step_idx[g_idx]]
                guess_mats = extract_guess_materials(step)
                guess_conds = extract_guess_conditions(step)
            else:
                guess_mats = []
                guess_conds = []

            mat_ious_guess.append(soft_iou(gold_mats, guess_mats, thresh=ITEM_THRESH))
            cond_ious_guess.append(soft_iou(gold_conds, guess_conds, thresh=ITEM_THRESH))

        mat_iou_gen = float(np.mean(mat_ious_gen)) if mat_ious_gen else 0.0
        mat_iou_guess = float(np.mean(mat_ious_guess)) if mat_ious_guess else 0.0
        cond_iou_gen = float(np.mean(cond_ious_gen)) if cond_ious_gen else 0.0
        cond_iou_guess = float(np.mean(cond_ious_guess)) if cond_ious_guess else 0.0

        matched_actions_gen = sum(1 for idx in matched_gen_idx if idx >= 0)
        matched_actions_guess = sum(1 for idx in matched_step_idx if idx >= 0)

        res = {
            "protocol_id": pid,
            "domain": domain_map.get(pid, ""),
            "gold_actions": len(gold_actions),
            "gen_actions": len(gen_actions),
            "matched_actions_gen": matched_actions_gen,
            "matched_actions_guess": matched_actions_guess,
            "mat_iou_gen": mat_iou_gen,
            "mat_iou_guess": mat_iou_guess,
            "cond_iou_gen": cond_iou_gen,
            "cond_iou_guess": cond_iou_guess,
        }
        results.append(res)

    print("\n=== Per-Protocol Soft IoU (Action-level, Hungarian) ===")
    for r in results:
        print(r)

    avg_mat_gen = np.mean([r["mat_iou_gen"] for r in results])
    avg_mat_guess = np.mean([r["mat_iou_guess"] for r in results])
    avg_cond_gen = np.mean([r["cond_iou_gen"] for r in results])
    avg_cond_guess = np.mean([r["cond_iou_guess"] for r in results])

    print("\n=== Averages ===")
    print(f"Materials IoU (gen actions) : {avg_mat_gen:.4f}")
    print(f"Materials IoU (step guess)  : {avg_mat_guess:.4f}")
    print(f"Conditions IoU (gen actions): {avg_cond_gen:.4f}")
    print(f"Conditions IoU (step guess) : {avg_cond_guess:.4f}")

    # CSV 저장
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "protocol_id",
            "domain",
            "gold_actions",
            "gen_actions",
            "matched_actions_gen",
            "matched_actions_guess",
            "mat_iou_gen",
            "mat_iou_guess",
            "cond_iou_gen",
            "cond_iou_guess",
        ])
        for r in results:
            writer.writerow([
                r["protocol_id"],
                r["domain"],
                r["gold_actions"],
                r["gen_actions"],
                r["matched_actions_gen"],
                r["matched_actions_guess"],
                f"{r['mat_iou_gen']:.4f}",
                f"{r['mat_iou_guess']:.4f}",
                f"{r['cond_iou_gen']:.4f}",
                f"{r['cond_iou_guess']:.4f}",
            ])

    print(f"\n[eval] CSV saved to {CSV_OUT}")


if __name__ == "__main__":
    main()
