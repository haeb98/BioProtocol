# scripts/eval_action_soft_iou_10.py

import csv
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer

# ---- 프로젝트 루트 sys.path 추가 ----
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

# ===== SciBERT 임베딩 모델 로드 =====
print("[eval] Loading SciBERT embedder...")
embedder = SentenceTransformer("allenai/scibert_scivocab_uncased")


def embed_texts(texts: List[str]) -> np.ndarray:
    """문자열 리스트를 SciBERT로 임베딩 (L2-normalize)"""
    if not texts:
        return np.zeros((0, 768))
    return embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def soft_iou(gold_items: List[str], pred_items: List[str], thresh: float = 0.7) -> float:
    """
    Throt 스타일 IoU에 가까운 soft IoU:
    - gold, pred는 문자열 리스트
    - semantic 유사도가 thresh 이상인 gold 항목은 intersection에 포함
    """
    if not gold_items and not pred_items:
        return 1.0  # 둘 다 비어있으면 IoU = 1로 취급 (취향 차이, 여기선 1)
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


def load_jsonl_map(path: Path, key_field: str) -> Dict[str, Any]:
    """jsonl 파일을 key_field 기준으로 dict로 로드"""
    data = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = rec[key_field]
            data[key] = rec
    return data


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
    gen_steps_B.* 파일을 유연하게 읽어오는 함수.
    - JSON 배열 형식: [ {...}, {...}, ... ]
    - JSONL 형식: {..}\n{..}\n...
    둘 다 지원.
    """
    print(f"[eval] Loading step-level guess from {path}...")
    text = path.read_text().strip()
    step_guess_map: Dict[str, List[Dict[str, Any]]] = {}

    if not text:
        print("[eval] WARNING: step guess file is empty.")
        return step_guess_map

    # JSON array 형식인지 먼저 시도
    if text[0] == "[":
        try:
            records = json.loads(text)
            if isinstance(records, dict):
                records = [records]
        except json.JSONDecodeError:
            records = []
    else:
        # JSONL 형식 가정
        records = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError:
                continue

    for rec in records:
        pid = rec.get("protocol_id")
        if not pid:
            continue
        step_guess_map.setdefault(pid, []).append(rec)

    print(f"[eval] Loaded step-level guess for {len(step_guess_map)} protocols.")
    return step_guess_map


def get_action_repr(act: Dict[str, Any]) -> str:
    """액션간 유사도 계산에 사용할 대표 텍스트 생성"""
    # 스키마에 따라 필드 이름이 다를 수 있으니 필요에 따라 수정
    for k in ["description", "action_text", "step_text"]:
        if k in act and act[k]:
            return act[k]

    # 없으면 materials/conditions를 이어붙이기
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
    """step guess (gen_steps_B)에서 액션 매칭용 표현"""
    # gen_steps_B.jsonl의 스키마에 맞게 수정 필요
    if "step_rationale" in step and step["step_rationale"]:
        return step["step_rationale"]
    if "span_text" in step and step["span_text"]:
        return step["span_text"]
    return json.dumps(step, ensure_ascii=False)[:200]


def extract_material_names(actions: List[Dict[str, Any]]) -> List[str]:
    mats = []
    for act in actions:
        for m in act.get("materials", []):
            name = m.get("name", "")
            if name:
                mats.append(name)
    return mats


def extract_condition_strings(actions: List[Dict[str, Any]]) -> List[str]:
    conds = []
    for act in actions:
        for c in act.get("conditions", []):
            t = c.get("type", "")
            v = c.get("value", "")
            if t or v:
                conds.append(f"{t}: {v}".strip(": "))
    return conds


def match_actions(
        gold_actions: List[Dict[str, Any]],
        pred_actions_repr: List[str],
        action_thresh: float = 0.6,
) -> List[int]:
    """
    gold_actions를 pred_actions(혹은 step_guess)와 매칭.
    - pred_actions_repr: 각 pred action의 대표 텍스트 리스트
    - return: 각 gold index에 대해 best pred index or -1
    """
    gold_repr = [get_action_repr(a) for a in gold_actions]
    if not gold_repr or not pred_actions_repr:
        return [-1] * len(gold_actions)

    gold_emb = embed_texts(gold_repr)
    pred_emb = embed_texts(pred_actions_repr)
    sim = np.matmul(gold_emb, pred_emb.T)  # (gold, pred)

    best_pred_idx = sim.argmax(axis=1)  # 각 gold에 대해 best pred index
    best_sim = sim.max(axis=1)

    matched = []
    for i in range(len(gold_actions)):
        if best_sim[i] >= action_thresh:
            matched.append(best_pred_idx[i])
        else:
            matched.append(-1)
    return matched


def main():
    GOLD_ACTIONS_PATH = Path("data/gold_actions_ir_10.jsonl")
    GEN_ACTIONS_PATH = Path("data/gen_actions_ir_10.jsonl")
    STEP_GUESS_PATH = Path("data/gen_steps_B_fixed.jsonl")
    PAIRS_PATH = Path("data/gold_pairs_testset_v2.jsonl")
    CSV_OUT = Path("data/eval_actions_soft_iou_10_re.csv")

    # Domain 정보 로드
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
                domain_map[pid] = domain

    print("[eval] Loading gold/gen actions...")
    gold_map = load_actions_jsonl(GOLD_ACTIONS_PATH)
    gen_map = load_actions_jsonl(GEN_ACTIONS_PATH)
    step_guess_map = load_step_guess(STEP_GUESS_PATH)

    results = []

    for pid, gold_actions in gold_map.items():
        gen_actions = gen_map.get(pid, [])
        step_guesses = step_guess_map.get(pid, [])

        # 액션/스텝 representation
        gen_repr = [get_action_repr(a) for a in gen_actions]
        step_repr = [get_step_repr(s) for s in step_guesses]

        # gold → gen 액션 매칭
        matched_gen_idx = match_actions(gold_actions, gen_repr, action_thresh=0.6)
        # gold → step_guess 매칭
        matched_step_idx = match_actions(gold_actions, step_repr, action_thresh=0.6)

        # per-action IoU 계산
        mat_ious_gen = []
        mat_ious_guess = []
        cond_ious_gen = []
        cond_ious_guess = []

        for g_idx, gold_act in enumerate(gold_actions):
            # gold action에서 materials/conditions 추출
            gold_mats = [m.get("name", "") for m in gold_act.get("materials", []) if m.get("name")]
            gold_conds = []
            for c in gold_act.get("conditions", []):
                t = c.get("type", "")
                v = c.get("value", "")
                if t or v:
                    gold_conds.append(f"{t}: {v}".strip(": "))

            # gold action에 materials/conditions가 전혀 없으면 IoU 평균에서 제외하는 것도 가능
            # 여기선 포함하되, union=0이면 IoU=1로 처리 (soft_iou 함수 정의 참고)

            # --- vs gen_actions ---
            if matched_gen_idx[g_idx] >= 0 and gen_actions:
                pred_act = gen_actions[matched_gen_idx[g_idx]]
                pred_mats = [m.get("name", "") for m in pred_act.get("materials", []) if m.get("name")]
                pred_conds = []
                for c in pred_act.get("conditions", []):
                    t = c.get("type", "")
                    v = c.get("value", "")
                    if t or v:
                        pred_conds.append(f"{t}: {v}".strip(": "))
            else:
                pred_mats = []
                pred_conds = []

            mat_ious_gen.append(soft_iou(gold_mats, pred_mats, thresh=0.7))
            cond_ious_gen.append(soft_iou(gold_conds, pred_conds, thresh=0.7))

            # --- vs step_guess (materials_llm_guess / parameters_llm_guess) ---
            if matched_step_idx[g_idx] >= 0 and step_guesses:
                step = step_guesses[matched_step_idx[g_idx]]
                guess_mats = []
                for x in step.get("materials_llm_guess", []):
                    if isinstance(x, str):
                        guess_mats.append(x)

                guess_conds = []
                for x in step.get("parameters_llm_guess", []):
                    if isinstance(x, dict):
                        guess_conds.extend([f"{k}: {v}" for k, v in x.items()])
                    elif isinstance(x, str):
                        guess_conds.append(x)
            else:
                guess_mats = []
                guess_conds = []

            mat_ious_guess.append(soft_iou(gold_mats, guess_mats, thresh=0.7))
            cond_ious_guess.append(soft_iou(gold_conds, guess_conds, thresh=0.7))

        # 프로토콜 단위 평균 IoU
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

    # 콘솔 출력
    print("\n=== Per-Protocol Soft IoU (Action-level) ===")
    for r in results:
        print(r)

    # 평균
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
