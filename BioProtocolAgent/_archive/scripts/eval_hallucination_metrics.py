#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
멀티 에이전트 GEN 결과 vs GOLD IR를 기반으로
- Step Hallucination Rate
- Parameter Supported Rate
- Contradiction Rate
- Evidence Missing Rate
을 프로토콜별로 계산하고 CSV 리포트로 저장하는 스크립트.

실행 예시:
    cd /Users/haeb/Workspaces/BioProtocol/BioProtocolAgent
    python scripts/eval_hallucination_metrics.py \
        --gold-actions data/gold_actions_ir_10.jsonl \
        --gold-pairs data/gold_pairs_testset_v2.jsonl \
        --gen-actions data/ablation/gen_actions_P4_10.jsonl \
        --output reports/hallucination_report_P4.csv
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util


# ---------------------------
# 기본 파싱 유틸
# ---------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def index_by_protocol(records: List[Dict[str, Any]], key: str = "protocol_id") -> Dict[str, List[Dict[str, Any]]]:
    out = defaultdict(list)
    for r in records:
        pid = r.get(key)
        if pid is None:
            continue
        out[pid].append(r)
    return out


def load_sec_text_by_protocol(gold_pairs_path: Path) -> Dict[str, str]:
    """
    gold_pairs_testset_v2.jsonl 에서
    각 protocol_id -> article.sec_text (또는 article.sections["Methods"]) 를 추출
    """
    sec_map = {}
    records = load_jsonl(gold_pairs_path)
    for r in records:
        pid = r.get("protocol_id")
        article = r.get("article", {})
        sec_text = r.get("sec_text")
        if sec_text is None:
            # 혹시 구조가 article.sections["Methods"] 인 경우 fallback
            sections = article.get("sections", {})
            if isinstance(sections, dict):
                sec_text = sections.get("Methods")
        if pid and isinstance(sec_text, str):
            sec_map[pid] = sec_text
    return sec_map


# ---------------------------
# Step text / parameter / evidence 추출
# (필드명이 조금 다를 수 있으므로 여기서 통일)
# ---------------------------

def get_step_text(rec: Dict[str, Any]) -> str:
    """
    action/step 텍스트 필드 추출.
    프로젝트 스키마에 맞게 필요하면 여기 수정.
    """
    for k in ["step_text", "action_text", "text", "description"]:
        if k in rec and isinstance(rec[k], str):
            return rec[k]
    return ""


def get_parameters(rec: Dict[str, Any]) -> List[Any]:
    """
    parameters 필드 추출.
    """
    for k in ["parameters", "params", "param_list"]:
        if k in rec and isinstance(rec[k], list):
            return rec[k]
    return []


def get_evidence(rec: Dict[str, Any]) -> Optional[Any]:
    """
    evidence 관련 필드 추출.
    """
    for k in ["span_chunk", "evidence", "evidence_spans"]:
        if k in rec:
            return rec[k]
    return None


# ---------------------------
# Parameter 문자열 / name / value 추출
# ---------------------------

def param_to_string(p: Any) -> str:
    """
    sec_text substring 검색에 사용할 parameter 표현.
    dict 인 경우 value+unit 기반, 아니면 str(p).
    """
    if isinstance(p, dict):
        val = p.get("value") or p.get("val") or p.get("amount") or ""
        unit = p.get("unit") or p.get("units") or ""
        name = p.get("name") or p.get("param") or p.get("parameter") or ""
        # 우선 value+unit, 없으면 name+value 등으로 만들기
        pieces = [str(val).strip(), str(unit).strip()]
        s = " ".join([x for x in pieces if x])
        if not s:
            # fallback
            pieces = [str(name).strip(), str(val).strip(), str(unit).strip()]
            s = " ".join([x for x in pieces if x])
        return s.strip()
    else:
        return str(p).strip()


def get_param_name_and_value(p: Any) -> Tuple[Optional[str], Optional[float]]:
    """
    contradiction 판단용 name, numeric value 추출.
    숫자가 없으면 value=None.
    """
    if isinstance(p, dict):
        name = p.get("name") or p.get("param") or p.get("parameter")
        val_str = p.get("value") or p.get("val") or p.get("amount")
    else:
        name = None
        val_str = str(p)

    if isinstance(val_str, (int, float)):
        num = float(val_str)
    else:
        # 문자열에서 숫자 하나 골라내기
        if not isinstance(val_str, str):
            return name, None
        m = re.search(r"[-+]?\d*\.?\d+([eE][-+]?\d+)?", val_str)
        if m:
            try:
                num = float(m.group(0))
            except ValueError:
                num = None
        else:
            num = None

    if isinstance(name, str):
        name = name.strip().lower() or None

    return name, num


# ---------------------------
# 임베딩 기반 step 매칭
# ---------------------------

def build_embeddings(model: SentenceTransformer,
                     steps: List[Dict[str, Any]]) -> Tuple[List[str], np.ndarray]:
    texts = [get_step_text(s) for s in steps]
    emb = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    return texts, emb


def match_steps(
        model: SentenceTransformer,
        gen_steps: List[Dict[str, Any]],
        gold_steps: List[Dict[str, Any]],
        threshold: float = 0.7,
) -> Tuple[Dict[int, int], List[int]]:
    """
    gen_step index -> gold_step index 매칭 (threshold 이상인 경우).
    매칭이 없는 gen index 는 hallucinated 로 간주.
    """
    if not gen_steps or not gold_steps:
        return {}, list(range(len(gen_steps)))

    _, gen_emb = build_embeddings(model, gen_steps)
    _, gold_emb = build_embeddings(model, gold_steps)

    sim = util.cos_sim(gen_emb, gold_emb).cpu().numpy()  # shape: [G, H]

    gen_to_gold = {}
    hallucinated_indices = []

    for gi in range(sim.shape[0]):
        gj = int(np.argmax(sim[gi]))
        if sim[gi, gj] >= threshold:
            gen_to_gold[gi] = gj
        else:
            hallucinated_indices.append(gi)

    return gen_to_gold, hallucinated_indices


# ---------------------------
# Metric 계산
# ---------------------------

def compute_step_hallucination_rate(
        gen_steps: List[Dict[str, Any]],
        gen_to_gold: Dict[int, int],
) -> float:
    if not gen_steps:
        return 0.0
    hallucinated = len(gen_steps) - len(gen_to_gold)
    return hallucinated / len(gen_steps)


def compute_parameter_supported_rate(
        gen_steps: List[Dict[str, Any]],
        sec_text: str,
) -> Tuple[float, int, int]:
    """
    sec_text에서 substring 검색 기반으로 지원 여부 판정.
    """
    sec_lower = sec_text.lower()
    total = 0
    supported = 0

    for step in gen_steps:
        params = get_parameters(step)
        for p in params:
            s = param_to_string(p)
            if not s:
                continue
            total += 1
            if s.lower() in sec_lower:
                supported += 1

    if total == 0:
        return 0.0, 0, 0
    return supported / total, supported, total


def compute_contradiction_rate(
        gen_steps: List[Dict[str, Any]],
        gold_steps: List[Dict[str, Any]],
        gen_to_gold: Dict[int, int],
        tol: float = 1e-6,
) -> Tuple[float, int, int]:
    """
    name + numeric value 기준으로 param 차이가 나는 경우 contradiction.
    """
    contradictions = 0
    comparable = 0

    for gi, gj in gen_to_gold.items():
        gen_params = get_parameters(gen_steps[gi])
        gold_params = get_parameters(gold_steps[gj])

        # gold param name -> numeric value map
        gold_map = {}
        for p in gold_params:
            name, val = get_param_name_and_value(p)
            if name is not None and val is not None:
                gold_map[name] = val

        for p in gen_params:
            name, val = get_param_name_and_value(p)
            if name is None or val is None:
                continue
            if name not in gold_map:
                continue
            comparable += 1
            if abs(val - gold_map[name]) > tol:
                contradictions += 1

    if comparable == 0:
        return 0.0, 0, 0
    return contradictions / comparable, contradictions, comparable


def compute_evidence_missing_rate(
        gen_steps: List[Dict[str, Any]],
) -> Tuple[float, int, int]:
    """
    evidence/span_chunk/evidence_spans 가 비어 있거나 없으면 missing.
    """
    if not gen_steps:
        return 0.0, 0, 0

    missing = 0
    for step in gen_steps:
        ev = get_evidence(step)
        if ev is None:
            missing += 1
        elif isinstance(ev, str) and not ev.strip():
            missing += 1
        elif isinstance(ev, list) and len(ev) == 0:
            missing += 1
        # dict 등 다른 타입은 "있다"고 가정

    rate = missing / len(gen_steps)
    return rate, missing, len(gen_steps)


# ---------------------------
# 메인 루프
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-actions", type=str, required=True)
    parser.add_argument("--gold-pairs", type=str, required=True)
    parser.add_argument("--gen-actions", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--step-threshold", type=float, default=0.7)
    args = parser.parse_args()

    gold_actions_path = Path(args.gold_actions)
    gold_pairs_path = Path(args.gold_pairs)
    gen_actions_path = Path(args.gen_actions)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading data...")
    gold_actions = load_jsonl(gold_actions_path)
    gen_actions = load_jsonl(gen_actions_path)

    gold_by_pid = index_by_protocol(gold_actions)
    gen_by_pid = index_by_protocol(gen_actions)
    sec_text_by_pid = load_sec_text_by_protocol(gold_pairs_path)

    print(f"[INFO] Loaded {len(gold_actions)} gold actions, {len(gen_actions)} gen actions")
    print(f"[INFO] Protocols in gen: {len(gen_by_pid)}, in gold: {len(gold_by_pid)}")

    print(f"[INFO] Loading embedding model (allenai/scibert_scivocab_uncased)...")
    model = SentenceTransformer("allenai/scibert_scivocab_uncased")

    rows = []

    # protocol_id 기준 공통으로 있는 경우만 평가
    protocol_ids = sorted(set(gen_by_pid.keys()) & set(gold_by_pid.keys()))

    for pid in protocol_ids:
        gen_steps = gen_by_pid[pid]
        gold_steps = gold_by_pid[pid]
        sec_text = sec_text_by_pid.get(pid, "")

        # 1) Step 매칭
        gen_to_gold, hallucinated_indices = match_steps(
            model, gen_steps, gold_steps, threshold=args.step_threshold
        )

        step_hall_rate = compute_step_hallucination_rate(gen_steps, gen_to_gold)

        # 2) Parameter Supported Rate
        if sec_text:
            param_sup_rate, param_sup_cnt, param_total = compute_parameter_supported_rate(
                gen_steps, sec_text
            )
        else:
            param_sup_rate, param_sup_cnt, param_total = (0.0, 0, 0)

        # 3) Contradiction Rate
        contr_rate, contr_cnt, contr_total = compute_contradiction_rate(
            gen_steps, gold_steps, gen_to_gold
        )

        # 4) Evidence Missing Rate
        evid_miss_rate, evid_missing_cnt, evid_total = compute_evidence_missing_rate(
            gen_steps
        )

        rows.append({
            "protocol_id": pid,
            "num_gen_steps": len(gen_steps),
            "num_gold_steps": len(gold_steps),
            "num_matched_steps": len(gen_to_gold),
            "num_hallucinated_steps": len(gen_steps) - len(gen_to_gold),
            "step_hallucination_rate": step_hall_rate,

            "param_supported_rate": param_sup_rate,
            "param_supported_count": param_sup_cnt,
            "param_total_count": param_total,

            "contradiction_rate": contr_rate,
            "contradiction_count": contr_cnt,
            "contradiction_total": contr_total,

            "evidence_missing_rate": evid_miss_rate,
            "evidence_missing_count": evid_missing_cnt,
            "evidence_total": evid_total,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved per-protocol report to: {output_path}")

    if not df.empty:
        # 전체 평균/합계 요약 출력
        summary = {
            "num_protocols": len(df),
            "avg_step_hallucination_rate": df["step_hallucination_rate"].mean(),
            "avg_param_supported_rate": df["param_supported_rate"].replace([np.inf, -np.inf, np.nan], 0).mean(),
            "avg_contradiction_rate": df["contradiction_rate"].replace([np.inf, -np.inf, np.nan], 0).mean(),
            "avg_evidence_missing_rate": df["evidence_missing_rate"].mean(),
        }
        print("[SUMMARY]")
        for k, v in summary.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
