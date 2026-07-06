#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
[개선된 Grounding Evaluation Script]

개선사항:
1. 더 정교한 문장 추출 (청크 기반 + 의미 단위)
2. Methods 청킹 전략 개선 (문장 단위 청킹 추가)
3. 할루시네이션 검출 개선
4. 자세한 로깅 및 디버깅 정보
5. 임계값 비교 (0.55, 0.60, 0.65)

핵심 지표:
- Grounded Rate: Methods에서 근거를 찾을 수 있는 문장 비율
- Hallucination Rate: 근거를 찾을 수 없는 문장 비율
- Semantic Grounding: 의미적으로 유사한 근거 찾기
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
import warnings

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings("ignore")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """JSONL 파일 로드"""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _norm_ws(s: str) -> str:
    """공백 정규화"""
    return re.sub(r"\s+", " ", (s or "")).strip()


def _as_str(x: Any) -> str:
    """안전한 문자열 변환"""
    return x.strip() if isinstance(x, str) else ""


def extract_methods_text(rec: Dict[str, Any]) -> str:
    """Methods 섹션 텍스트 추출"""
    if _as_str(rec.get("sec_text")):
        return rec["sec_text"].strip()
    article = rec.get("article") or {}
    if _as_str(article.get("sec_text")):
        return article["sec_text"].strip()
    sections = article.get("sections")
    if isinstance(sections, dict) and _as_str(sections.get("Methods")):
        return sections["Methods"].strip()
    return ""


def split_into_sentences(text: str) -> List[str]:
    """문장 단위로 분할"""
    # 문장 분할 정규식
    sent_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\n'
    sentences = re.split(sent_pattern, text)

    # 필터링: 너무 짧거나 비어있는 문장 제거
    sentences = [_norm_ws(s) for s in sentences]
    sentences = [s for s in sentences if len(s) > 10]
    return sentences


def chunk_text_by_sentences(text: str, chunk_sents: int = 3, overlap_sents: int = 1) -> List[str]:
    """문장 기반 청킹"""
    sents = split_into_sentences(text)
    if not sents:
        return []

    chunks = []
    for i in range(0, len(sents), chunk_sents - overlap_sents):
        chunk = " ".join(sents[i:i + chunk_sents])
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_sents >= len(sents):
            break
    return chunks


def chunk_text_by_chars(text: str, chunk_chars: int = 1200, overlap: int = 200) -> List[str]:
    """문자 기반 청킹"""
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


def extract_sentences_from_protocol(protocol: Dict[str, Any]) -> List[str]:
    """생성된 프로토콜에서 문장 추출"""
    sentences = []

    # 1. 직접 'sentences' 필드
    if isinstance(protocol.get("sentences"), list):
        sentences.extend([s for s in protocol["sentences"] if isinstance(s, str) and len(s.strip()) > 10])

    # 2. 'protocol_text' 필드
    if isinstance(protocol.get("protocol_text"), str):
        text = protocol["protocol_text"].strip()
        if text:
            sentences.extend(split_into_sentences(text))

    # 3. ActionIR 리스트에서 설명 추출
    if isinstance(protocol.get("actions"), list):
        for action in protocol["actions"]:
            if isinstance(action, dict):
                if isinstance(action.get("description"), str):
                    sentences.append(_norm_ws(action["description"]))
                if isinstance(action.get("action"), str):
                    sentences.append(_norm_ws(action["action"]))

    # 4. 재귀적으로 모든 문자열 필드 추출
    def collect_strings(obj, out):
        if isinstance(obj, str) and len(obj.strip()) > 10:
            out.append(_norm_ws(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                collect_strings(v, out)
        elif isinstance(obj, list):
            for item in obj:
                collect_strings(item, out)

    collect_strings(protocol, sentences)

    # 중복 제거
    unique = []
    seen = set()
    for s in sentences:
        if s not in seen and len(s) > 10:
            seen.add(s)
            unique.append(s)

    return unique


def compute_grounding(
        model: SentenceTransformer,
        methods_text: str,
        sentences: List[str],
        thresholds: List[float] = None,
        chunk_type: str = "both"
) -> Dict[str, Any]:
    """
    Grounding 계산 (향상된 버전)

    Args:
        model: SentenceTransformer 모델
        methods_text: Methods 섹션 텍스트
        sentences: 평가할 문장 리스트
        thresholds: 유사도 임계값 리스트
        chunk_type: "sent" (문장 기반), "char" (문자 기반), "both" (둘 다)
    """
    if thresholds is None:
        thresholds = [0.55, 0.60, 0.65]

    sents = [s for s in (sentences or []) if isinstance(s, str) and len(s.strip()) > 10]

    if not sents:
        result = {"n_sents": 0, "methods_length": len(methods_text)}
        for thr in thresholds:
            result[f"grounded_rate_{thr}"] = 0.0
            result[f"hallucination_rate_{thr}"] = 1.0
            result[f"grounded_{thr}"] = 0
        return result

    if not methods_text or len(methods_text.strip()) < 50:
        result = {"n_sents": len(sents), "methods_length": len(methods_text)}
        for thr in thresholds:
            result[f"grounded_rate_{thr}"] = 0.0
            result[f"hallucination_rate_{thr}"] = 1.0
            result[f"grounded_{thr}"] = 0
        return result

    # 청킹 전략 선택
    if chunk_type == "sent":
        chunks = chunk_text_by_sentences(methods_text, chunk_sents=3, overlap_sents=1)
    elif chunk_type == "char":
        chunks = chunk_text_by_chars(methods_text, chunk_chars=1200, overlap=200)
    else:  # both
        chunks_sent = chunk_text_by_sentences(methods_text, chunk_sents=3, overlap_sents=1)
        chunks_char = chunk_text_by_chars(methods_text, chunk_chars=1200, overlap=200)
        chunks = list(set(chunks_sent + chunks_char))  # 중복 제거

    if not chunks:
        result = {"n_sents": len(sents), "methods_length": len(methods_text)}
        for thr in thresholds:
            result[f"grounded_rate_{thr}"] = 0.0
            result[f"hallucination_rate_{thr}"] = 1.0
            result[f"grounded_{thr}"] = 0
        return result

    # 임베딩 계산
    try:
        chunk_emb = model.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
        sent_emb = model.encode(sents, convert_to_tensor=True, show_progress_bar=False)
        sim_matrix = util.cos_sim(sent_emb, chunk_emb).cpu().numpy()  # (n_sents, n_chunks)
    except Exception as e:
        print(f"❌ 임베딩 계산 오류: {e}")
        result = {"n_sents": len(sents), "methods_length": len(methods_text), "error": str(e)}
        for thr in thresholds:
            result[f"grounded_rate_{thr}"] = 0.0
            result[f"hallucination_rate_{thr}"] = 1.0
            result[f"grounded_{thr}"] = 0
        return result

    result = {
        "n_sents": len(sents),
        "n_chunks": len(chunks),
        "methods_length": len(methods_text),
        "methods_n_sents": len(split_into_sentences(methods_text))
    }

    # 각 임계값에 대해 계산
    for thr in thresholds:
        grounded = (sim_matrix.max(axis=1) >= thr).sum()  # 각 문장의 최대 유사도가 임계값 이상인 경우
        grounded_rate = grounded / len(sents)
        hallucination_rate = 1.0 - grounded_rate

        result[f"grounded_rate_{thr}"] = grounded_rate
        result[f"hallucination_rate_{thr}"] = hallucination_rate
        result[f"grounded_{thr}"] = grounded

        # 상세 분석
        max_sims = sim_matrix.max(axis=1)
        result[f"avg_max_sim_{thr}"] = float(max_sims.mean())
        result[f"median_max_sim_{thr}"] = float(np.median(max_sims))
        result[f"min_max_sim_{thr}"] = float(max_sims.min())
        result[f"max_max_sim_{thr}"] = float(max_sims.max())

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default="/Users/haeb/Workspaces/BioProtocol/BioProtocolAgent")
    ap.add_argument("--gold_pairs", type=str, default="data/gold_pairs_testset_v2.jsonl")
    ap.add_argument("--generated_dir", type=str, default="reports/llm_protocols")
    ap.add_argument("--pattern", type=str, default="generated_P*.jsonl")
    ap.add_argument("--out_dir", type=str, default="reports/grounding_eval")

    ap.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--embed_device", type=str, default=None)
    ap.add_argument("--chunk_type", type=str, default="both", choices=["sent", "char", "both"])
    ap.add_argument("--verbose", action="store_true", help="상세 출력")

    args = ap.parse_args()

    root = Path(args.project_root)
    gold_pairs_path = root / args.gold_pairs
    gen_dir = root / args.generated_dir
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 프로젝트 루트: {root}")
    print(f"📄 Gold 페어: {gold_pairs_path}")
    print(f"📊 생성된 프로토콜: {gen_dir}")
    print(f"💾 출력 디렉토리: {out_dir}")
    print(f"🔤 임베딩 모델: {args.embed_model}")
    print(f"🔀 청킹 타입: {args.chunk_type}")
    print()

    # Gold pairs 로드
    print("📥 Gold pairs 로드 중...")
    gold_pairs = load_jsonl(gold_pairs_path)
    methods_map = {}
    for rec in gold_pairs:
        pid = rec.get("protocol_id")
        if pid:
            methods_map[pid] = extract_methods_text(rec)
    print(f"✅ {len(methods_map)}개 프로토콜 로드")
    print()

    # 모델 로드
    print(f"🤖 {args.embed_model} 로드 중...")
    model = SentenceTransformer(args.embed_model, device=args.embed_device)
    print("✅ 모델 로드 완료")
    print()

    rows = []
    thresholds = [0.55, 0.60, 0.65]

    # 생성된 프로토콜 평가
    gen_files = sorted(gen_dir.glob(args.pattern))
    if not gen_files:
        print(f"❌ 생성된 파일을 찾을 수 없음: {gen_dir / args.pattern}")
        return

    print(f"📋 {len(gen_files)}개 모드 평가 중...")
    for gen_path in gen_files:
        mode = gen_path.stem.replace("generated_", "")
        print(f"\n  📄 {mode}...")

        recs = load_jsonl(gen_path)
        for i, rec in enumerate(recs):
            pid = rec.get("protocol_id")
            if not pid or pid not in methods_map:
                continue

            methods_text = methods_map[pid]
            sentences = extract_sentences_from_protocol(rec)

            if args.verbose:
                print(f"    [{i+1}/{len(recs)}] {pid}: {len(sentences)} 문장, Methods {len(methods_text)} 자")

            grounding = compute_grounding(model, methods_text, sentences, thresholds, args.chunk_type)

            rows.append({
                "mode": mode,
                "protocol_id": pid,
                **grounding
            })

    print()
    print("💾 결과 저장 중...")

    # Per-protocol 결과
    df = pd.DataFrame(rows).sort_values(["mode", "protocol_id"])
    out_pp = out_dir / "per_protocol_grounding_v2.csv"
    df.to_csv(out_pp, index=False)
    print(f"✅ {out_pp}")

    # 모드별 요약
    agg_dict = {}
    for col in df.columns:
        if col not in ["mode", "protocol_id"]:
            agg_dict[col] = "mean"

    df_mode = df.groupby("mode").agg(agg_dict).reset_index()
    out_sum = out_dir / "summary_modes_grounding_v2.csv"
    df_mode.to_csv(out_sum, index=False)
    print(f"✅ {out_sum}")

    # 통계 출력
    print()
    print("=" * 80)
    print("📊 GROUNDING EVALUATION RESULTS")
    print("=" * 80)
    print()

    print("🎯 모드별 할루시네이션율 (Hallucination Rate):")
    print("-" * 80)
    for mode in sorted(df["mode"].unique()):
        mode_data = df[df["mode"] == mode]
        print(f"\n  {mode}:")
        print(f"    프로토콜 수: {len(mode_data)}")
        for thr in thresholds:
            avg_hall = mode_data[f"hallucination_rate_{thr}"].mean()
            avg_ground = mode_data[f"grounded_rate_{thr}"].mean()
            print(f"    Threshold {thr}: Grounded={avg_ground:.1%}, Hallucination={avg_hall:.1%}")

    print()
    print("🔍 상세 통계 (Threshold 0.60):")
    print("-" * 80)
    for mode in sorted(df["mode"].unique()):
        mode_data = df[df["mode"] == mode]
        print(f"\n  {mode}:")
        print(f"    평균 문장 수: {mode_data['n_sents'].mean():.1f}")
        if 'n_chunks' in mode_data.columns:
            print(f"    평균 청크 수: {mode_data['n_chunks'].mean():.1f}")
        if 'avg_max_sim_0.6' in mode_data.columns:
            print(f"    평균 최대 유사도: {mode_data['avg_max_sim_0.6'].mean():.3f}")
            print(f"    중간값 최대 유사도: {mode_data['median_max_sim_0.6'].mean():.3f}")

    print()
    print("✅ 평가 완료!")
    print()


if __name__ == "__main__":
    main()
