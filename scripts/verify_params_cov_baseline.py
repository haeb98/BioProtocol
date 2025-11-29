#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_params_cov_baseline.py

현재 Task→Step→IR 파이프라인에서 생성된
runs/ir_graphs_from_steps_llm.jsonl 을 대상으로
"기본 CoV-style" 파라미터 검증을 수행하는 스크립트.

- 입력: IR JSONL (protocol_id, methods_text, nodes, edges, ...)
- 각 노드의 params를 flatten해서 param 리스트 생성
- LLM에 Methods 전체 + 단일 파라미터를 주고
  verdict ∈ {supported, ambiguous, unsupported}를 받는 baseline

pseudo-ReAct(verify_params_from_ir_react.py)와
동일한 IR을 쓰되, 도구 사용 없이 단순 CoV 프롬프트만 사용하는 버전.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI


# -----------------------------
# 공통 유틸
# -----------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def save_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# -----------------------------
# IR → 파라미터 플랫화
# -----------------------------

def flatten_params_from_ir_record(ir_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    ir_from_steps.py 로 생성된 IR 레코드에서
    nodes[*].params 를 flat한 param 레코드 리스트로 변환.
    """
    protocol_id = str(ir_record.get("protocol_id", ""))
    nodes = ir_record.get("nodes", []) or []

    params: List[Dict[str, Any]] = []

    for node in nodes:
        node_type = node.get("type", "Step")
        node_id = node.get("id") or ""
        node_title = node.get("title")
        task_ref = node.get("task_ref") or node.get("task_id")

        for p in node.get("params", []) or []:
            # p 안에는 name, value, unit, raw, source 등이 들어있다고 가정
            name = p.get("name") or ""
            value = p.get("value")
            unit = p.get("unit")
            raw = p.get("raw")
            source = p.get("source")
            if not name and not raw:
                continue

            rec = {
                "protocol_id": protocol_id,
                "step_id": node_id,
                "task_ref": task_ref,
                "node_title": node_title,
                "node_type": node_type,
                "name": name,
                "value": value,
                "unit": unit,
                "raw": raw,
                "source": source,
            }
            params.append(rec)

    return params


# -----------------------------
# CoV-style LLM 프롬프트
# -----------------------------

SYS_PROMPT = """
You are a strict parameter verifier (CoV-style) for biology protocols.

You will be given:
- The full Methods text of a protocol.
- A single parameter (name, value, unit, raw).
Your job is to decide whether this parameter is actually supported by the Methods text.

You MUST:
1. Carefully read the Methods text.
2. Check whether the exact or clearly equivalent value+unit (or meaningfully same condition)
   is explicitly stated.
3. Classify the parameter into one of:
   - "supported": clearly and directly supported (value+unit or equivalent) in the text.
   - "ambiguous": text is related, but the value/unit is approximate, inferred, or partially missing.
   - "unsupported": no reasonable support in the Methods text.

Return a JSON object with fields:
- verdict: "supported" | "ambiguous" | "unsupported"
- evidence_span: short quote from the Methods text if supported or ambiguous; "" if unsupported.

Do NOT hallucinate values. If in doubt, choose "ambiguous" rather than "supported".
"""


def build_cov_prompt(methods_text: str, param: Dict[str, Any]) -> str:
    """
    Methods 전체 + param 정보를 하나의 JSON payload로 구성해서
    user 메시지 content로 넘김.
    """
    payload = {
        "methods_text": methods_text,
        "parameter": {
            "name": param.get("name"),
            "value": param.get("value"),
            "unit": param.get("unit"),
            "raw": param.get("raw"),
            "step_id": param.get("step_id"),
            "task_ref": param.get("task_ref"),
            "node_title": param.get("node_title"),
            "node_type": param.get("node_type"),
        },
        "instruction": (
            "Decide whether this parameter is supported by the Methods text. "
            "Return JSON with fields {\"verdict\", \"evidence_span\"} as described."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


import time


def call_llm_cov(prompt: str, client: OpenAI, model: str,
                 max_retries: int = 5, sleep_sec: float = 3.0) -> Dict[str, Any]:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            # 정상 응답 → throttle
            time.sleep(sleep_sec)
            content = resp.choices[0].message.content

            try:
                js = json.loads(content)
            except json.JSONDecodeError:
                js = {}

            verdict = js.get("verdict", "ambiguous")
            evidence_span = js.get("evidence_span", "")

            if verdict not in ("supported", "ambiguous", "unsupported"):
                verdict = "ambiguous"
            if not isinstance(evidence_span, str):
                evidence_span = ""

            # ✅ 항상 raw 포함해서 리턴
            return {
                "verdict": verdict,
                "evidence_span": evidence_span,
                "raw": js,
            }

        except Exception as e:
            msg = str(e).lower()
            if "rate limit" in msg or "429" in msg:
                wait = sleep_sec * (attempt + 1)
                print(f"[WARN] RateLimit: waiting {wait}s ...")
                time.sleep(wait)
                continue
            print(f"[ERROR] Unexpected error: {e}")
            # 오류 시에도 형태를 맞춰서 돌려줌
            return {
                "verdict": "ambiguous",
                "evidence_span": "",
                "raw": {"error": str(e)},
            }

    print("[ERROR] Failed after retries → ambiguous fallback")
    return {
        "verdict": "ambiguous",
        "evidence_span": "",
        "raw": {"error": "max_retries_exceeded"},
    }


# -----------------------------
# 메인 실행
# -----------------------------

def run_baseline_cov(ir_path: Path, out_path: Path, model: str) -> None:
    client = OpenAI()

    ir_records = load_jsonl(ir_path)
    print(f"[INFO] Loaded {len(ir_records)} IR records from {ir_path}")

    all_results: List[Dict[str, Any]] = []
    total_params = 0

    for idx, rec in enumerate(ir_records, start=1):
        protocol_id = rec.get("protocol_id")
        methods_text = rec.get("methods_text") or ""
        if not protocol_id:
            print("[WARN] Missing protocol_id in IR record, skipping.")
            continue

        params = flatten_params_from_ir_record(rec)
        if not params:
            print(f"[INFO] No params found for protocol {protocol_id}, skipping.")
            continue

        print(f"[INFO] [{idx}] protocol_id={protocol_id}, #params={len(params)}")
        for p in params:
            prompt = build_cov_prompt(methods_text, p)
            js = call_llm_cov(prompt, client, model)

            result = {
                "protocol_id": p["protocol_id"],
                "step_id": p["step_id"],
                "task_ref": p.get("task_ref"),
                "param_name": p.get("name"),
                "param_value": p.get("value"),
                "param_unit": p.get("unit"),
                "param_raw": p.get("raw"),
                "node_title": p.get("node_title"),
                "node_type": p.get("node_type"),
                "verdict": js["verdict"],
                "evidence_span": js["evidence_span"],
                "llm_raw": js["raw"],
            }
            all_results.append(result)
            total_params += 1

    print(f"[INFO] Total params processed: {total_params}")
    print(f"[INFO] Saving {len(all_results)} CoV baseline verdicts to {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(out_path, all_results)
    print("[INFO] Done.")


def main():
    parser = argparse.ArgumentParser(
        description="CoV-style baseline verifier on IR graphs from steps."
    )
    parser.add_argument(
        "--ir",
        type=str,
        default="runs/ir_graphs_from_steps_llm.jsonl",
        help="IR JSONL file produced by ir_from_steps.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/param_verdicts_cov_baseline.jsonl",
        help="Output JSONL for param-level CoV verdicts.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI model name (e.g., gpt-4.1-mini, gpt-4o-mini).",
    )
    args = parser.parse_args()

    run_baseline_cov(Path(args.ir), Path(args.output), args.model)


if __name__ == "__main__":
    main()
