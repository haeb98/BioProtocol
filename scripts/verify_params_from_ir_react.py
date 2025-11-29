#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List

from openai import OpenAI

client = OpenAI()


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def save_jsonl(path: Path, records: List[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ------------------------
# ReAct용 "도구" 구현부
# ------------------------

def unit_normalize(value, unit: str):
    """간단한 단위 정규화. (필요하면 점점 확장 가능)"""
    if unit is None:
        return value, unit

    u = unit.strip()
    mapping = {
        "hr": "h",
        "hrs": "h",
        "hour": "h",
        "hours": "h",
        "μL": "uL",
        "µL": "uL",
        "uL": "uL",
        "ml": "mL",
        "ML": "mL",
        "°C": "C",
        "degC": "C",
        "x g": "xg",
        "x g.": "xg",
    }
    norm_unit = mapping.get(u, u)
    return value, norm_unit


def split_sentences(text: str) -> List[str]:
    # 아주 단순한 문장 분할 ('.', '?', '!' 기준)
    # 필요시 더 고도화 가능
    sents = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    return [s for s in sents if s]


def doc_search(methods_text: str, param: Dict[str, Any], top_k: int = 5) -> List[str]:
    """param 관련 evidence가 있을 법한 문장들을 methods에서 찾아서 반환."""
    sents = split_sentences(methods_text)
    name = (param.get("name") or "").lower()
    raw = (param.get("raw") or "").lower()
    unit = (param.get("unit") or "").lower()

    scored = []
    for idx, s in enumerate(sents):
        s_low = s.lower()
        score = 0
        if raw and raw in s_low:
            score += 3
        if name and name in s_low:
            score += 2
        if unit and unit in s_low:
            score += 1
        # 숫자 포함 여부도 약간 가점
        if re.search(r"\d", s_low):
            score += 0.5
        if score > 0:
            scored.append((score, idx, s))

    scored.sort(key=lambda x: (-x[0], x[1]))
    hits = [s for _, _, s in scored[:top_k]]
    return hits


# ------------------------
# LLM 호출부
# ------------------------

SYS_PROMPT = """
You are a ReAct-style parameter verifier for biology protocols.

You will be given:
- A parameter (with name, value, unit, raw).
- Tool outputs:
  - DOC_SEARCH: sentences from the Methods that are likely relevant.
  - UNIT_NORMALIZE: a normalized (value, unit) suggestion.

Your job:
1. Decide whether the parameter is supported by the Methods.
2. Classify into one of:
   - "supported": clearly and directly stated (value+unit or equivalent).
   - "ambiguous": related info exists but value/unit is approximate, inferred, or unclear.
   - "unsupported": cannot find reasonable support in the Methods.
3. Provide a short evidence_span (a quote from the Methods) if supported or ambiguous.
4. If unsupported, evidence_span can be "".

Return a JSON object with fields:
- verdict: "supported" | "ambiguous" | "unsupported"
- evidence_span: string
"""


def build_react_prompt(methods_text: str, param: Dict[str, Any],
                       doc_hits: List[str], norm_value, norm_unit: str) -> str:
    tools_section = {
        "DOC_SEARCH_hits": doc_hits,
        "UNIT_NORMALIZE": {
            "input": {"value": param.get("value"), "unit": param.get("unit")},
            "normalized": {"value": norm_value, "unit": norm_unit},
        },
    }

    payload = {
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
        "tools_output": tools_section,
        # full Methods는 토큰 절약을 위해 (옵션)으로 줄 수도 있지만
        # 여기선 doc_search가 이미 문맥 축소를 해줬으므로 전체는 빼거나 짧게 trunc 해도 됨.
        "methods_note": "Full Methods text is available to you conceptually, but you must rely primarily on DOC_SEARCH hits above."
    }

    return json.dumps(payload, ensure_ascii=False, indent=2)


def call_llm(prompt: str, model: str):
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    content = resp.choices[0].message.content
    try:
        js = json.loads(content)
    except Exception as e:
        print(f"[ERROR] Failed to parse LLM JSON: {e}")
        js = {"verdict": "ambiguous", "evidence_span": ""}
    js.setdefault("verdict", "ambiguous")
    js.setdefault("evidence_span", "")
    return js


# ------------------------
# 메인 로직
# ------------------------

def flatten_params_from_nodes(nodes: List[Dict[str, Any]], protocol_id: str) -> List[Dict[str, Any]]:
    """IR nodes 안의 params를 flat param 레코드 리스트로 변환."""
    params = []
    for node in nodes:
        node_type = node.get("type", "Step")
        node_id = node.get("id")
        node_title = node.get("title")
        task_ref = node.get("task_ref") or node.get("task_id")

        for p in node.get("params", []) or []:
            rec = dict(p)  # name, value, unit, raw, source ...
            rec["protocol_id"] = protocol_id
            rec["step_id"] = node_id
            rec["task_ref"] = task_ref
            rec["node_title"] = node_title
            rec["node_type"] = node_type
            params.append(rec)
    return params


def main():
    parser = argparse.ArgumentParser(
        description="ReAct-style parameter verifier on IR graphs (nodes+edges)."
    )
    parser.add_argument(
        "--ir",
        type=str,
        default="runs/ir_graphs_from_steps_llm.jsonl",
        help="IR JSONL file produced by ir_from_steps.py (with nodes, edges, methods_text).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/param_verdicts_react_from_steps_llm.jsonl",
        help="Output JSONL for param-level ReAct verdicts.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI model name for ReAct verifier.",
    )
    parser.add_argument(
        "--max_protocols",
        type=int,
        default=0,
        help="Optional cap on number of protocols to process (0 = all).",
    )
    parser.add_argument(
        "--max_params",
        type=int,
        default=0,
        help="Optional cap on total number of params across all protocols (0 = all).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Top-k DOC_SEARCH hits per param.",
    )

    args = parser.parse_args()
    ir_path = Path(args.ir)
    out_path = Path(args.output)

    assert ir_path.exists(), f"IR file not found: {ir_path}"

    all_verdicts: List[Dict[str, Any]] = []
    print(f"[INFO] Loading IR graphs from {ir_path}")

    total_params = 0

    for i, rec in enumerate(load_jsonl(ir_path), start=1):
        protocol_id = rec.get("protocol_id")
        methods_text = rec.get("methods_text", "")
        nodes = rec.get("nodes", [])

        if not protocol_id:
            print("[WARN] Missing protocol_id in IR record, skipping.")
            continue
        if not nodes:
            print(f"[WARN] No nodes for protocol {protocol_id}, skipping.")
            continue

        params = flatten_params_from_nodes(nodes, protocol_id)
        if not params:
            print(f"[WARN] No params found in IR for {protocol_id}, skipping.")
            continue

        print(f"[INFO] [{i}] protocol_id={protocol_id}, #params={len(params)}")

        for p in params:
            if args.max_params and total_params >= args.max_params:
                print(f"[INFO] Reached max_params={args.max_params}, stopping.")
                break

            # 1) TOOL: unit_normalize
            norm_value, norm_unit = unit_normalize(p.get("value"), p.get("unit"))

            # 2) TOOL: doc_search
            hits = doc_search(methods_text, p, top_k=args.top_k)

            # 3) Prompt 구성
            prompt = build_react_prompt(methods_text, p, hits, norm_value, norm_unit)

            # 4) LLM 호출
            js = call_llm(prompt, args.model)

            verdict_rec = {
                "protocol_id": p["protocol_id"],
                "step_id": p["step_id"],
                "task_ref": p.get("task_ref"),
                "param_name": p.get("name"),
                "param_value": p.get("value"),
                "param_unit": p.get("unit"),
                "param_raw": p.get("raw"),
                "node_title": p.get("node_title"),
                "node_type": p.get("node_type"),
                "verdict": js.get("verdict", "ambiguous"),
                "evidence_span": js.get("evidence_span", ""),
                "doc_hits": hits,
                "normalized_unit": norm_unit,
            }
            all_verdicts.append(verdict_rec)
            total_params += 1

        if args.max_params and total_params >= args.max_params:
            break

        if args.max_protocols and i >= args.max_protocols:
            print(f"[INFO] Reached max_protocols={args.max_protocols}, stopping.")
            break

    print(f"[INFO] Total params processed: {total_params}")
    print(f"[INFO] Saving {len(all_verdicts)} ReAct verdicts to {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(out_path, all_verdicts)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
