#!/usr/bin/env python3
"""
react_verifier_mini.py

미니 ReAct 스타일 파라미터 검증기. IR에서 추출한 파라미터들에 대해
Thought→Action→Observation→Verdict 과정을 수행한다. 각 파라미터는 독립적으로
처리되며, 간단한 문장 검색(doc_search)과 LLM 호출을 통해 supported/ambiguous/
unsupported verdict를 반환한다.

사용 예:
    python react_verifier_mini.py \
        --gold data/gold/gold_pairs_testset.jsonl \
        --ir runs/s2_parser.ir.jsonl \
        --out runs/react_verifier_results.jsonl \
        --model gpt-4o-mini \
        --max-protocols 2

실행 전 OPENAI_API_KEY 환경변수를 설정해야 하며, openai 패키지가 설치되어 있어야 한다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore


@dataclass
class ParamRecord:
    protocol_id: str
    node_id: str
    param_index: int
    name: Optional[str]
    value: Any
    unit: Optional[str]

    @property
    def label(self) -> str:
        parts = [str(self.name or ""), str(self.value or ""), str(self.unit or "")]
        return " ".join(p for p in parts if p).strip()


def load_gold_pairs(path: Path) -> Dict[str, str]:
    mapping = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pid = obj.get("protocol_id")
            text = obj.get("sec_text") or obj.get("text")
            if pid and text:
                mapping[str(pid)] = text
    return mapping


def load_ir(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
    return records


def extract_params_from_ir(ir_record: Dict[str, Any]) -> List[ParamRecord]:
    protocol_id = str(ir_record.get("protocol_id", ""))
    params: List[ParamRecord] = []
    for node in ir_record.get("nodes", []):
        node_id = node.get("id", "UNKNOWN")
        for idx, p in enumerate(node.get("params") or []):
            params.append(ParamRecord(
                protocol_id=protocol_id,
                node_id=node_id,
                param_index=idx,
                name=p.get("name"),
                value=p.get("value"),
                unit=p.get("unit"),
            ))
    return params


def extract_sentences(text: str) -> List[str]:
    raw = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    sentences, cur = [], []
    for tok in re.split(r"([.!?])", raw):
        cur.append(tok)
        if tok in (".", "?", "!"):
            s = "".join(cur).strip()
            if s: sentences.append(s)
            cur = []
    rest = "".join(cur).strip()
    if rest: sentences.append(rest)
    return sentences


def doc_search(param: ParamRecord, sentences: List[str], max_hits=5) -> List[str]:
    hits, patterns = [], []
    name = (param.name or "").lower().strip()
    value = str(param.value).lower().strip() if param.value is not None else ""
    unit = (param.unit or "").lower().strip()
    if name: patterns.append(re.escape(name))
    if value and unit:
        patterns += [re.escape(f"{value} {unit}"), re.escape(f"{value}{unit}")]
    elif value:
        patterns.append(re.escape(value))
    elif unit:
        patterns.append(re.escape(unit))
    for s in sentences:
        s_lower = s.lower()
        if any(re.search(p, s_lower) for p in patterns):
            hits.append(s)
            if len(hits) >= max_hits: break
    return hits


def build_verification_prompt(context_text: str, param: ParamRecord) -> str:
    system_instr = (
        "You are a strict scientific protocol verifier. "
        "Only trust information explicitly present in the context. "
        "Do not guess or infer plausible values. "
        "If not clearly supported, mark 'unsupported' or 'ambiguous'."
    )
    user = f"""
[Context]
---------
{context_text}
---------

[Parameter]
- name: {param.name}
- value: {param.value}
- unit: {param.unit}

Task:
1. Decide if the parameter (name + value + unit) is explicitly supported.
2. If yes, set 'verdict' to 'supported' and copy the relevant sentence into 'evidence_span'.
3. If the name is mentioned but value or unit are unclear, set 'ambiguous'.
4. If not found, set 'unsupported' and leave evidence_span empty.

Output a JSON object with 'verdict' and 'evidence_span' only.
"""
    return f"{system_instr}\n\nNow answer in JSON only.\n{user.strip()}"


# def call_llm(prompt: str, client: Any, model: str) -> str:
#     resp = client.chat.completions.create(
#         model=model,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.0,
#         response_format={"type": "json_object"},
#     )
#     return resp.choices[0].message.content

def call_llm(prompt: str, client: Any, model: str, max_tokens: int = 256) -> str:
    """Call the OpenAI chat completion API and return the raw JSON string."""
    last_err: Optional[Exception] = None
    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model=model,  # 예: "gpt-4.1-mini"
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,  # ★ 출력 길이 제한
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    # 실패 시 fallback
    return json.dumps({"verdict": "ambiguous", "evidence_span": "", "error": str(last_err) if last_err else ""})


def parse_llm_verdict(raw_output: str) -> Dict[str, Any]:
    try:
        obj = json.loads(raw_output.strip())
        verdict = obj.get("verdict", "ambiguous")
        evidence = obj.get("evidence_span", "")
    except Exception:
        verdict, evidence = "ambiguous", ""
    if verdict not in ("supported", "ambiguous", "unsupported"):
        verdict = "ambiguous"
    return {"verdict": verdict, "evidence_span": evidence, "llm_raw": raw_output}


def react_single_param(methods_text: str, param: ParamRecord, client: Any, model: str) -> Dict[str, Any]:
    trace = []
    sentences = extract_sentences(methods_text)
    # Thought 1: direct match?
    trace.append({"thought": "Check if parameter appears directly in the Methods text."})
    hits = doc_search(param, sentences, max_hits=5)
    if hits:
        trace.append({"action": "doc_search", "observation": f"{len(hits)} sentence(s) mention parameter"})
        context = "\n".join(hits)
    else:
        trace.append({"action": "doc_search", "observation": "no sentence mentioning parameter"})
        context = methods_text  # fall back to full context
    trace.append({"action": "call_llm", "observation": "LLM called for verdict"})
    prompt = build_verification_prompt(context, param)
    raw = call_llm(prompt, client, model)
    verdict_info = parse_llm_verdict(raw)
    return {
        **asdict(param),
        "verdict": verdict_info["verdict"],
        "evidence_span": verdict_info["evidence_span"],
        "llm_raw": verdict_info["llm_raw"],
        "trace": trace,
    }


def run_react_verification(gold_pairs_path: Path, ir_path: Path, out_path: Path, client: Any, model: str,
                           max_protocols=0) -> None:
    print(f"[INFO] Loading gold pairs from {gold_pairs_path}")
    protocol_to_methods = load_gold_pairs(gold_pairs_path)
    print(f"[INFO] Loaded Methods for {len(protocol_to_methods)} protocols.")
    print(f"[INFO] Loading IR records from {ir_path}")
    ir_records = load_ir(ir_path)
    print(f"[INFO] Loaded {len(ir_records)} IR records.")
    out_f = out_path.open("w", encoding="utf-8")
    total_params = 0
    verdict_counts = {"supported": 0, "ambiguous": 0, "unsupported": 0}
    processed = 0
    for ir in ir_records:
        if max_protocols and processed >= max_protocols:
            break
        pid = ir.get("protocol_id")
        methods = protocol_to_methods.get(str(pid))
        if not methods:
            continue
        processed += 1
        params = extract_params_from_ir(ir)
        for p in params:
            total_params += 1
            result = react_single_param(methods, p, client, model)
            verdict_counts[result["verdict"]] = verdict_counts.get(result["verdict"], 0) + 1
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
    out_f.close()
    print("[INFO] Done.")
    print(f"  processed protocols: {processed}")
    print(f"  total params: {total_params}")
    for k, v in verdict_counts.items():
        rate = v / total_params if total_params else 0.0
        print(f"  {k}: {v} ({rate:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini ReAct verifier for IR parameters")
    parser.add_argument("--gold", required=True, help="Path to gold_pairs_testset.jsonl")
    parser.add_argument("--ir", required=True, help="Path to s2_parser.ir.jsonl")
    parser.add_argument("--out", required=True, help="Output JSONL file")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="OpenAI chat model name")
    parser.add_argument("--max-protocols", type=int, default=0, help="Limit number of protocols (0 = no limit)")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("ERROR: OPENAI_API_KEY environment variable is not set.")
    if OpenAI is None:
        raise SystemExit("ERROR: install openai package with `pip install openai`.")

    client = OpenAI()
    run_react_verification(Path(args.gold), Path(args.ir), Path(args.out), client, args.model, args.max_protocols)


if __name__ == "__main__":
    main()
