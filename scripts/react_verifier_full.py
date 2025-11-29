#!/usr/bin/env python3
"""
react_verifier_full.py

"진짜" ReAct 스타일 파라미터 검증기.

- 입력: ir_from_steps.py 등으로 생성한 IR JSONL
  (각 레코드: {protocol_id, methods_text, nodes, edges, ...})
- 각 노드의 params를 flatten 해서 파라미터 레코드를 만들고,
- 각 파라미터에 대해 ReAct 루프:
    Thought -> Action(tool) -> Observation -> ... -> Final Answer(JSON)
  를 수행한다.

Action으로 사용할 수 있는 tool:
- doc_search: Methods 텍스트에서 관련 문장 검색
- unit_normalize: 단위/값 간단 정규화
- bio_concept_lookup: 생물학 개념 간단 정의(작은 dict 기반)

※ 이건 **케이스 스터디용**이라, 전체 46개 프로토콜/수백 파라미터에 돌리는 것보다
2~5개 프로토콜, 20~50개 파라미터에 돌려서 ReAct trace를 분석하는 용도로 사용하는 걸 권장.
"""

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# -----------------------
# 기본 유틸
# -----------------------

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


# -----------------------
# IR → 파라미터 플랫화
# -----------------------

@dataclass
class ParamRecord:
    protocol_id: str
    step_id: str
    task_ref: Optional[str]
    node_title: Optional[str]
    node_type: str
    name: str
    value: Any
    unit: Optional[str]
    raw: Optional[str]
    source: Optional[str]


def flatten_params_from_ir_record(ir_record: Dict[str, Any]) -> List[ParamRecord]:
    protocol_id = str(ir_record.get("protocol_id", ""))
    nodes = ir_record.get("nodes", []) or []
    params: List[ParamRecord] = []

    for node in nodes:
        node_type = node.get("type", "Step")
        node_id = node.get("id") or ""
        node_title = node.get("title")
        task_ref = node.get("task_ref") or node.get("task_id")

        for p in node.get("params", []) or []:
            name = p.get("name") or ""
            value = p.get("value")
            unit = p.get("unit")
            raw = p.get("raw")
            source = p.get("source")
            if not name and not raw:
                continue
            params.append(
                ParamRecord(
                    protocol_id=protocol_id,
                    step_id=node_id,
                    task_ref=task_ref,
                    node_title=node_title,
                    node_type=node_type,
                    name=name,
                    value=value,
                    unit=unit,
                    raw=raw,
                    source=source,
                )
            )
    return params


# -----------------------
# ReAct용 툴 구현
# -----------------------

def unit_normalize(value: Any, unit: Optional[str]) -> Tuple[Any, Optional[str]]:
    """간단한 단위 정규화."""
    if unit is None:
        return value, unit
    u = str(unit).strip()

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
    sents = re.split(r"(?<=[\.!\?])\s+", text.strip())
    return [s for s in sents if s]


def doc_search(methods_text: str, query: str, top_k: int = 5) -> List[str]:
    """아주 단순한 문장 검색: query 토큰들이 많이 겹치는 문장을 우선."""
    sents = split_sentences(methods_text)
    if not query:
        return sents[:top_k]

    q_tokens = [t for t in re.split(r"\W+", query.lower()) if t]
    scored: List[Tuple[float, int, str]] = []
    for idx, s in enumerate(sents):
        s_low = s.lower()
        score = 0.0
        for t in q_tokens:
            if t and t in s_low:
                score += 1.0
        # 숫자 포함 여부 약간 가점
        if re.search(r"\d", s_low):
            score += 0.5
        if score > 0:
            scored.append((score, idx, s))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [s for _, _, s in scored[:top_k]]


# (선택) bio_concept_lookup 도구는 단순 정의 dict로만 처리 (확장 포인트)
BIO_CONCEPTS = {
    "confluence": "Fraction of surface area covered by adherent cells, often expressed as percent (e.g., 80% confluence).",
    "plating density": "Number of cells per surface area or per well when seeding cells.",
    "dilution": "Ratio of solute to solvent or stock to final volume (e.g., 1:100).",
}


def bio_concept_lookup(term: str) -> str:
    if not term:
        return ""
    t = term.lower()
    for key, desc in BIO_CONCEPTS.items():
        if key in t:
            return desc
    return ""


# -----------------------
# ReAct 프롬프트 템플릿
# -----------------------

SYSTEM_PROMPT = """
You are a ReAct-style verifier for parameters in biology protocols.

You must follow this interaction pattern STRICTLY:

Thought: <your reasoning about what to do next>
Action: <one of: doc_search, unit_normalize, bio_concept_lookup, finish>
Action Input: <JSON for the action>

Then you will receive:

Observation: <tool output>

You may repeat Thought→Action→Action Input→Observation multiple times.
When you are ready to answer, use:

Thought: <final reasoning>
Action: finish
Action Input: {}

and then output:

Final Answer: {"verdict": "...", "evidence_span": "..."}

where verdict ∈ {"supported", "ambiguous", "unsupported"} and evidence_span is
a short quote from the Methods (or "" if unsupported).

Tools semantics:

1) doc_search
   - Input JSON: {"query": string, "top_k": int}
   - The query can be the raw parameter string, its name, value, unit, etc.
   - The tool returns a list of sentences from the Methods that seem relevant.

2) unit_normalize
   - Input JSON: {"value": <number or string>, "unit": <string or null>}
   - The tool returns a JSON with a normalized unit/value, e.g. {"value": 130, "unit": "xg"}

3) bio_concept_lookup
   - Input JSON: {"term": string}
   - Returns a short definition of the biological concept (e.g., confluence, dilution).

You MUST think step by step and decide yourself which tool to use first.
Do not hallucinate evidence: if Methods do not clearly support the parameter, answer "unsupported".
If evidence is related but approximate or partially specified, answer "ambiguous".
"""


def build_initial_user_message(param: ParamRecord, methods_text: str) -> str:
    summary = {
        "parameter": {
            "name": param.name,
            "value": param.value,
            "unit": param.unit,
            "raw": param.raw,
            "step_id": param.step_id,
            "task_ref": param.task_ref,
            "node_title": param.node_title,
            "node_type": param.node_type,
        },
        "instruction": (
            "You are verifying whether this parameter is supported by the Methods text. "
            "Use the tools to search the Methods and, if helpful, normalize units or look up biological concepts. "
            "The Methods text is available to tools but NOT directly printed here to save space."
        ),
        "hint": "Start by using doc_search with a query based on the raw parameter string or its name.",
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)


# -----------------------
# LLM 호출 & ReAct 루프
# -----------------------

def call_llm(messages: List[Dict[str, str]], client: Any, model: str,
             max_retries: int = 3, sleep_sec: float = 1.0) -> str:
    last_err = None
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            time.sleep(sleep_sec)
    # 실패 시
    return f"ERROR: {last_err}"


ACTION_RE = re.compile(r"Action:\s*(\w+)\s*", re.IGNORECASE)
ACTION_INPUT_RE = re.compile(r"Action Input:\s*(\{.*\})", re.IGNORECASE | re.DOTALL)
FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(\{.*\})", re.IGNORECASE | re.DOTALL)


def parse_action_and_input(text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    LLM 응답에서 마지막 Action / Action Input 블록을 파싱한다.
    """
    action_match = None
    for m in ACTION_RE.finditer(text):
        action_match = m
    if not action_match:
        return None, None
    action = action_match.group(1).strip()

    action_input_match = None
    for m in ACTION_INPUT_RE.finditer(text):
        action_input_match = m
    if not action_input_match:
        return action, {}

    raw_json = action_input_match.group(1)
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        data = {}
    return action, data


def parse_final_answer(text: str) -> Optional[Dict[str, Any]]:
    m = FINAL_ANSWER_RE.search(text)
    if not m:
        return None
    raw_json = m.group(1)
    try:
        obj = json.loads(raw_json)
    except json.JSONDecodeError:
        return None
    verdict = obj.get("verdict", "ambiguous")
    evidence_span = obj.get("evidence_span", "")
    if verdict not in ("supported", "ambiguous", "unsupported"):
        verdict = "ambiguous"
    return {
        "verdict": verdict,
        "evidence_span": evidence_span,
        "raw": obj,
    }


def react_single_param(methods_text: str, param: ParamRecord, client: Any, model: str,
                       max_steps: int = 5) -> Dict[str, Any]:
    """
    하나의 파라미터에 대해 ReAct 루프를 실행.
    Thought / Action / Observation / ... / Final Answer 전체 trace를 반환.
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_initial_user_message(param, methods_text),
        },
    ]
    trace: List[Dict[str, Any]] = []

    for step_idx in range(max_steps):
        # 1) LLM에게 Thought/Action/Action Input 생성 요청
        content = call_llm(messages, client, model)
        trace.append({"role": "assistant", "step": step_idx, "content": content})
        messages.append({"role": "assistant", "content": content})

        # 2) Final Answer가 포함되어 있는지 확인
        final = parse_final_answer(content)
        if final is not None:
            return {
                **asdict(param),
                "verdict": final["verdict"],
                "evidence_span": final["evidence_span"],
                "llm_raw": final["raw"],
                "trace": trace,
            }

        # 3) Action / Action Input 파싱
        action, action_input = parse_action_and_input(content)
        if not action:
            # Action이 없으면 더 이상 진행 불가 → ambiguous로 정리
            return {
                **asdict(param),
                "verdict": "ambiguous",
                "evidence_span": "",
                "llm_raw": {"error": "no_action"},
                "trace": trace,
            }

        action = action.lower()

        # 4) finish 액션이면 종료 요청인데 Final Answer가 없으면 ambiguous
        if action == "finish":
            return {
                **asdict(param),
                "verdict": "ambiguous",
                "evidence_span": "",
                "llm_raw": {"error": "finish_without_final_answer"},
                "trace": trace,
            }

        # 5) 해당 도구 실행 → Observation 생성
        if action == "doc_search":
            query = action_input.get("query")
            top_k = int(action_input.get("top_k", 5))
            if not query:
                # query가 없으면 raw/name/unit 기반으로 기본 쿼리 구성
                q_parts = [param.raw or "", param.name or ""]
                if param.unit:
                    q_parts.append(str(param.unit))
                query = " ".join([q for q in q_parts if q]).strip()
            hits = doc_search(methods_text, query, top_k=top_k)
            observation = {"tool": "doc_search", "query": query, "hits": hits}
        elif action == "unit_normalize":
            value = action_input.get("value", param.value)
            unit = action_input.get("unit", param.unit)
            norm_v, norm_u = unit_normalize(value, unit)
            observation = {
                "tool": "unit_normalize",
                "input": {"value": value, "unit": unit},
                "normalized": {"value": norm_v, "unit": norm_u},
            }
        elif action == "bio_concept_lookup":
            term = action_input.get("term") or param.name or ""
            desc = bio_concept_lookup(term)
            observation = {"tool": "bio_concept_lookup", "term": term, "definition": desc}
        else:
            observation = {"tool": action, "error": "unknown_action"}

        # trace 및 messages에 Observation 추가
        trace.append({"role": "tool", "step": step_idx, "observation": observation})
        messages.append(
            {
                "role": "user",
                "content": f"Observation: {json.dumps(observation, ensure_ascii=False)}",
            }
        )

    # max_steps 도달 시 종료
    return {
        **asdict(param),
        "verdict": "ambiguous",
        "evidence_span": "",
        "llm_raw": {"error": "max_steps_reached"},
        "trace": trace,
    }


# -----------------------
# 메인: 전체 IR에 대한 실행
# -----------------------

def run_react_verification_on_ir(ir_path: Path, out_path: Path, client: Any, model: str,
                                 max_protocols: int = 0, max_params: int = 0) -> None:
    print(f"[INFO] Loading IR from {ir_path}")
    ir_records = load_jsonl(ir_path)
    print(f"[INFO] Loaded {len(ir_records)} IR records.")

    results: List[Dict[str, Any]] = []
    total_params = 0

    for idx, rec in enumerate(ir_records):
        protocol_id = rec.get("protocol_id")
        methods_text = rec.get("methods_text", "")
        if not protocol_id:
            print("[WARN] Missing protocol_id in IR record, skipping.")
            continue

        param_records = flatten_params_from_ir_record(rec)
        if not param_records:
            print(f"[INFO] No params in protocol {protocol_id}, skipping.")
            continue

        print(f"[INFO] Protocol {protocol_id}: {len(param_records)} params")

        for p in param_records:
            if max_params and total_params >= max_params:
                print(f"[INFO] Reached max_params={max_params}, stopping.")
                break
            res = react_single_param(methods_text, p, client, model)
            results.append(res)
            total_params += 1

        if max_params and total_params >= max_params:
            break

        if max_protocols and (idx + 1) >= max_protocols:
            print(f"[INFO] Reached max_protocols={max_protocols}, stopping.")
            break

    print(f"[INFO] Total parameters processed: {total_params}")
    print(f"[INFO] Saving results to {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(out_path, results)
    print("[INFO] Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Full ReAct-style verifier on IR graphs from steps (nodes+params)."
    )
    parser.add_argument(
        "--ir",
        type=str,
        default="runs/ir_graphs_from_steps_llm.jsonl",
        help="IR JSONL file with protocol_id, methods_text, nodes, edges, ...",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="runs/react_verifier_full_case.jsonl",
        help="Output JSONL file for ReAct parameter verdicts.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI model name to use.",
    )
    parser.add_argument(
        "--max-protocols",
        type=int,
        default=0,
        help="Limit number of protocols to process (0 = no limit).",
    )
    parser.add_argument(
        "--max-params",
        type=int,
        default=0,
        help="Limit total number of parameters to process (0 = no limit).",
    )

    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("ERROR: OPENAI_API_KEY environment variable is not set.")
    if OpenAI is None:
        raise SystemExit("ERROR: openai package is not installed. Run `pip install openai`.")

    client = OpenAI()

    run_react_verification_on_ir(
        Path(args.ir),
        Path(args.out),
        client,
        args.model,
        max_protocols=args.max_protocols,
        max_params=args.max_params,
    )


if __name__ == "__main__":
    main()
