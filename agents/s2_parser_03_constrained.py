#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S2 Parser (Constrained, v2):
- Goal: Methods 텍스트를 IR(Graph)로 구조화. S2A(Task Miner) 결과를 '최소 제약'으로 걸어
        각 task_id 당 최소 N개의 Step 노드가 생성되도록 강제.
- Node types: Step, QCGate, DataAnalysis  (1급 시민)
- Edges: {from, to, label}  (label ∈ {'then','on_accept','on_reject'})

Input:
  --pairs  : data/gold/gold_pairs_testset.jsonl  (필수: protocol_id, sec_text, pmcid, domain)
  --tasks  : runs/s2a_tasks_llm.methods.jsonl     (필수: protocol_id, task_id, title, description, key_materials, goal)
Output:
  --out    : runs/s2_parser.ir.jsonl              (프로토콜별 1라인 JSON)
Options:
  --model, --min-steps-per-task, --max-steps-per-task, --max-nodes, --min-chars, --temperature

Env:
  OPENAI_API_KEY must be set.

Why (이유):
- 그래프 IR은 S3/S4 단계에서 절차 복원(DAG)과 검증(CoV)에 유리.
- 파라미터를 표준 스키마(name/value/unit/raw/source)로 강제하면
  downstream(grounder, checker, writer)의 파싱·정규화 비용과 오류를 줄임.

Refs:
- OpenAI Chat Completions JSON mode (response_format=json_object)
  https://platform.openai.com/docs/guides/structured-outputs
- JSON Lines 포맷 개요
  https://jsonlines.org/
"""
import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple


# ---------- sentence split ----------
def split_sentences(text: str) -> List[str]:
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw: return []
    out, cur = [], []
    for tok in re.split(r"(\.|\?|\!)", raw):
        if tok is None: continue
        cur.append(tok)
        if tok in (".", "?", "!"):
            s = "".join(cur).strip()
            if s: out.append(s)
            cur = []
    rest = "".join(cur).strip()
    if rest: out.append(rest)
    return [s for s in out if s]


# ---------- normalize helpers ----------
_NUM_UNIT_RX = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([^\s]+)?\s*$")


def _to_float_or_none(x: Any) -> Any:
    try:
        if x is None: return None
        s = str(x).strip()
        if s == "": return None
        return float(s)
    except:
        return None


def _parse_value_unit(s: str) -> Tuple[Any, Any]:
    m = _NUM_UNIT_RX.match(s.strip())
    if not m: return None, None
    val = _to_float_or_none(m.group(1))
    unit = (m.group(2) or "").strip() or None
    return val, unit


def _normalize_param_list(obj: Any, source="parser") -> List[Dict[str, Any]]:
    """
    표준 스키마로 변환:
    [{"name": str, "value": number|null, "unit": str|null, "raw": str|null, "source": "parser"}]
    허용 입력:
      - list[str]
      - list[dict] (단일키-딕셔너리 포함)
      - dict (name->"10 ng/ml" 형태)
      - None / "" -> []
    """
    if obj is None or obj == "": return []
    out: List[Dict[str, Any]] = []

    def push(name, value, unit, raw):
        out.append({
            "name": (name or "").strip(),
            "value": value,
            "unit": unit,
            "raw": (raw or None),
            "source": source
        })

    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                # 단일키 딕셔너리 {"temp": "37 C"} or {"name":"temp","value":"37","unit":"C"}
                if set(it.keys()) & {"name", "value", "unit", "raw"}:
                    name = it.get("name")
                    val = _to_float_or_none(it.get("value"))
                    unit = (it.get("unit") or None)
                    raw = it.get("raw")
                    push(name, val, unit, raw)
                elif len(it.keys()) == 1:
                    k, v = list(it.items())[0]
                    if isinstance(v, str):
                        val, unit = _parse_value_unit(v)
                        push(k, val, unit, v)
                    else:
                        push(k, None, None, str(v))
                else:
                    # 복합 dict는 raw로 보존
                    push(json.dumps(it, ensure_ascii=False), None, None, json.dumps(it, ensure_ascii=False))
            elif isinstance(it, str):
                s = it.strip()
                # "temp: 37 C" / "time=5 min" / "speed 9700 g" 등
                m = re.match(r"^([^:=]+)[:=\s]\s*(.+)$", s)
                if m:
                    name = m.group(1).strip()
                    rest = m.group(2).strip()
                    val, unit = _parse_value_unit(rest)
                    push(name, val, unit, s)
                else:
                    push(s, None, None, s)
            else:
                push(str(it), None, None, str(it))
        return out

    if isinstance(obj, dict):
        # {"temp": "37 C", "time": "5 min"} 스타일
        for k, v in obj.items():
            if isinstance(v, dict) and ("value" in v or "unit" in v):
                push(k, _to_float_or_none(v.get("value")), v.get("unit"), json.dumps(v, ensure_ascii=False))
            elif isinstance(v, str):
                val, unit = _parse_value_unit(v)
                push(k, val, unit, v)
            else:
                push(k, None, None, str(v))
        return out

    if isinstance(obj, str):
        parts = [p.strip() for p in re.split(r"[;,]\s*", obj) if p.strip()]
        for s in parts:
            m = re.match(r"^([^:=]+)[:=\s]\s*(.+)$", s)
            if m:
                name = m.group(1).strip()
                rest = m.group(2).strip()
                val, unit = _parse_value_unit(rest)
                push(name, val, unit, s)
            else:
                push(s, None, None, s)
        return out

    # 기타 타입
    push(str(obj), None, None, str(obj))
    return out


# ---------- LLM ----------
SYS_PROMPT = (
    "You are a meticulous protocol structurer. Convert the given Methods into a graph IR.\n"
    "STRICT OUTPUT JSON KEYS: nodes (array), edges (array), warnings (array).\n"
    "NODE TYPES (use exact labels): Step, QCGate, DataAnalysis.\n"
    "\n"
    "Step schema:\n"
    "{id, type='Step', title, action, materials: string[], params: Param[], produces: string[], task_ref}\n"
    "Param schema (MANDATORY):\n"
    "{name: string, value: number|null, unit: string|null, raw: string|null, source: 'parser'}\n"
    "\n"
    "QCGate schema:\n"
    "{id, type='QCGate',\n"
    " measurement: {what, method, units},\n"
    " acceptance_criteria: {operator, lower, upper, unit} | null,\n"
    " decision: {on_accept, on_reject, max_retries},\n"
    " fallback: string|null,\n"
    " evidence_hint: {sent_idx: number[]}\n"
    "}\n"
    "\n"
    "DataAnalysis schema:\n"
    "{id, type='DataAnalysis', method, inputs: string[], params: Param[], outputs: string[], task_ref}\n"
    "\n"
    "Edges: {from, to, label} with label in {'then','on_accept','on_reject'}.\n"
    "\n"
    "CONSTRAINTS:\n"
    "- You are given TASKS (task_id, title, description, key_materials). For EACH task, create BETWEEN MIN_STEPS and MAX_STEPS Step nodes (unless Methods text is empty). Attach task_ref.\n"
    "- If measurement/criteria/'until'/'proceed if' statements exist, create QCGate nodes.\n"
    "- If analysis/quantification/statistics instructions exist, create DataAnalysis nodes.\n"
    "- Use ONLY explicit info from Methods sentences; DO NOT invent values. If a field is missing set null or empty.\n"
    "- Node IDs must be unique and short like S1,S2,Q1,D1. Keep total nodes <= MAX_NODES, prefer Steps linked to tasks.\n"
    "- params MUST follow the Param schema above. If a parameter text is non-numeric, set value=null and copy original text into raw.\n"
)


def call_llm_openai(model: str, sys_prompt: str, user_payload: Dict, temperature: float, max_retries=3) -> Dict:
    # Docs: https://platform.openai.com/docs/guides/structured-outputs
    from openai import OpenAI
    client = OpenAI()
    last = None
    for i in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
                ],
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            last = e
            time.sleep(1.1 * (i + 1))
    raise RuntimeError(f"LLM call failed: {last}")


# ---------- IO ----------
def load_pairs(path: str) -> Dict[str, Dict[str, Any]]:
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except:
                continue
            pid = r.get("protocol_id")
            if pid: m[pid] = r
    return m


def load_tasks(path: str) -> Dict[str, List[Dict[str, Any]]]:
    by = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except:
                continue
            pid = r.get("protocol_id")
            if not pid: continue
            by.setdefault(pid, []).append(r)
    for k in by.keys():
        by[k].sort(key=lambda x: str(x.get("task_id")))
    return by


# ---------- post-process & enforcement ----------
def ensure_unique_ids(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    used = set();
    s = q = d = 1
    for n in nodes:
        nid = n.get("id")
        if (not nid) or (nid in used):
            t = n.get("type")
            if t == "Step":
                nid = f"S{s}"; s += 1
            elif t == "QCGate":
                nid = f"Q{q}"; q += 1
            else:
                nid = f"D{d}"; d += 1
            n["id"] = nid
        used.add(nid)
    return nodes


def normalize_node_fields(n: Dict[str, Any]) -> Dict[str, Any]:
    t = n.get("type", "Step")
    if t not in ("Step", "QCGate", "DataAnalysis"):
        t = "Step"
    n["type"] = t

    if t == "Step":
        n.setdefault("title", "")
        n.setdefault("action", None)
        n["materials"] = n.get("materials") or []
        n["params"] = _normalize_param_list(n.get("params"), source="parser")
        n["produces"] = n.get("produces") or []
        n.setdefault("task_ref", None)
    elif t == "QCGate":
        n.setdefault("measurement", {"what": None, "method": None, "units": None})
        ac = n.get("acceptance_criteria")
        if ac is not None and not isinstance(ac, dict):
            # 비정형이면 버림
            n["acceptance_criteria"] = None
        n.setdefault("decision", {"on_accept": None, "on_reject": None, "max_retries": 0})
        n.setdefault("fallback", None)
        eh = n.get("evidence_hint")
        if not isinstance(eh, dict): eh = {}
        if not isinstance(eh.get("sent_idx"), list): eh["sent_idx"] = []
        n["evidence_hint"] = eh
    else:
        n.setdefault("method", None)
        n["inputs"] = n.get("inputs") or []
        n["params"] = _normalize_param_list(n.get("params"), source="parser")
        n["outputs"] = n.get("outputs") or []
        n.setdefault("task_ref", None)
    return n


def enforce_min_steps_per_task(nodes: List[Dict[str, Any]], tasks: List[Dict[str, Any]],
                               min_steps: int, max_nodes: int, warnings: List[str]) -> List[Dict[str, Any]]:
    by = {}
    for n in nodes:
        if n.get("type") == "Step":
            by.setdefault(n.get("task_ref"), []).append(n)

    def synth_placeholder_step(task, idx):
        title = f"{(task.get('title') or 'Task')} — placeholder {idx}"
        mats = [str(m) for m in (task.get("key_materials") or [])[:4]]
        return {"id": None, "type": "Step", "title": title[:120], "action": "unspecified",
                "materials": mats, "params": [], "produces": [], "task_ref": task.get("task_id")}

    for t in tasks:
        tref = t.get("task_id")
        arr = by.get(tref, [])
        need = max(0, min_steps - len(arr))
        for i in range(need):
            if len(nodes) >= max_nodes:
                warnings.append(f"max_nodes reached; cannot add placeholder for {tref}")
                break
            ph = synth_placeholder_step(t, len(arr) + i + 1)
            nodes.append(ph);
            arr.append(ph)
            warnings.append(f"added_placeholder_step for {tref}")
        by[tref] = arr
    return nodes


def chain_edges_if_missing(edges: List[Dict[str, Any]], nodes: List[Dict[str, Any]]):
    have = {(e.get("from"), e.get("to")) for e in edges}
    by_task = {}
    for n in nodes:
        if n.get("type") != "Step": continue
        tref = n.get("task_ref") or "_no_task"
        by_task.setdefault(tref, []).append(n)

    def keyf(n):
        m = re.search(r"(\d+)$", n.get("id") or "")
        return int(m.group(1)) if m else 99999

    for tref, arr in by_task.items():
        arr.sort(key=keyf)
        for a, b in zip(arr, arr[1:]):
            if (a.get("id"), b.get("id")) not in have:
                edges.append({"from": a.get("id"), "to": b.get("id"), "label": "then"})
                have.add((a.get("id"), b.get("id")))
    return edges


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--min-steps-per-task", type=int, default=2)
    ap.add_argument("--max-steps-per-task", type=int, default=6)
    ap.add_argument("--max-nodes", type=int, default=26)
    ap.add_argument("--min-chars", type=int, default=160)
    ap.add_argument("--sent-max", type=int, default=120)
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("ERROR: OPENAI_API_KEY not set")

    pairs = load_pairs(args.pairs)
    tasks_map = load_tasks(args.tasks)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out = open(args.out, "w", encoding="utf-8")

    for pid, prec in pairs.items():
        methods = (prec.get("sec_text") or "").strip()
        if len(methods) < args.min_chars:
            continue
        tlist = tasks_map.get(pid, [])

        sents = split_sentences(methods)[:args.sent_max]
        payload = {
            "protocol_id": pid,
            "pmcid": prec.get("pmcid"),
            "domain": prec.get("domain"),
            "methods_text": methods[:14000],
            "sentences": [{"idx": i, "text": s} for i, s in enumerate(sents)],
            "tasks": [
                {
                    "task_id": t.get("task_id"),
                    "title": t.get("title"),
                    "description": t.get("description"),
                    "key_materials": t.get("key_materials") or [],
                    "goal": t.get("goal")
                } for t in tlist
            ],
            "constraints": {
                "MIN_STEPS": args.min_steps_per_task,
                "MAX_STEPS": args.max_steps_per_task,
                "MAX_NODES": args.max_nodes
            }
        }

        try:
            js = call_llm_openai(args.model, SYS_PROMPT, payload, args.temperature)
        except Exception as e:
            js = {"nodes": [], "edges": [], "warnings": [f"llm_error: {str(e)}"]}

        nodes = js.get("nodes", [])
        edges = js.get("edges", [])
        warnings = list(js.get("warnings", []))

        # 1) 기본 필드/타입 표준화 + params 스키마 강제
        nodes = [normalize_node_fields(n) for n in nodes]
        nodes = ensure_unique_ids(nodes)

        # 2) 태스크별 최소 스텝 수 강제
        nodes = enforce_min_steps_per_task(nodes, tlist, args.min_steps_per_task, args.max_nodes, warnings)

        # 3) 엣지 보강(태스크 내부 체이닝)
        edges = chain_edges_if_missing(edges, nodes)

        # 4) 노드 수 하드 클램프
        if len(nodes) > args.max_nodes:
            warnings.append(f"node_overflow: {len(nodes)} > {args.max_nodes}, tail_dropped")
            nodes = nodes[:args.max_nodes]

        # 5) 간단 검증: 모든 Step.params는 표준 스키마여야 함
        bad = []
        for n in nodes:
            if n.get("type") == "Step":
                ok = all(isinstance(p, dict) and {"name", "value", "unit", "raw", "source"} <= set(p.keys())
                         for p in (n.get("params") or []))
                if not ok: bad.append(n.get("id"))
        if bad:
            warnings.append(f"nonstandard_params_in_steps: {bad}")

        out.write(json.dumps({
            "protocol_id": pid,
            "pmcid": prec.get("pmcid"),
            "domain": prec.get("domain"),
            "nodes": nodes,
            "edges": edges,
            "warnings": warnings,
            "task_stats": {
                "n_tasks": len(tlist),
                "min_steps_per_task": args.min_steps_per_task,
                "max_steps_per_task": args.max_steps_per_task,
                "max_nodes": args.max_nodes
            }
        }, ensure_ascii=False) + "\n")

    out.close()
    print(f"[OK] constrained IR v2 -> {args.out}")


if __name__ == "__main__":
    main()
