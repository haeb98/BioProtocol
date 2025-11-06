#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gold IR Builder from testset pairs (LLM-first, hierarchical-friendly)

- Input : e.g., data/gold/gold_pairs_testset_top15.jsonl
           (each line has: protocol_id, domain, article{...}, protocol{...})
- Output: e.g., runs/gold_ir_top15.llm.jsonl  (LLM 구조화 IR)
         or runs/gold_ir_top15.hier.jsonl     (계층형 규칙 파서 IR)

Design:
  1) 스키마 정규화: 다양한 protocol 필드(text/regular/input/protocol/hierarchical_protocol)를 통일해서 텍스트로 확보
  2) 기본값: hierarchical_protocol이 존재하면 **무조건 그걸 우선 사용**
  3) LLM → JSON(IR 스키마) 강제 반환(response_format=json_object)
  4) --from-hier 플래그를 쓰면 LLM 없이 계층형을 규칙으로 IR 변환(공정 비교용)
  5) 실패/스킵은 --log에 jsonl로 기록

Run (LLM 권장):
  export OPENAI_API_KEY=sk-...
  python eval/gold_ir_00_from_protocol_llm.py \
    --pairs data/gold/gold_pairs_testset_top15.jsonl \
    --out   runs/gold_ir_top15.llm.jsonl \
    --model gpt-4o-mini \
    --log   runs/gold_ir_build.log.jsonl

Run (규칙 파서 참고용):
  python eval/gold_ir_00_from_protocol_llm.py \
    --pairs data/gold/gold_pairs_testset_top15.jsonl \
    --out   runs/gold_ir_top15.hier.jsonl \
    --from-hier \
    --log   runs/gold_ir_build.log.jsonl
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Optional


# ------------------ IO helpers ------------------

def read_jsonl(p: str) -> Iterable[Dict[str, Any]]:
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(p: str, rows: Iterable[Dict[str, Any]]) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_log(log_path: Optional[str], rec: Dict[str, Any]) -> None:
    if not log_path:
        return
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ------------------ Hierarchical → IR (rule baseline) ------------------

ACTION_HINT = re.compile(
    r"\b(incubat|centrifug|vortex|transfer|add|mix|measure|wash|dry|resuspend|dilut|heat|cool|pipett|aliquot)\w*",
    re.I,
)
UNIT_HINT = re.compile(
    r"\b(°c|c|min|hr|h|s|ml|μl|ul|l|rpm|xg|%|mm|μm|um|mM|μM|uM|M|ng|μg|ug|mg|nm)\b",
    re.I,
)
NUM = re.compile(r"[-+]?\d+(\.\d+)?")


def hier_to_ir(rec: Dict[str, Any]) -> Dict[str, Any]:
    """아주 단순한 휴리스틱 규칙으로 계층형을 IR steps로 변환(참고용)."""
    proto = rec.get("protocol", {}).get("hierarchical_protocol") or {}
    steps = []

    # 키를 순서대로 보장하기 위해 숫자/부분문자 기준 정렬
    def keyer(k: str):
        return [int(p) if p.isdigit() else p for p in k.split(".")]

    for k in sorted(proto.keys(), key=keyer):
        v = proto[k]
        if isinstance(v, str):
            text = v.strip()
            # action 추정
            m = ACTION_HINT.search(text)
            action = m.group(0).lower() if m else "step"
            # parameter naive 추출
            params = []
            for um in UNIT_HINT.finditer(text):
                unit = um.group(0)
                ctx = text[max(0, um.start() - 10): um.end() + 10]
                nm = NUM.search(ctx)
                if nm:
                    params.append(
                        {
                            "name": "param",
                            "value": nm.group(0),
                            "unit": unit,
                            "source_span": ctx.strip(),
                        }
                    )
            steps.append(
                {
                    "step_id": f"S{k}",
                    "action": action,
                    "materials": [],
                    "parameters": params,
                    "tools": [],
                    "conditions": [],
                    "dependencies": [],
                    "evidence_sent_ids": [],
                }
            )
        # dict(title=...)는 섹션 헤더로 간주하고 스킵
    return {
        "protocol_id": rec.get("protocol_id", ""),
        "article_id": rec.get("article", {}).get("id", ""),
        "domain": rec.get("domain", "Unknown"),
        "steps": steps,
    }


# ------------------ Protocol text normalization ------------------

def flatten_hier(h: Dict[str, Any]) -> str:
    """hierarchical_protocol을 사람이 읽기 쉬운 순서/텍스트로 평탄화."""
    if not isinstance(h, dict):
        return ""

    def keyer(k: str):
        return [int(p) if p.isdigit() else p for p in k.split(".")]

    lines = []
    for k in sorted(h.keys(), key=keyer):
        v = h[k]
        if isinstance(v, dict) and "title" in v:
            lines.append(f"## {v['title']}")
        elif isinstance(v, str):
            lines.append(v)
    return "\n".join(lines)


def extract_protocol_text(rec: Dict[str, Any]) -> Tuple[str, str]:
    """
    가능한 모든 형태를 커버해 protocol 텍스트를 확보.
    우선순위(기본): 1) hierarchical_protocol 2) text/regular 3) title+input/protocol 4) str 직렬화
    반환: (텍스트, 소스태그)
    """
    p = rec.get("protocol")
    if not p:
        return "", "no_protocol_block"

    # 1) hierarchical_protocol 최우선
    hier = p.get("hierarchical_protocol")
    if isinstance(hier, dict) and hier:
        flat = flatten_hier(hier).strip()
        if flat:
            return flat, "protocol.hierarchical_protocol"

    # 2) 일반 텍스트 후보
    for k in ("text", "regular", "regular_text", "body", "content"):
        v = p.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip(), f"protocol.{k}"

    # 3) title + (keywords|input) + protocol 조합
    title = (p.get("title") or "").strip()
    # 일부 데이터는 'keywords'가 실질적 재료/인풋처럼 쓰인 케이스가 있어 fallback로 포함
    inp = (p.get("input") or p.get("keywords") or "").strip()
    proto = (p.get("protocol") or "").strip()
    if title or inp or proto:
        parts = []
        if title:
            parts.append(title)
        if inp:
            parts.append("# Materials & Inputs\n" + inp)
        if proto:
            parts.append("# Protocol\n" + proto)
        txt = "\n\n".join(parts).strip()
        if txt:
            return txt, "protocol.title+input+protocol"

    # 4) 문자열 블록
    if isinstance(p, str) and p.strip():
        return p.strip(), "protocol.str"

    return "", "protocol_text_not_found"


# ------------------ LLM client ------------------

def build_client(api_key: Optional[str] = None, api_base: Optional[str] = None):
    """
    OpenAI 호환 클라이언트. 환경변수 OPENAI_API_KEY 필수.
    커스텀 엔드포인트 사용시 --api-base 지정.
    """
    from openai import OpenAI
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    if api_base:
        return OpenAI(api_key=key, base_url=api_base)
    return OpenAI(api_key=key)


SYS = "You are an expert information extractor that MUST return STRICT JSON only."
USR_TMPL = """Convert the following protocol text into a structured IR JSON.

Schema:
{{
  "protocol_id": "<id>",
  "article_id": "<pmcid>",
  "domain": "<domain>",
  "steps": [
    {{
      "step_id": "S<number>",
      "action": "<verb>",
      "materials": ["..."],
      "parameters": [{{"name": "time|temp|volume|speed|conc|...", "value": "<num or str>", "unit": "<unit>", "source_span": "<snippet>"}}],
      "tools": ["..."],
      "conditions": ["..."],
      "dependencies": ["S<number>"],
      "evidence_sent_ids": [0,1,2]
    }}
  ]
}}

Rules:
- STRICT JSON only (no markdown).
- Use null/empty when not explicit. Do not invent extra facts.
- Preserve numeric values and units as written when possible.

META: protocol_id={pid} | article_id={aid} | domain={dom}

TEXT:
\"\"\" 
{txt}
\"\"\""""


def llm_protocol_to_ir(client, model: str, rec: Dict[str, Any], max_retries: int = 2) -> Dict[str, Any]:
    txt, src = extract_protocol_text(rec)
    if not txt or len(txt) < 50:
        raise ValueError(f"protocol text missing/too short (src={src})")

    pid = rec.get("protocol_id", "")
    aid = rec.get("article", {}).get("id", "")
    dom = rec.get("domain", "Unknown")

    prompt = USR_TMPL.format(pid=pid, aid=aid, dom=dom, txt=txt[:20000])
    last = None
    for t in range(max_retries + 1):
        try:
            rsp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            out = rsp.choices[0].message.content or "{}"
            js = json.loads(out)
            js.setdefault("protocol_id", pid)
            js.setdefault("article_id", aid)
            js.setdefault("domain", dom)
            js["_source_protocol_text"] = src
            return js
        except Exception as e:
            last = e
            time.sleep(1.2 * (t + 1))
    raise RuntimeError(f"LLM IR build failed: {last}")


# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="gold_pairs_testset_*.jsonl")
    ap.add_argument("--out", required=True, help="output IR jsonl")
    ap.add_argument("--from-hier", action="store_true",
                    help="LLM 없이 hierarchical_protocol을 규칙 파서로 IR 변환(공정 비교용)")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--log", default=None, help="jsonl 로그 경로")
    ap.add_argument("--max-retries", type=int, default=2)
    args = ap.parse_args()

    rows = []
    kept = 0
    skipped = 0

    if args.from_hier:
        # 규칙 파서 변환
        for rec in read_jsonl(args.pairs):
            try:
                ir = hier_to_ir(rec)
                rows.append(ir);
                kept += 1
            except Exception as e:
                skipped += 1
                write_log(args.log, {
                    "protocol_id": rec.get("protocol_id", ""),
                    "reason": f"hier-parse-failed: {str(e)[:300]}",
                })
    else:
        # LLM 변환
        client = build_client(args.api_key, args.api_base)
        for rec in read_jsonl(args.pairs):
            try:
                ir = llm_protocol_to_ir(client, args.model, rec, max_retries=args.max_retries)
                rows.append(ir);
                kept += 1
            except Exception as e:
                skipped += 1
                write_log(args.log, {
                    "protocol_id": rec.get("protocol_id", ""),
                    "reason": f"llm-failed: {str(e)[:300]}",
                })

    write_jsonl(args.out, rows)
    print(f"[OK] wrote -> {args.out} (rows={len(rows)}, kept={kept}, skipped={skipped})")
    if args.log:
        print(f"[LOG] {args.log}")


if __name__ == "__main__":
    main()
