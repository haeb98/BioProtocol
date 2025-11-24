#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
S2 Parser (LLM Structured) — robust sections normalization + JSON mode
- Input: data/gold/gold_articles_sections_pmc.jsonl
- Filter: CSV of protocol_id (optional)
- Output: IR JSONL
- Uses OPENAI_API_KEY from environment by default; can override via --api-key/--api-base

Run:
  python agents/s2_parser_01_llm_structured.py \
    --arts data/gold/gold_articles_sections_pmc.jsonl \
    --filter-ids data/splits/test_biop_ids_top15.csv \
    --out runs/s2_llm_top15.ir.jsonl \
    --model gpt-4o-mini
"""

import argparse
import json
import os
import re
import time
from typing import Dict, Any, List, Iterable, Union

from openai import OpenAI, APIConnectionError, RateLimitError, BadRequestError, AuthenticationError


# ---------- IO utils ----------
def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_id_set(csv_path: str) -> set:
    if not csv_path:
        return set()
    ids = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            pid = line.strip().split(",")[0]
            if pid and pid != "protocol_id":
                ids.add(pid)
    return ids


# ---------- Sections normalization ----------
METHOD_LIKE = re.compile(
    r"(method|materials\s*&?\s*methods|experimental\s+procedur|star\s*methods|methodology)",
    re.I
)


def normalize_sections(secs: Union[List, Dict, str]) -> List[Dict[str, Any]]:
    """
    Normalize various legacy shapes to: List[ {title,text,title_norm,is_target,sec_type?} ]
    Accepts:
      - list[str]  -> wrap as {"title":"", "text": item}
      - list[dict] -> pass-through with defaults
      - dict       -> assume keys are titles, values are text/dict
      - str        -> single blob
    """
    out: List[Dict[str, Any]] = []
    if secs is None:
        return out

    if isinstance(secs, str):
        t = secs.strip()
        if t:
            out.append({
                "title": "",
                "text": t,
                "title_norm": "",
                "is_target": bool(METHOD_LIKE.search(t[:200])),
            })
        return out

    if isinstance(secs, list):
        for item in secs:
            if isinstance(item, dict):
                title = (item.get("title") or "").strip()
                text = (item.get("text") or "").strip()
                sec_type = (item.get("sec_type") or "").strip()
                tnorm = (item.get("title_norm") or title).strip().lower()
                is_target = bool(item.get("is_target")) or bool(METHOD_LIKE.search(title)) or bool(
                    METHOD_LIKE.search(tnorm)) or (sec_type.lower() in {"methods", "materials|methods"})
                if text:
                    out.append({
                        "title": title, "text": text,
                        "title_norm": tnorm, "is_target": is_target,
                        **({"sec_type": sec_type} if sec_type else {})
                    })
            elif isinstance(item, str):
                txt = item.strip()
                if txt:
                    out.append({
                        "title": "", "text": txt,
                        "title_norm": "", "is_target": bool(METHOD_LIKE.search(txt[:200]))
                    })
        return out

    if isinstance(secs, dict):
        for k, v in secs.items():
            title = (k or "").strip()
            if isinstance(v, dict):
                text = (v.get("text") or "").strip()
                sec_type = (v.get("sec_type") or "").strip()
            else:
                text = (str(v) or "").strip()
                sec_type = ""
            if not text:
                continue
            tnorm = title.lower()
            is_target = bool(METHOD_LIKE.search(title)) or (sec_type.lower() in {"methods", "materials|methods"})
            out.append({
                "title": title, "text": text,
                "title_norm": tnorm, "is_target": is_target,
                **({"sec_type": sec_type} if sec_type else {})
            })
        return out

    # Unknown type -> ignore
    return out


# ---------- Prompt ----------
SYSTEM_PROMPT = (
    "You are an information extraction system that MUST return STRICT JSON. "
    "Extract an IR of a laboratory protocol from the given Methods-like text. "
    "Return ONLY JSON (no extra text)."
)

USER_PROMPT_TEMPLATE = """\
Task: Convert the following scientific 'Methods' text into a structured IR.

IR schema (JSON):
{{
  "protocol_id": "<Bio-protocol-*>",
  "article_id": "<PMCID or internal id>",
  "domain": "<domain string>",
  "steps": [
    {{
      "step_id": "S<number>",
      "action": "<verb/action>",
      "materials": ["<material-1>", "<material-2>", "..."],
      "parameters": [
        {{"name":"time","value":"<e.g., 30>","unit":"min","source_span":"<text snippet>"}},
        {{"name":"temperature","value":"<e.g., 37>","unit":"C"}}
      ],
      "tools": ["<tool-1>", "..."],
      "conditions": ["<e.g., dark, sterile>"],
      "dependencies": ["<S<number>>"],
      "evidence_sent_ids": [<indices of evidence sentences>]
    }}
  ]
}}

Constraints:
- Output MUST be valid JSON (single object).
- Use null or empty list for unknown fields; do not hallucinate.
- Keep numeric values separate from units when obvious (e.g., 30 + min).
- Provide evidence_sent_ids to support each step.

INPUT META:
protocol_id={protocol_id}
article_id={article_id}
domain={domain}

METHODS TEXT:
\"\"\"
{methods_text}
\"\"\""""


# ---------- LLM ----------
def build_client(args) -> OpenAI:
    if args.api_key and args.api_base:
        return OpenAI(api_key=args.api_key, base_url=args.api_base)
    elif args.api_key:
        return OpenAI(api_key=args.api_key)
    elif args.api_base:
        return OpenAI(base_url=args.api_base)
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set. export OPENAI_API_KEY=sk-...")
        return OpenAI()


def call_llm_json(client: OpenAI, model: str, system: str, user: str, max_retries: int = 2) -> Dict[str, Any]:
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                # JSON mode -> valid JSON 보장(스키마 일치까지는 아님)
                response_format={"type": "json_object"},  # json_schema는 gpt-4o(2024-08-06+)/4o-mini에서만 동작
                temperature=0.2,
            )
            text = resp.choices[0].message.content or "{}"
            return json.loads(text)
        except (BadRequestError, APIConnectionError, RateLimitError, AuthenticationError) as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
        except json.JSONDecodeError as e:
            last_err = e
            time.sleep(1.0)
    raise RuntimeError(f"LLM call failed after retries: {last_err}")


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arts", required=True, help="gold_articles_sections_pmc.jsonl")
    ap.add_argument("--filter-ids", default="", help="CSV of protocol_id to run (header: protocol_id)")
    ap.add_argument("--out", required=True, help="output JSONL for IR")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (supports JSON mode)")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--max-retries", type=int, default=2)
    args = ap.parse_args()

    client = build_client(args)
    idset = load_id_set(args.filter_ids)

    total, picked = 0, 0
    out_rows: List[Dict[str, Any]] = []

    for rec in read_jsonl(args.arts):
        total += 1
        pid = rec.get("protocol_id") or ""
        if idset and pid not in idset:
            continue

        # 표준화된 섹션 확보
        sections_norm = normalize_sections(rec.get("sections", []))
        if not sections_norm:
            continue

        # Methods 계열 우선 선택: is_target=True, sec_type=methods, title에 method류 포함
        methods_blocks = [
            s for s in sections_norm
            if s.get("is_target")
               or (s.get("sec_type", "").lower() in {"methods", "materials|methods"})
               or METHOD_LIKE.search(s.get("title_norm", "") or s.get("title", "") or "")
        ]

        if not methods_blocks:
            # 최후 보루: 본문 중 가장 긴 섹션 후보 1-2개
            methods_blocks = sorted(sections_norm, key=lambda x: len(x.get("text", "")), reverse=True)[:2]

        # 텍스트 병합 및 최소 길이 필터
        methods_text = "\n\n".join(s.get("text", "").strip() for s in methods_blocks if s.get("text"))
        if not methods_text or len(methods_text) < 200:
            continue

        user_prompt = USER_PROMPT_TEMPLATE.format(
            protocol_id=pid,
            article_id=rec.get("article_id") or rec.get("pmcid") or "",
            domain=rec.get("domain", "Unknown"),
            methods_text=methods_text[:20000],
        )

        ir = call_llm_json(
            client=client, model=args.model,
            system=SYSTEM_PROMPT, user=user_prompt,
            max_retries=args.max_retries
        )
        # 최소 메타 보강
        ir.setdefault("protocol_id", pid)
        ir.setdefault("article_id", rec.get("article_id") or rec.get("pmcid") or "")
        ir.setdefault("domain", rec.get("domain", "Unknown"))

        out_rows.append(ir);
        picked += 1

    write_jsonl(args.out, out_rows)
    print(f"[OK] wrote IR -> {args.out} (total_lines={total}, picked={picked})")


if __name__ == "__main__":
    main()
