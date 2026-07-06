#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
[Writer-only Generator | Chat Completions + dotenv]

왜 이 스크립트가 필요한가(이유)
- 이미 생성된 IR(gen_actions_P1~P6)을 재사용하여 앞단(Methods→IR 등)을 다시 돌리지 않기 위해.
- 평가를 위해 "액션 1개당 문장 1개" 정렬을 강제해야 하므로,
  LLM 출력 포맷을 JSON array of strings로 고정한다.

공식 문서 근거(링크/출처 태그)
- python-dotenv: https://pypi.org/project/python-dotenv/
- OpenAI Python SDK (Chat Completions 사용): https://github.com/openai/openai-python
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # .env에서 OPENAI_API_KEY 로드


# -------------------------
# IO
# -------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -------------------------
# Parsing helpers
# -------------------------
def parse_json_array_only(text: str) -> List[str]:
    """
    LLM이 JSON array만 반환하도록 강제하지만, 혹시 실패했을 때를 대비해
    [ ... ] 구간만 떼어 파싱하는 fallback도 포함.
    """
    text = (text or "").strip()
    if not text:
        return []

    # 1) direct JSON parse
    try:
        obj = json.loads(text)
        if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
            return [x.strip() for x in obj]
    except Exception:
        pass

    # 2) extract bracketed part
    s = text.find("[")
    e = text.rfind("]")
    if s != -1 and e != -1 and e > s:
        try:
            obj = json.loads(text[s: e + 1])
            if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
                return [x.strip() for x in obj]
        except Exception:
            return []
    return []


# -------------------------
# LLM call
# -------------------------
def build_prompt(actions: List[Dict[str, Any]]) -> str:
    """
    '액션 1개당 문장 1개' 정렬을 강제하는 프롬프트.
    """
    return f"""
You are an expert wet-lab protocol writer.

TASK:
Given a list of Action IR objects (JSON), produce a JSON array of natural-language step sentences.

STRICT REQUIREMENTS:
1) Output MUST be ONLY a JSON array of strings. No markdown, no comments, no extra text.
2) The number of sentences MUST equal the number of action objects.
3) Sentence i MUST correspond to action i (same order, 1-to-1 alignment).
4) Each sentence MUST be exactly ONE sentence (no semicolons that create multiple sentences; avoid "and then").
5) Use imperative style. Include materials/conditions ONLY if present in the action.
6) Do NOT invent any new materials, instruments, quantities, temperatures, times, or steps.

Action IR list:
{json.dumps(actions, ensure_ascii=False, indent=2)}
""".strip()


def call_writer(
        client: OpenAI,
        actions: List[Dict[str, Any]],
        model: str,
        temperature: float,
        max_retries: int = 2,
) -> Tuple[List[str], str]:
    """
    returns: (sentences, raw_text)
    """
    prompt = build_prompt(actions)

    for attempt in range(max_retries + 1):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return ONLY a JSON array of strings."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        raw = resp.choices[0].message.content or ""
        sents = parse_json_array_only(raw)

        # Validate: len match + all non-empty strings allowed (empty도 허용할지 선택)
        if sents and len(sents) == len(actions):
            return sents, raw
        # 재시도: 마지막에는 실패 반환
        if attempt == max_retries:
            return (["" for _ in actions], raw)

    return (["" for _ in actions], "")


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default="/Users/haeb/Workspaces/BioProtocol/BioProtocolAgent")
    ap.add_argument("--ablation_dir", type=str, default="data/ablation")
    ap.add_argument("--pattern", type=str, default="gen_actions_P*_10.jsonl")

    ap.add_argument("--out_dir", type=str, default="reports/llm_protocols")
    ap.add_argument("--model", type=str, default="gpt-4.1-mini")
    ap.add_argument("--temperature", type=float, default=0.2)

    # 긴 actions 때문에 프롬프트가 터질 수 있어 샘플링 옵션 제공(필요 시만)
    ap.add_argument("--max_actions", type=int, default=None, help="If set, truncate actions per protocol to this many.")
    args = ap.parse_args()

    # dotenv 로드 확인
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found. Put it in .env or export it in your shell.")

    root = Path(args.project_root)
    ablation_dir = root / args.ablation_dir
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI()

    gen_files = sorted(ablation_dir.glob(args.pattern))
    if not gen_files:
        raise FileNotFoundError(f"No files matched: {ablation_dir / args.pattern}")

    for gen_path in gen_files:
        mode = gen_path.stem.replace("gen_actions_", "").replace("_10", "")
        recs = load_jsonl(gen_path)

        out_rows = []
        for rec in recs:
            pid = rec.get("protocol_id")
            actions = rec.get("actions") or []
            if not pid:
                continue

            if args.max_actions is not None and len(actions) > args.max_actions:
                actions_in = actions[: args.max_actions]
            else:
                actions_in = actions

            sentences, raw = call_writer(
                client=client,
                actions=actions_in,
                model=args.model,
                temperature=args.temperature,
            )

            out_rows.append({
                "mode": mode,
                "protocol_id": pid,
                "num_actions": len(actions_in),
                "sentences": sentences,
                "protocol_text": "\n".join([s for s in sentences if s.strip()]),
                "llm_model": args.model,
                "temperature": args.temperature,
                "raw_response": raw,  # 디버깅/재현용(원하면 제거 가능)
            })

        out_path = out_dir / f"generated_{mode}.jsonl"
        save_jsonl(out_path, out_rows)
        print(f"[SAVE] {out_path}")

    print("✅ Done: generated protocols from gen_actions (writer-only).")


if __name__ == "__main__":
    main()
