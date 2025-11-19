#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM Task Miner (s2a_task_miner_00)
- Input : gold_pairs_testset.jsonl (from pmc_04 v2)
- Output: runs/s2a_tasks_llm.methods.jsonl (one task per line)

LLM이 a.sec_text 전체(또는 긴 경우 청크)를 읽고, 수행해야 할 실험 태스크를
{title, description, goal, key_materials} 중심 JSON으로 반환.
'본문에 없는 정보는 추론/환각하지 말 것'을 강하게 지시.

Requirements:
  pip install openai
  export OPENAI_API_KEY=...

References:
  - Chat Completions API & JSON mode: response_format={"type":"json_object"} :contentReference[oaicite:1]{index=1}
  - Structured outputs guidance (schema 지시와 병행): :contentReference[oaicite:2]{index=2}

LLM Task Miner (s2a_task_miner_00) — with --context mr
- Input : data/gold/gold_pairs_testset.jsonl
- Output: runs/s2a_tasks_llm.<context>.jsonl (one task per line)
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict

SENT_RX = re.compile(r"(?<=[\.\?\!])\s+")
RESULTS_TITLE_RX = re.compile(
    r"\b(results?|findings?|outcomes?|observations?|analysis)\b", re.I
)


def split_paragraphs(text: str) -> List[str]:
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    parts = [p.strip() for p in raw.split("\n\n")]
    return [p for p in parts if p]


def dedup_merge(tasks: List[Dict]) -> List[Dict]:
    norm = lambda s: re.sub(r"\s+", " ", (s or "").strip().lower())
    seen = {}
    for t in tasks:
        key = norm(t.get("title"))
        if not key:
            continue
        if key not in seen:
            seen[key] = t
        else:
            a = seen[key]
            km = set([*(a.get("key_materials") or []), *(t.get("key_materials") or [])])
            a["key_materials"] = sorted([x for x in km if x])[:20]
    return list(seen.values())


def build_system_prompt():
    """
    Construct the system prompt for the LLM task miner.

    The prompt instructs the model to extract a concise set of high‑level tasks
    that would allow a scientist to reproduce the experiment.  It emphasises
    staying within the information provided and avoiding over‑segmentation.

    Returns:
        A string to be used as the system message when calling the LLM.
    """
    return (
        "You are a meticulous lab protocol analyst. "
        "Read the provided scientific text and extract a concise list of experimental TASKS.\n\n"
        "STRICT RULES:\n"
        "1) DO NOT invent facts. If a field is not explicit in the text, set it to null.\n"
        "2) Each task MUST come from the provided text span(s) only and should correspond to a major phase of the experiment.\n"
        "3) Produce only as many tasks as necessary to reproduce the experiment (typically 4–12).\n"
        "   Do not break a single logical step into multiple tasks, and do not combine multiple unrelated steps into one task.\n"
        "4) Keep titles short (<= 12 words).\n"
        "5) key_materials: list 1–8 important reagents/media/buffers if explicitly mentioned; else [].\n"
        "6) description: 1–3 sentences summarizing what the task does.\n"
        "7) goal: a brief 'to + verb ...' purpose if present; else null.\n"
        "8) Return ONLY JSON with a top-level key 'tasks' containing an array of task objects."
    )


def call_llm_json(client, model: str, sys_prompt: str, user_payload: Dict,
                  temperature: float = 0.0, max_retries: int = 3) -> Dict:
    last_err = None
    for i in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            last_err = e
            time.sleep(1.2 * (i + 1))
    raise RuntimeError(f"LLM call failed: {last_err}")


def collect_results_text(rec: Dict) -> str:
    secs = (rec.get("article") or {}).get("sections") or {}
    picked = []
    for title, txt in secs.items():
        t = (title or "").strip()
        if t and RESULTS_TITLE_RX.search(t):
            if txt and txt.strip():
                picked.append(txt.strip())
    return "\n\n".join(picked)


def mine_tasks_llm_for_record(rec: Dict, client, model: str,
                              context: str, limit_chars: int, max_tasks_keep: int) -> List[Dict]:
    pid = rec.get("protocol_id")
    pmcid = rec.get("pmcid")
    section_list = rec.get("metrics", {}).get("section_list") or rec.get("article", {}).get("section_list") or []

    methods_text = rec.get("sec_text") or ""
    if context == "mr":
        results_text = collect_results_text(rec)
        if results_text:
            full_text = methods_text + "\n\n[RESULTS]\n\n" + results_text
        else:
            full_text = methods_text
    else:
        full_text = methods_text

    paras = split_paragraphs(full_text)
    if not paras:
        return []

    # chunk by ~12k chars
    chunks, cur, cur_len = [], [], 0
    for p in paras:
        if cur_len + len(p) + 2 > limit_chars and cur:
            chunks.append("\n\n".join(cur))
            cur, cur_len = [p], len(p)
        else:
            cur.append(p);
            cur_len += len(p) + 2
    if cur: chunks.append("\n\n".join(cur))

    sys_prompt = build_system_prompt()
    all_tasks = []
    for cidx, chunk in enumerate(chunks):
        payload = {"context": context, "chunk_index": cidx, "section_list": section_list, "text": chunk[:limit_chars]}
        js = call_llm_json(client, model, sys_prompt, payload, temperature=0.0, max_retries=3)
        tasks = js.get("tasks") or []
        for t in tasks:
            title = (t.get("title") or "").strip()
            desc = t.get("description");
            desc = (desc if isinstance(desc, str) and desc.strip() else None)
            goal = t.get("goal");
            goal = (goal if isinstance(goal, str) and goal.strip() else None)
            kms = t.get("key_materials") or []
            if isinstance(kms, str):
                kms = [x.strip() for x in re.split(r"[,\n;]", kms) if x.strip()]
            kms = [re.sub(r"\s+", " ", k).strip() for k in kms if k and isinstance(k, str)]

            all_tasks.append({
                "protocol_id": pid,
                "pmcid": pmcid,
                "title": title[:180] if title else "",
                "description": desc,
                "goal": goal,
                "key_materials": kms[:8],
                "section_list": section_list,
                "span_hint": f"{context}:chunk:{cidx}"
            })

    merged = dedup_merge([t for t in all_tasks if t.get("title")])
    out = []
    for i, t in enumerate(merged[:max_tasks_keep], start=1):
        t["task_id"] = f"{pid}::{i}"
        out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="data/gold/gold_pairs_testset.jsonl")
    ap.add_argument("--out", required=True, help="runs/s2a_tasks_llm.<context>.jsonl")
    ap.add_argument("--model", default="gpt-4.1-mini", help="OpenAI chat model")
    ap.add_argument("--context", choices=["methods", "mr"], default="methods",
                    help="methods: Methods only; mr: Methods + Results appended")
    ap.add_argument("--limit-chars", type=int, default=12000)
    ap.add_argument("--max-tasks-keep", type=int, default=8)
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("ERROR: OPENAI_API_KEY not set")

    from openai import OpenAI
    client = OpenAI()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(args.out, "w", encoding="utf-8") as w, open(args.pairs, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except:
                continue
            tasks = mine_tasks_llm_for_record(
                rec, client, args.model, args.context,
                limit_chars=args.limit_chars, max_tasks_keep=args.max_tasks_keep
            )
            for t in tasks:
                w.write(json.dumps(t, ensure_ascii=False) + "\n")
            total += len(tasks)
    print(f"[OK] mined tasks ({args.context}): {total} -> {args.out}")


if __name__ == "__main__":
    main()
