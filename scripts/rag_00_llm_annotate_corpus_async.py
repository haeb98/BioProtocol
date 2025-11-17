#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast LLM annotator for corpus JSONL:
- 비동기 요청 (asyncio + AsyncOpenAI)
- 디스크 캐시(텍스트 SHA1 키)
- JSON 모드 응답, 실패시 재시도
- 입력 텍스트 클립(최대 N chars)
- 모드 2종: full(역할/라우팅/서술/근거), tags_only(라우팅 태그만)
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

FAST_PROMPT_TAGS = """You are a protocol curator.
Read TEXT and return STRICT JSON:
{
  "role": "protocol|guideline|datasheet|analysis|safety|troubleshooting|background",
  "router_hint": "recipe|measurement|analysis|safety|troubleshooting|general",
  "router_hint_tags": ["recipe","measurement"],   // 1~3 tags
  "confidence": 0.6,                               // 0..1
  "evidence": {
    "role_span": "...",
    "router_hint_span": "..."
  }
}
Only JSON. TEXT:
"""

FULL_PROMPT = """You are an expert protocol curator.
Read TEXT and return STRICT JSON:
{
  "role": "protocol|guideline|datasheet|analysis|safety|troubleshooting|background",
  "router_hint": "recipe|measurement|analysis|safety|troubleshooting|general",
  "router_hint_tags": ["recipe","measurement"],
  "confidence": 0.6,
  "inputs": [],
  "materials": [],
  "problem": "",
  "method": "",
  "innovation": "",
  "application": "",
  "description": "",
  "evidence": {
    "role_span": "...",
    "router_hint_span": "...",
    "inputs_span": []
  }
}
Only JSON. TEXT:
"""


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"): s = "\n".join(s.split("\n")[1:])
    if s.endswith("```"):   s = "\n".join(s.split("\n")[:-1])
    return s.strip()


async def annotate_one(client, rec, model, mode, max_chars, cache_dir, semaphore, max_retries=3):
    text = rec.get("text") or rec.get("protocol") or ""
    if not text.strip():
        return rec, False
    key = sha1(f"{mode}:{model}:{text[:max_chars]}")
    cpath = Path(cache_dir) / f"{key}.json"
    if cpath.exists():
        try:
            rec["llm_meta"] = json.loads(cpath.read_text(encoding="utf-8"))
            # 상위필드에 덮어쓰기
            rec["role"] = rec["llm_meta"].get("role", "protocol")
            rec.setdefault("meta", {})["router_hint"] = rec["llm_meta"].get("router_hint", "general")
            return rec, True
        except:
            pass

    prompt = (FAST_PROMPT_TAGS if mode == "tags_only" else FULL_PROMPT) + text[:max_chars]

    # 비동기 호출
    for t in range(max_retries):
        try:
            async with semaphore:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    # JSON 모드: 응답을 JSON 오브젝트로 강제
                    response_format={"type": "json_object"}  # 공식 지원, JSON mode.
                )
            out = resp.choices[0].message.content
            out = strip_fences(out)
            data = json.loads(out)
            # 최소 필드 보정
            data.setdefault("role", "background")
            data.setdefault("router_hint", "general")
            data.setdefault("router_hint_tags", [data.get("router_hint", "general")])
            data.setdefault("confidence", 0.6)
            data.setdefault("evidence", {})
            if mode != "tags_only":
                for k in ["inputs", "materials", "problem", "method", "innovation", "application", "description"]:
                    data.setdefault(k, [] if k in ["inputs", "materials"] else "")
            rec["llm_meta"] = data
            rec["role"] = data.get("role", "protocol")
            rec.setdefault("meta", {})["router_hint"] = data.get("router_hint", "general")
            # 캐시 저장
            cpath.parent.mkdir(parents=True, exist_ok=True)
            cpath.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            return rec, True
        except Exception as e:
            if t == max_retries - 1:
                return rec, False
            await asyncio.sleep(1.5 * (t + 1))


async def run(args):
    from openai import AsyncOpenAI
    client = AsyncOpenAI()  # 공식 SDK의 비동기 클라이언트

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir);
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 입력 전체를 메모리에 올리지 않고 스트리밍 처리
    sem = asyncio.Semaphore(args.concurrency)
    reader = open(args.input, "r", encoding="utf-8")
    writer = open(args.out, "w", encoding="utf-8")

    pending = []
    processed = 0

    async def submit(line):
        try:
            rec = json.loads(line)
        except:
            return None
        return await annotate_one(client, rec, args.model, args.mode, args.max_chars, cache_dir, sem)

    async def drain():
        nonlocal pending, processed
        if not pending: return
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            res = task.result()
            if res is None: continue
            rec, ok = res
            writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
            processed += 1

    # 메인 루프
    for line in reader:
        if len(pending) >= args.concurrency * 4:
            await drain()
        pending.add(asyncio.create_task(submit(line))) if isinstance(pending, set) else pending.append(
            asyncio.create_task(submit(line)))

    # 남은 작업 플러시
    while pending:
        await drain()

    writer.close();
    reader.close()
    print(f"[ANNOT] wrote -> {args.out} (processed={processed})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["tags_only", "full"], default="tags_only",
                    help="tags_only가 가장 빠름(라우팅/부스팅용). full은 서술 필드까지 채움.")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--max-chars", type=int, default=3500, help="텍스트 앞부분만 사용(토큰 절약).")
    ap.add_argument("--concurrency", type=int, default=16, help="동시 요청 수(요금제/레이트 한도 내에서 조절).")
    ap.add_argument("--cache-dir", default="data/cache/llm_annot")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
